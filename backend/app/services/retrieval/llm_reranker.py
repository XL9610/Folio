"""LLM pointwise reranker.

After hybrid retrieval, score each candidate chunk against the query
using an LLM and re-sort. Improves ranking quality when query/document
vocabulary diverges or when RRF fusion misorders relevant chunks.

Pointwise design: each (query, chunk) pair scored independently 0-100.
Adapted from facebookresearch/sira/scripts/configs/rerank/prompts/relevance_v04.txt.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .base import RetrievedChunk

logger = logging.getLogger(__name__)


RERANK_PROMPT = """Given a study question and a textbook passage, rate how useful the passage is for answering the question.

Question: {query}

Passage: {document}

Before scoring, follow these steps:
1. What is the question actually asking? Identify the specific information needed (a definition, a comparison, a procedure, an example, a date, a name, a list).
2. Scan the passage sentence by sentence: does ANY sentence address what the question asks? Even one sentence in a long passage counts.
3. If the passage is from a chapter or section explicitly about the question's main topic (e.g., the question asks about Gregorian chant and the passage IS from a section on Gregorian chant), score at least 41 — passages from on-topic sections almost always carry useful information, even if they don't contain a literal textbook-style definition.
4. If the passage describes an instance or example of the concept the user asked about (e.g., the question asks "what is a data structure" and the passage describes arrays, trees, or B-trees), score at least 41 — instances ARE valid evidence of a concept.

Rate usefulness from 0 to 100:
- 0: completely unrelated topic, shares no entities or concepts with the question
- 1-20: peripherally related but no sentence addresses the question
- 21-40: discusses related concepts without containing useful information
- 41-60: from a section about the question's topic, OR contains partial/contextual information, OR describes an instance of the asked-about concept
- 61-80: contains substantive information that addresses the question, even if only in part of the passage
- 81-100: directly and clearly addresses the question

IMPORTANT: Score based on whether the passage contributes USEFUL information to a synthesized answer, not whether it contains a textbook-style definition. A passage about array operations contributes to "what is a data structure" because arrays are data structures. A passage from "Characteristics of Chant" contributes to "what is Gregorian chant" because it describes properties of chant.

IMPORTANT: A score of 0 means the passage has NOTHING to do with the question — no shared concepts, entities, or topic relevance. If the passage discusses ANY entity or concept from the question, the score must be above 0.

Output ONLY a JSON object: {{"score": <integer 0-100>}}"""


@dataclass
class RerankResult:
    chunk: RetrievedChunk
    llm_score: float
    parse_failed: bool = False


class LLMReranker:
    """Pointwise rerank top-N chunks using LLM relevance scoring.

    Usage:
        reranker = LLMReranker(llm_client)
        reranked = reranker.rerank(query, chunks, top_n=20)
        # reranked[0] is now the highest-scoring chunk per LLM judgment

    Failure handling:
        - Single-chunk parse/timeout failure: keep RRF rank, score=0
        - All chunks fail: return original order unchanged
    """

    def __init__(
        self,
        llm_client,
        max_concurrency: int = 10,
        max_doc_chars: int = 2000,
        max_tokens: int = 64,
    ):
        self._llm = llm_client
        self._concurrency = max(1, max_concurrency)  # guard against invalid values
        self._max_doc_chars = max_doc_chars
        self._max_tokens = max_tokens

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int = 20,
    ) -> list[RetrievedChunk]:
        """Score chunks via LLM and return re-sorted list.

        Only the first `top_n` are reranked; chunks beyond that are appended
        in their original RRF order to preserve recall.
        """
        if not chunks or top_n <= 0:
            return chunks

        head = chunks[:top_n]
        tail = chunks[top_n:]

        results = self._score_batch(query, head)

        # All failed — keep original order
        if all(r.parse_failed for r in results):
            logger.warning(
                "Reranker: all %d chunks failed scoring, keeping RRF order",
                len(head),
            )
            return chunks

        # Sort by LLM score desc; ties broken by RRF score desc
        results.sort(key=lambda r: (-r.llm_score, -r.chunk.rrf_score))

        reranked = [r.chunk for r in results]
        # Stamp llm_score onto the chunk for downstream debugging/UI
        for r in results:
            setattr(r.chunk, "llm_rerank_score", r.llm_score)

        return reranked + tail

    def _score_batch(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RerankResult]:
        results: list[RerankResult | None] = [None] * len(chunks)

        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            futures = {
                pool.submit(self._score_one, query, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as exc:
                    logger.warning(
                        "Reranker scoring failed for chunk %d: %s", i, exc
                    )
                    results[i] = RerankResult(
                        chunk=chunks[i], llm_score=0.0, parse_failed=True
                    )

        return [r for r in results if r is not None]

    def _score_one(self, query: str, chunk: RetrievedChunk) -> RerankResult:
        doc_text = chunk.text[: self._max_doc_chars]
        prompt = RERANK_PROMPT.format(query=query, document=doc_text)

        try:
            raw = self._llm.complete_chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            logger.warning("LLM rerank call failed: %s", exc)
            return RerankResult(chunk=chunk, llm_score=0.0, parse_failed=True)

        score = self._parse_score(raw)
        if score is None:
            return RerankResult(chunk=chunk, llm_score=0.0, parse_failed=True)
        return RerankResult(chunk=chunk, llm_score=float(score))

    @staticmethod
    def _parse_score(raw: str) -> float | None:
        """Extract a 0-100 score from LLM output. Robust to extra text."""
        # Try JSON object first
        idx = raw.find("{")
        if idx >= 0:
            end = raw.rfind("}")
            if end > idx:
                try:
                    obj = json.loads(raw[idx : end + 1])
                    if isinstance(obj, dict) and "score" in obj:
                        s = float(obj["score"])
                        return max(0.0, min(100.0, s))
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
        # Fallback: first number in output
        m = re.search(r"\b(\d{1,3}(?:\.\d+)?)\b", raw)
        if m:
            s = float(m.group(1))
            if 0 <= s <= 100:
                return s
        return None
