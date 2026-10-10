"""Quiz generator — LLM creates multiple-choice questions from chunks."""

from __future__ import annotations

import json
import logging
from typing import Any

from .scope_resolver import ScopeResolver
from ..retrieval.base import RetrievedChunk
from ..qa.types import collect_book_citations
from ._title_utils import clean_filename
from .sampling import sample_body_chunks, is_body_chunk

logger = logging.getLogger(__name__)


QUIZ_SYSTEM_PROMPT = """You are creating a study quiz from textbook material. Generate multiple-choice questions that test understanding of the key concepts in the provided passages.

Rules:
1. Use only the provided passages — do not invent facts or concepts.
2. Each question must have exactly 4 choices (A, B, C, D).
3. Exactly one choice is correct. Mark it with the "answer" field (e.g. "B").
4. Provide a 1-2 sentence explanation of why the correct answer is right.
5. Include a "source" field (integer) referencing the passage number that supports the question.
6. The wrong choices should be plausible but clearly incorrect upon careful reading.
7. Aim for the requested count. Generate fewer if the material does not support that many distinct, meaningful questions — do not pad.

Output ONLY valid JSON in this exact shape:
{
  "questions": [
    {
      "question": "...",
      "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "B",
      "explanation": "...",
      "source": 1
    },
    ...
  ]
}

No extra text, no markdown, no code fences, just the JSON object."""


class QuizGenerator:
    """Generate multiple-choice quizzes from document chunks using LLM."""

    def __init__(self, llm_client, scope_resolver: ScopeResolver):
        self._llm = llm_client
        self._resolver = scope_resolver

    def generate(
        self,
        notebook_id: str,
        document_id: str,
        scope: str,
        count: int = 10,
        difficulty: str = "medium",
        doc_filename: str = "",
        requirements: str = "",
    ) -> dict[str, Any]:
        result = self._resolver.resolve(document_id, notebook_id, scope)
        chunks = result.chunks
        meta = result.metadata

        if not chunks:
            message = _unresolved_scope_message(meta)
            return {
                "title": self._make_title(scope or "", doc_filename),
                "message": message or "No relevant content was found in the selected scope.",
                "questions": [],
                "sources": [],
                "stats": {
                    "question_count": 0,
                    "requested_count": count,
                    "scope_layer": meta.get("layer", "unknown"),
                    "scope_resolved": meta.get("scope_resolved", True),
                    "unresolved_reason": message or meta.get("unresolved_reason", ""),
                },
            }

        # Filter to body chunks
        body_chunks = [c for c in chunks if is_body_chunk(c)]
        if not body_chunks:
            body_chunks = chunks

        # Sample chunks for LLM context
        source_chunks = sample_body_chunks(body_chunks, max_chunks=25)

        user_prompt = self._build_user_prompt(source_chunks, scope, doc_filename, count, difficulty, requirements)

        try:
            raw = self._llm.complete_chat(
                messages=[
                    {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=min(8192, 300 + count * 300),
            )
        except Exception as exc:
            logger.error("Quiz LLM call failed: %s", exc)
            raise

        questions_raw = self._parse_questions(raw)
        repair_attempted = False
        if 0 < len(questions_raw) < count:
            repair_attempted = True
            missing = count - len(questions_raw)
            repair_prompt = self._build_repair_prompt(
                source_chunks,
                scope,
                doc_filename,
                missing,
                difficulty,
                requirements,
                existing_questions=questions_raw,
            )
            try:
                repair_raw = self._llm.complete_chat(
                    messages=[
                        {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                        {"role": "user", "content": repair_prompt},
                    ],
                    max_tokens=min(8192, 300 + missing * 300),
                )
                questions_raw.extend(self._parse_questions(repair_raw))
            except Exception as exc:
                logger.warning("Quiz repair LLM call failed: %s", exc)

        if len(questions_raw) > count:
            questions_raw = questions_raw[:count]
        questions = self._enrich_questions(questions_raw, source_chunks)

        if not questions:
            raise RuntimeError(
                "LLM produced no valid quiz questions. The output may have been "
                "truncated or malformed. Try again, or reduce the requested count."
            )

        return {
            "title": self._make_title(scope or "", doc_filename),
            "questions": questions,
            "sources": collect_book_citations(questions, document_id, doc_filename),
            "stats": {
                "question_count": len(questions),
                "requested_count": count,
                "scope_layer": meta.get("layer", "unknown"),
                "scope": scope or "",
                "difficulty": difficulty,
                "repair_attempted": repair_attempted,
            },
        }

    def _build_user_prompt(
        self,
        chunks: list[RetrievedChunk],
        scope: str,
        doc_filename: str,
        count: int,
        difficulty: str,
        requirements: str = "",
    ) -> str:
        lines = [
            f"Document: {doc_filename or '(untitled)'}",
            f"Scope: {scope or 'whole document'}",
            f"Requested count: {count}",
            f"Difficulty: {difficulty}",
        ]
        if requirements:
            lines.append(f"Special instructions: {requirements}")
        lines.append("")
        lines.append("Source passages:")
        lines.append("")
        for i, c in enumerate(chunks, 1):
            sec = c.section_path or c.title or ""
            loc = f"p.{c.page_start}" if c.page_start > 0 else ""
            header = f"[{i}] {sec}" + (f" ({loc})" if loc else "")
            lines.append(header)
            lines.append(c.text[:1500])
            lines.append("---")
        lines.append("")
        lines.append(
            f"Generate exactly {count} questions in the JSON format specified, "
            "unless the provided passages truly cannot support that many distinct questions."
        )
        return "\n".join(lines)

    def _build_repair_prompt(
        self,
        chunks: list[RetrievedChunk],
        scope: str,
        doc_filename: str,
        missing: int,
        difficulty: str,
        requirements: str,
        existing_questions: list[dict],
    ) -> str:
        existing_prompts = [
            str(q.get("question", "")).strip()
            for q in existing_questions
            if q.get("question")
        ]
        lines = [
            "The previous quiz response produced too few valid questions.",
            f"Generate exactly {missing} additional questions unless the material is truly insufficient.",
            "Do not repeat or paraphrase these existing questions:",
        ]
        lines.extend(f"- {p}" for p in existing_prompts)
        lines.append("")
        lines.append(self._build_user_prompt(chunks, scope, doc_filename, missing, difficulty, requirements))
        return "\n".join(lines)

    def _parse_questions(self, raw: str) -> list[dict]:
        """Extract questions array from LLM output. Same fence-strip + truncation recovery as flashcards."""
        text = raw.strip()

        if text.startswith("```"):
            parts = text.split("```", 2)
            text = parts[1] if len(parts) >= 2 else text
            text = text.lstrip("json").strip()

        start = text.find("{")
        if start == -1:
            logger.warning("Quiz: no JSON object in response")
            return []

        end = text.rfind("}")
        if end > start:
            try:
                obj = json.loads(text[start : end + 1])
                questions = obj.get("questions", [])
                return [q for q in questions if isinstance(q, dict) and q.get("question") and q.get("choices")]
            except json.JSONDecodeError as e:
                logger.warning("Quiz JSON parse failed: %s. Attempting truncation recovery.", e)

        recovered = self._recover_truncated_questions(text[start:])
        if recovered:
            logger.info("Quiz: recovered %d questions via truncation repair", len(recovered))
        return recovered

    def _recover_truncated_questions(self, text: str) -> list[dict]:
        """Salvage complete question objects from truncated JSON. Same algorithm as flashcards."""
        arr_idx = text.find('"questions"')
        if arr_idx == -1:
            return []
        bracket = text.find("[", arr_idx)
        if bracket == -1:
            return []

        questions: list[dict] = []
        i = bracket + 1
        n = len(text)
        while i < n:
            while i < n and text[i] in " \t\n\r,":
                i += 1
            if i >= n or text[i] == "]":
                break
            if text[i] != "{":
                break

            depth = 0
            in_string = False
            escape = False
            obj_start = i
            while i < n:
                ch = text[i]
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif in_string:
                    if ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                i += 1

            if depth != 0:
                break

            chunk = text[obj_start:i]
            try:
                q = json.loads(chunk)
                if isinstance(q, dict) and q.get("question") and q.get("choices"):
                    questions.append(q)
            except json.JSONDecodeError:
                pass

        return questions

    def _enrich_questions(self, questions_raw: list[dict], chunks: list[RetrievedChunk]) -> list[dict]:
        """Map LLM's 'source' (passage number) back to chunk metadata."""
        enriched = []
        for q in questions_raw:
            src_idx = q.get("source")
            chunk = None
            if isinstance(src_idx, int) and 1 <= src_idx <= len(chunks):
                chunk = chunks[src_idx - 1]
            enriched.append({
                "question": str(q.get("question", "")).strip(),
                "choices": [str(c).strip() for c in (q.get("choices") or [])],
                "answer": str(q.get("answer", "")).strip().upper(),
                "explanation": str(q.get("explanation", "")).strip(),
                "section_path": chunk.section_path if chunk else "",
                "page_start": chunk.page_start if chunk else -1,
                "page_end": chunk.page_end if chunk else -1,
                "source_chunk_id": chunk.chunk_id if chunk else "",
            })
        return enriched

    @staticmethod
    def _make_title(scope: str, doc_filename: str) -> str:
        base = clean_filename(doc_filename)
        if not scope:
            return f"Quiz: {base}"
        return f"Quiz: {base} — {scope}"


def _unresolved_scope_message(meta: dict) -> str:
    if meta.get("scope_resolved", True) is not False:
        return ""
    return (
        meta.get("unresolved_reason")
        or "The requested chapter or section could not be matched. Use a page range or topic keyword instead."
    )
