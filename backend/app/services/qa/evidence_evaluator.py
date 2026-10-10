"""Deterministic gate — only catches structural impossibilities.

Returns a definite EvidenceEval ONLY for cases where there is nothing
meaningful for the generator to work with:
  - No documents selected at all
  - All tool calls returned empty (no chunks, no outline, no metadata)

Semantic sufficiency (e.g., "do these chunks actually answer the query?")
is the generator's job. Synthesize-first prompt handles partial coverage,
off-topic chunks, and missing definitions without needing a pre-generation gate.

LLM-based semantic evaluation was removed (irreconcilable with the input it
received: 100-char snippets cannot support "is this sufficient?" judgments).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .agent_planner import PlanResult, SubQuestion
from .query_intent import OVERVIEW_QUERY_RE
from .tools import ToolResults

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class SubquestionEval:
    id: str = ""
    status: str = "pending"
    reason: str = ""
    evidence_count: int = 0


@dataclass
class EvidenceEval:
    """Result of structural guard check.

    Statuses:
        proceed     — no structural blocker; generator owns answer quality
        ambiguous   — missing documents or unclear user intent; clarify
        insufficient — zero evidence or confirmed section mismatch
        error       — retrieval tools raised errors; search unavailable
    """
    status: str = "proceed"
    reason: str = ""
    subquestion_evals: list[SubquestionEval] = field(default_factory=list)
    information_gaps: list[dict] = field(default_factory=list)
    clarifying_question: str = ""
    suggested_retrieval_query: str = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CHUNK_TEXT_LENGTH = 10
MIN_CHUNKS_FOR_CONTENT_ANSWER = 1

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "according",
    "book",
    "chapter",
    "chapters",
    "cite",
    "define",
    "did",
    "discuss",
    "discussed",
    "discussion",
    "does",
    "explain",
    "for",
    "from",
    "give",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "passage",
    "paper",
    "say",
    "says",
    "section",
    "source",
    "stand",
    "stands",
    "study",
    "tell",
    "the",
    "this",
    "to",
    "used",
    "what",
    "whats",
    "who",
    "won",
}

STRUCTURAL_QUERY_RE = re.compile(
    r"\b(?:chapter|section|heading|outline|toc|table\s+of\s+contents)\b.*"
    r"\b(?:title|titles|page|pages|start|starts|what\s+chapters)\b",
    re.IGNORECASE,
)
DEFINITION_QUERY_RE = re.compile(
    r"\b(?:what\s+is|what\s+are|define|definition\s+of|meaning\s+of|explain|describe)\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|\d+")


# ---------------------------------------------------------------------------
# Evaluator — deterministic gates only
# ---------------------------------------------------------------------------

class EvidenceEvaluator:
    """Deterministic structural guard for the ask pipeline.

    Only blocks structurally impossible cases. The generator receives
    full chunk context and is responsible for semantic grounding.
    """

    def __init__(self, llm_client):
        # llm_client kept for interface compatibility; NOT called
        self._llm = llm_client

    def evaluate(
        self,
        query: str,
        plan: PlanResult,
        tool_results: ToolResults,
        notebook_id: str,
        document_ids: list[str],
        follow_up: bool = False,
    ) -> EvidenceEval:
        """Run deterministic structural checks. Return proceed if none trigger."""
        det = self._deterministic_checks(query, plan, tool_results, document_ids)
        if det is not None:
            logger.info("Evidence guard (deterministic) → %s", det.status)
            return det

        # Follow-ups rewrite/deepen the prior answer (already in conversation
        # history) rather than requesting fresh facts, so the exact-term
        # off-topic gate does not apply.
        if follow_up:
            return EvidenceEval(
                status="proceed",
                reason="Follow-up: grounded in conversation history.",
            )

        semantic_gap = self._exact_signal_gap_check(query, tool_results)
        if semantic_gap is not None:
            logger.info("Evidence guard (exact-term) → %s", semantic_gap.status)
            return semantic_gap

        return EvidenceEval(
            status="proceed",
            reason="No structural blocker. Generator will answer from retrieved evidence.",
        )

    # ------------------------------------------------------------------
    # Deterministic checks
    # ------------------------------------------------------------------

    def _deterministic_checks(
        self,
        query: str,
        plan: PlanResult,
        tool_results: ToolResults,
        document_ids: list[str],
    ) -> EvidenceEval | None:
        """Return a definite EvidenceEval only for structural impossibilities.

        Only two cases:
          1. No documents selected → ambiguous (can't answer without a source)
          2. All tools returned empty → insufficient (nothing to work with)

        Everything else returns None — the generator owns answer quality.
        """
        has_chunks = bool(tool_results.chunks)
        has_outline = bool(tool_results.outline)
        has_metadata = bool(tool_results.metadata)
        has_section_matches = bool(tool_results.section_matches)
        has_web = bool(getattr(tool_results, "web_results", []))

        # No sources selected → ambiguous
        if not document_ids:
            return EvidenceEval(
                status="ambiguous",
                reason="No documents selected. User must select at least one source.",
                clarifying_question="Which document(s) would you like me to look at?",
            )

        # A body/search tool raised and no body evidence (chunks/web) came back:
        # retrieval is broken. Metadata, outline, and section labels are
        # deterministic DB reads that never hit the search backend, so they must
        # not mask the failure — an answer built only from a filename or TOC is
        # ungrounded.
        if getattr(tool_results, "tool_error", False) and not has_chunks and not has_web:
            return EvidenceEval(
                status="error",
                reason="Retrieval failed: search tools raised errors. The index may be unavailable.",
            )

        # Everything empty → insufficient (nothing to work with)
        if not has_chunks and not has_outline and not has_metadata and not has_section_matches and not has_web:
            return EvidenceEval(
                status="insufficient",
                reason="No evidence gathered at all. Tools returned no data.",
            )

        return None

    def _exact_signal_gap_check(
        self,
        query: str,
        tool_results: ToolResults,
    ) -> EvidenceEval | None:
        """Block obvious off-topic retrieval when exact high-signal terms vanish."""
        if not tool_results.chunks:
            return None
        if STRUCTURAL_QUERY_RE.search(query or ""):
            return None
        if OVERVIEW_QUERY_RE.search(query or ""):
            return None

        mandatory = _mandatory_query_terms(query)
        evidence = _evidence_text(tool_results)
        if mandatory:
            missing = [term for term in mandatory if not _contains_term(evidence, term)]
            if missing:
                return EvidenceEval(
                    status="insufficient",
                    reason=(
                        "Retrieved evidence is off-topic: high-signal query term(s) "
                        f"not found in gathered evidence: {', '.join(missing[:5])}."
                    ),
                )

        content_terms = _content_terms(query)
        if len(content_terms) >= 2:
            hits = sum(1 for term in content_terms if _contains_term(evidence, term))
            if hits == 0:
                return EvidenceEval(
                    status="insufficient",
                    reason=(
                        "Retrieved evidence is off-topic: none of the query's "
                        "content terms appear in gathered evidence."
                    ),
                )
            if _definition_like_query(query):
                units = _evidence_units(tool_results)
                max_unit_hits = _max_unit_term_hits(units, content_terms)
                has_adjacent_phrase = any(
                    _contains_phrase(unit, phrase)
                    for unit in units
                    for phrase in _content_bigrams(content_terms)
                )
                if len(content_terms) >= 3 and max_unit_hits < 2 and not has_adjacent_phrase:
                    return EvidenceEval(
                        status="insufficient",
                        reason=(
                            "Retrieved evidence is off-topic: query terms appear "
                            "only as isolated words, not as the requested concept."
                        ),
                    )
                if len(content_terms) == 2 and max_unit_hits < 2 and not has_adjacent_phrase:
                    return EvidenceEval(
                        status="insufficient",
                        reason=(
                            "Retrieved evidence is off-topic: the requested "
                            "two-word concept is not present in any evidence unit."
                        ),
                    )

        return None


def _content_terms(query: str) -> list[str]:
    terms: list[str] = []
    for raw in TOKEN_RE.findall(query or ""):
        term = raw.lower()
        if len(term) < 3 and not term.isdigit():
            continue
        if term in STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
    return terms


def _mandatory_query_terms(query: str) -> list[str]:
    terms: list[str] = []
    tokens = TOKEN_RE.findall(query or "")
    for idx, raw in enumerate(tokens):
        term = raw.lower()
        if term in STOPWORDS:
            continue
        if raw.isdigit() and len(raw) >= 4:
            terms.append(term)
            continue
        if raw.isupper() and len(raw) >= 2:
            terms.append(term)
            continue
        if idx > 0 and raw[:1].isupper() and len(raw) >= 4:
            terms.append(term)
    return list(dict.fromkeys(terms))


def _evidence_text(tool_results: ToolResults) -> str:
    return "\n".join(_evidence_units(tool_results)).lower()


def _evidence_units(tool_results: ToolResults) -> list[str]:
    parts: list[str] = []
    for chunk in tool_results.chunks:
        parts.append("\n".join([
            chunk.text or "",
            chunk.section_path or "",
            chunk.title or "",
        ]))
    for item in tool_results.outline:
        outline_parts = [
            str(item.get("filename", "")),
            str(item.get("structure_status", "")),
        ]
        for section in item.get("sections", []) or []:
            outline_parts.append(str(section.get("section_path", "")))
            outline_parts.append(str(section.get("title", "")))
        parts.append("\n".join(outline_parts))
    for item in tool_results.metadata:
        parts.append(str(item.get("filename", "")))
    for item in getattr(tool_results, "web_results", []):
        parts.append("\n".join([
            str(item.get("title", "")),
            str(item.get("content", "")),
        ]))
    return [part.lower() for part in parts if part]


def _contains_term(evidence: str, term: str) -> bool:
    escaped = re.escape(term.lower())
    if re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", evidence):
        return True
    if term.endswith("y") and len(term) > 3:
        plural = re.escape(term[:-1] + "ies")
        return bool(re.search(rf"(?<![a-z0-9]){plural}(?![a-z0-9])", evidence))
    if not term.endswith("s"):
        plural = re.escape(term + "s")
        return bool(re.search(rf"(?<![a-z0-9]){plural}(?![a-z0-9])", evidence))
    return False


def _definition_like_query(query: str) -> bool:
    return bool(DEFINITION_QUERY_RE.search(query or ""))


def _content_bigrams(content_terms: list[str]) -> list[str]:
    return [
        f"{left} {right}"
        for left, right in zip(content_terms, content_terms[1:])
    ]


def _contains_phrase(evidence: str, phrase: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", evidence.lower())
    return f" {phrase.lower()} " in f" {normalized} "


def _max_unit_term_hits(units: list[str], content_terms: list[str]) -> int:
    max_hits = 0
    for unit in units:
        hits = sum(1 for term in content_terms if _contains_term(unit, term))
        max_hits = max(max_hits, hits)
    return max_hits
