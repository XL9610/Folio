"""Source-framing answer enrichment.

After the LLM generates an answer, surface the explicit questions the source
frames a concept with (e.g. a textbook's "What is X?" scaffolding) under the
matching heading, so the answer inherits the source's framing.
"""

from __future__ import annotations

import re

from .tools import ToolResults


_CONCEPT_QUERY_PREFIX_RE = re.compile(
    r"^(?:what\s+is\s+(?:a|an|the)?\s*|what\s+are\s+(?:the)?\s*|"
    r"define\s+|definition\s+of\s+|meaning\s+of\s+|explain\s+|describe\s+)",
    re.IGNORECASE,
)
_CONCEPT_STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "according",
    "book",
    "chapter",
    "does",
    "source",
    "the",
    "this",
    "to",
    "what",
}


def _with_source_framing_questions(
    query: str,
    tool_results: ToolResults,
    answer_text: str,
) -> str:
    """Preserve explicit source questions under a matching concept heading."""
    concept = _concept_phrase_from_query(query)
    if not concept:
        return answer_text

    normalized_answer = _normalize_for_phrase(answer_text)
    refs = _citation_ref_by_chunk(tool_results)
    for chunk in tool_results.chunks[:12]:
        ref_label = refs.get(chunk.chunk_id)
        if not ref_label:
            continue
        if not _chunk_contains_concept(chunk, concept):
            continue
        questions = _extract_questions_after_concept(chunk.text or "", concept)
        if not questions:
            continue
        missing = [
            question for question in questions[:3]
            if _normalize_for_phrase(question) not in normalized_answer
        ]
        if not missing:
            return answer_text
        addition = (
            "The source frames this with explicit questions: "
            + "; ".join(f"{question} [{ref_label}]" for question in missing)
            + "."
        )
        if addition in answer_text:
            return answer_text
        return answer_text.rstrip() + "\n\n" + addition

    return answer_text


def _concept_phrase_from_query(query: str) -> str:
    q = (query or "").strip().lower()
    q = _CONCEPT_QUERY_PREFIX_RE.sub("", q).strip()
    q = re.sub(r"\baccording\s+to\s+(?:the\s+)?(?:source|book|textbook)\b", "", q)
    q = re.sub(r"[^a-z0-9]+", " ", q).strip()
    terms = [
        term
        for term in q.split()
        if len(term) > 1 and term not in _CONCEPT_STOPWORDS
    ]
    if len(terms) < 2:
        return ""
    return " ".join(terms)


def _normalize_for_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _chunk_contains_concept(chunk, concept: str) -> bool:
    phrase = _normalize_for_phrase(concept)
    haystacks = [
        chunk.section_path or "",
        chunk.title or "",
        (chunk.text or "")[:5000],
    ]
    return any(
        f" {phrase} " in f" {_normalize_for_phrase(text)} "
        for text in haystacks
    )


def _extract_questions_after_concept(text: str, concept: str) -> list[str]:
    normalized_text = _normalize_for_phrase(text)
    normalized_concept = _normalize_for_phrase(concept)
    concept_idx = normalized_text.find(normalized_concept)
    if concept_idx < 0:
        return []

    raw_text = re.sub(r"[ \t]+", " ", text or "")
    raw_lower = raw_text.lower()
    raw_idx = raw_lower.find(concept.lower())
    if raw_idx < 0:
        raw_idx = 0
    window = raw_text[raw_idx: raw_idx + 1600]
    questions: list[str] = []
    for match in re.finditer(r"([^?\n]{3,180}\?)", window):
        question = re.sub(r"\s+", " ", match.group(1)).strip(" -;:\u2022")
        if not question:
            continue
        if question not in questions:
            questions.append(question)
        if len(questions) >= 3:
            break
    return questions


def _citation_ref_by_chunk(tool_results: ToolResults) -> dict[str, str]:
    refs: dict[str, str] = {}
    for citation in tool_results.chunk_citations:
        chunk_id = str(citation.get("chunk_id", ""))
        ref_label = str(citation.get("ref_label", ""))
        if chunk_id and ref_label and chunk_id not in refs:
            refs[chunk_id] = ref_label
    return refs
