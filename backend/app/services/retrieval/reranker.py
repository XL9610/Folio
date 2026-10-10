"""Reranker — post-retrieval filtering, boosting, and section diversity.

Applied after the HybridRetriever returns results, before they're
passed to the AnswerGenerator.
"""

from __future__ import annotations

import logging
import re

from ..retrieval.base import RetrievedChunk
from ..retrieval.boost_config import BoostConfig

logger = logging.getLogger(__name__)


# ── Section role categories (detected from section_path) ──

_DOWNRANK_PATTERNS = {
    "exercises": re.compile(
        r'exercises|exercise|problem set|problems and applications|'
        r'questions for review|review questions|questions and problems|'
        r'problems$|^problems\b',
        re.IGNORECASE,
    ),
    "epilogue": re.compile(r'epilogue|afterword', re.IGNORECASE),
    "appendix": re.compile(r'^appendix|^annex', re.IGNORECASE),
    "references": re.compile(r'references|bibliography|further reading|works cited', re.IGNORECASE),
    "index": re.compile(r'^index$|^glossary$', re.IGNORECASE),
    "summary": re.compile(r'^summary$|^conclusion$|chapter summary|key terms', re.IGNORECASE),
}

_BOOST_PATTERNS = {
    "intro": re.compile(r'introduction|intro\b|getting started|overview', re.IGNORECASE),
    "fundamental": re.compile(r'fundamental|basic|primer|foundation', re.IGNORECASE),
    "definition": re.compile(r'definition|defining|abstract data type|adt\b', re.IGNORECASE),
}

_FRONT_MATTER_CONTENT_RE = re.compile(
    r'preface|foreword|introduction(?!\s+to\s+statistics)',
    re.IGNORECASE,
)

_FRONT_MATTER_META_RE = re.compile(
    r'acknowledgments?|about\s+the\s+author|colophon|copyright|dedication|title\s+page|toc|table\s+of\s+contents',
    re.IGNORECASE,
)

_QUERY_PREFIX_RE = re.compile(
    r"^(?:what\s+is\s+(?:a|an|the)?\s*|what\s+are\s+(?:the)?\s*|"
    r"define\s+|definition\s+of\s+|meaning\s+of\s+|explain\s+|describe\s+)",
    re.IGNORECASE,
)

_QUERY_STOPWORDS = {
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


def get_downrank_penalty(section_path: str) -> float:
    """Return a penalty factor (0.0–1.0) based on section role.

    1.0 = no penalty, 0.5 = 50% score reduction.
    """
    if not section_path:
        return 1.0

    for category, pattern in _DOWNRANK_PATTERNS.items():
        if pattern.search(section_path):
            if category == "exercises":
                return 0.1   # Near-remove for exercises/problem sets
            if category == "summary":
                return 0.2   # Summaries are not primary content
            if category == "epilogue":
                return 0.3
            if category == "appendix":
                return 0.5
            if category in ("references", "index"):
                return 0.3
    if _FRONT_MATTER_META_RE.search(section_path):
        return 0.3
    return 1.0


def get_boost_factor(section_path: str) -> float:
    """Return a boost factor (≥1.0) based on section role.

    1.0 = no boost, 1.3 = 30% score increase.
    """
    if not section_path:
        return 1.0

    for category, pattern in _BOOST_PATTERNS.items():
        if pattern.search(section_path):
            if category == "definition":
                return 1.3
            if category == "fundamental":
                return 1.2
            if category == "intro":
                return 1.15
    return 1.0


def get_front_matter_boost(section_path: str, boost_config: BoostConfig) -> float:
    """Boost content front matter when front_matter_boost flag is enabled.

    Returns:
        1.5 if front_matter_boost=True AND section_path matches content front matter
        1.0 otherwise
    """
    if not boost_config.front_matter_boost:
        return 1.0
    if not section_path:
        return 1.0
    if _FRONT_MATTER_CONTENT_RE.search(section_path.lower()):
        return 1.5
    return 1.0


def get_lead_paragraph_boost(chunk_page_start: int, boost_config: BoostConfig) -> float:
    """Boost chunks in the first N% of pages for low-structure documents.

    Returns:
        1.2 if lead_paragraph_boost=True AND chunk_page_start <= lead_page_threshold
        1.0 otherwise
    """
    if not boost_config.lead_paragraph_boost:
        return 1.0
    if boost_config.lead_page_threshold <= 0:
        return 1.0
    if chunk_page_start < 1:
        return 1.0
    if chunk_page_start <= boost_config.lead_page_threshold:
        return 1.2
    return 1.0


def get_query_match_boost(query: str, chunk: RetrievedChunk) -> float:
    """Boost chunks that contain the requested concept as a coherent phrase.

    This is intentionally lexical and source-agnostic: if the user asks
    "What is X?" and a chunk has an exact heading or phrase for X, it should
    outrank a generic passage that merely shares isolated query words.
    """
    phrases = _query_concept_phrases(query)
    if not phrases:
        return 1.0

    title_text = _normalize_for_phrase(
        " ".join([chunk.section_path or "", chunk.title or ""])
    )
    body_text = _normalize_for_phrase((chunk.text or "")[:1200])
    for phrase in phrases:
        normalized_phrase = _normalize_for_phrase(phrase)
        if not normalized_phrase:
            continue
        if _contains_normalized_phrase(title_text, normalized_phrase):
            return 2.5
        if _contains_normalized_phrase(body_text, normalized_phrase):
            return 2.0
    return 1.0


def clean_query(query: str) -> str:
    """Clean a user query before sending to the retriever.

    - Strip outer quotes ("What is X" → What is X)
    - Strip leading/trailing whitespace
    """
    q = query.strip()
    # Strip outer double/single quotes
    if (q.startswith('"') and q.endswith('"')) or (q.startswith("'") and q.endswith("'")):
        q = q[1:-1].strip()
    return q


def rewrite_for_definition(query: str) -> str:
    """Rewrite a concept-definition query for better retrieval.

    "What is a data structure" → "data structure definition fundamental introduction"
    """
    # Remove question words
    q = query.lower().strip()
    # Strip outer quotes
    q = clean_query(q)
    # Remove leading question patterns
    q = re.sub(r'^(what is a |what is an |what is |what are |define |explain |describe )', '', q, flags=re.IGNORECASE)
    # Add definition-oriented boost terms
    return f"{q} definition fundamental introduction"


def rerank_results(
    chunks: list[RetrievedChunk],
    downrank_roles: list[str] | None = None,
    boost_terms: list[str] | None = None,
    diversify_by_section: bool = False,
    query: str | None = None,
    boost_config: BoostConfig | None = None,
) -> list[RetrievedChunk]:
    """Apply post-retrieval reranking: downrank, boost, diversify.

    Args:
        chunks: Retrieved chunks from hybrid retriever.
        downrank_roles: Section role keywords to penalize.
            If None, applies all default downrank patterns.
            If empty list [], no downrank is applied.
        boost_terms: Terms to boost (not used directly; pattern-based).
            If None, applies default boost patterns.
            If empty list [], no boost is applied.
        diversify_by_section: If True, limit to max 1 chunk per section_path.
        boost_config: Optional BoostConfig for front matter / lead paragraph boosts.

    Returns:
        Reranked list of chunks.
    """
    if not chunks:
        return chunks

    _bc = boost_config if boost_config is not None else BoostConfig()

    # ── Determine whether to apply defaults ──
    apply_downrank = downrank_roles is None or len(downrank_roles) > 0
    apply_boost = boost_terms is None or len(boost_terms) > 0

    # ── 1. Apply downrank/boost to scores ──
    for chunk in chunks:
        if apply_downrank:
            penalty = get_downrank_penalty(chunk.section_path or "")
            chunk.rrf_score = chunk.rrf_score * penalty
        if apply_boost:
            boost = get_boost_factor(chunk.section_path or "")
            chunk.rrf_score = chunk.rrf_score * boost
        fm_boost = get_front_matter_boost(chunk.section_path or "", _bc)
        lead_boost = get_lead_paragraph_boost(chunk.page_start, _bc)
        chunk.rrf_score = chunk.rrf_score * fm_boost * lead_boost
        if query:
            chunk.rrf_score = chunk.rrf_score * get_query_match_boost(query, chunk)

    # ── 2. Re-sort by adjusted score ──
    chunks.sort(key=lambda c: c.rrf_score, reverse=True)

    # ── 3. Section diversity: keep only highest-scoring chunk per section ──
    if diversify_by_section:
        seen_sections: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            key = chunk.section_path or chunk.title or chunk.chunk_id
            # Normalize: just use first 2 components for diversity
            parts = key.split(" > ")
            broad_key = " > ".join(parts[:2]) if len(parts) > 1 else key
            if broad_key not in seen_sections or chunk.rrf_score > seen_sections[broad_key].rrf_score:
                seen_sections[broad_key] = chunk
        chunks = list(seen_sections.values())
        chunks.sort(key=lambda c: c.rrf_score, reverse=True)

    # ── 4. Deduplicate by chunk_id ──
    seen_ids: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.chunk_id not in seen_ids:
            seen_ids.add(chunk.chunk_id)
            deduped.append(chunk)

    return deduped


def _query_concept_phrases(query: str) -> list[str]:
    q = clean_query(query or "").lower()
    q = _QUERY_PREFIX_RE.sub("", q).strip()
    q = re.sub(r"\baccording\s+to\s+(?:the\s+)?(?:source|book|textbook)\b", "", q)
    q = re.sub(r"[^a-z0-9]+", " ", q).strip()
    if not q:
        return []

    terms = [
        term
        for term in q.split()
        if len(term) > 1 and term not in _QUERY_STOPWORDS
    ]
    phrases: list[str] = []
    if len(terms) >= 2:
        phrases.append(" ".join(terms))
        phrases.extend(f"{left} {right}" for left, right in zip(terms, terms[1:], strict=False))
    elif terms:
        phrases.append(terms[0])
    return list(dict.fromkeys(phrases))


def _normalize_for_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "
