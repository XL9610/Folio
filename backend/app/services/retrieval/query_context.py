import re
from dataclasses import dataclass

_CHAPTER_REF_RE: re.Pattern = re.compile(r'\bChapter\s+(\d+)\b', re.IGNORECASE)
_CHAPTER_ABBREV_RE: re.Pattern = re.compile(r'\bCh\.\s*(\d+)\b', re.IGNORECASE)
_FRONT_MATTER_KEYWORDS: frozenset[str] = frozenset({
    "preface", "foreword", "introduction", "intended audience", "preamble"
})
_LOW_STRUCTURE_STATUSES: frozenset[str] = frozenset({"no-TOC", "low"})

_QUESTION_PREFIXES: list[str] = [
    "what is", "what are", "what does", "what do",
    "explain", "define", "describe", "tell me about",
    "what is the definition of", "what is an",
]

_STOPWORDS_SMALL = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "and",
    "or", "not", "no", "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "why", "when", "where",
})


@dataclass(frozen=True)
class QueryContext:
    """Pre-computed query metadata used by QAToolExecutor, QueryExpander,
    ProfileSelector, and reranker.

    Computed once per query, not per document. All fields are immutable.
    """
    has_chapter_ref: bool = False
    chapter_numbers: tuple[int, ...] = ()
    resolved_titles: tuple[str, ...] = ()
    has_front_matter: bool = False
    concept_phrase: str = ""
    is_low_structure: bool = False


def build_query_context(
    user_query: str,
    planner_task: str = "",
    document_metadata: list[dict] | None = None,
    resolved_chapters: dict[int, str] | None = None,
) -> QueryContext:
    """Build QueryContext from user query, planner output, and document metadata.

    Args:
        user_query: Raw user query string.
        planner_task: Normalized planner label (e.g. "explain_concept", "locate").
        document_metadata: List of doc metadata dicts with 'structure_status', 'page_count', 'id'.
        resolved_chapters: Map of chapter number → resolved title.

    Returns:
        Immutable QueryContext.
    """
    if document_metadata is None:
        document_metadata = []
    if resolved_chapters is None:
        resolved_chapters = {}

    chapter_numbers: list[int] = []
    for match in _CHAPTER_REF_RE.finditer(user_query):
        chapter_numbers.append(int(match.group(1)))
    if not chapter_numbers:
        for match in _CHAPTER_ABBREV_RE.finditer(user_query):
            chapter_numbers.append(int(match.group(1)))
    seen: set[int] = set()
    unique_chapters: list[int] = []
    for n in chapter_numbers:
        if n not in seen:
            seen.add(n)
            unique_chapters.append(n)

    has_chapter_ref = len(unique_chapters) > 0

    resolved_titles_list: list[str] = []
    for ch_num in unique_chapters:
        title = resolved_chapters.get(ch_num)
        if title:
            resolved_titles_list.append(title)

    query_lower = user_query.lower()
    has_front_matter = any(kw in query_lower for kw in _FRONT_MATTER_KEYWORDS)

    concept_phrase = ""
    if planner_task == "explain_concept":
        concept_phrase = _extract_concept_phrase(user_query)

    is_low_structure = any(
        m.get("structure_status") in _LOW_STRUCTURE_STATUSES
        for m in document_metadata
    )

    return QueryContext(
        has_chapter_ref=has_chapter_ref,
        chapter_numbers=tuple(unique_chapters),
        resolved_titles=tuple(resolved_titles_list),
        has_front_matter=has_front_matter,
        concept_phrase=concept_phrase,
        is_low_structure=is_low_structure,
    )


def _extract_concept_phrase(query: str) -> str:
    """Extract the main noun phrase from an explain_concept query.

    e.g. 'What is opportunity cost?' → 'opportunity cost'
    Strips question prefixes, keeps the longest non-stopword sequence.
    """
    cleaned = query.strip().rstrip("?").strip()
    cleaned_lower = cleaned.lower()

    for prefix in _QUESTION_PREFIXES:
        if cleaned_lower.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break

    cleaned = re.sub(r'^(what|who|when|where|why|how|is|are|does|do)\s+', '', cleaned, flags=re.IGNORECASE).strip()

    words = cleaned.split()
    content_words = [w for w in words if w.lower() not in _STOPWORDS_SMALL and len(w) > 1]

    if not content_words:
        return ""

    return " ".join(content_words)
