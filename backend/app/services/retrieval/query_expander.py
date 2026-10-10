"""Deterministic query expansion — zero LLM calls, zero network, zero file I/O.

Expands user queries by injecting:
  - High-signal fragments (quoted phrases, Title Case terms, chapter references)
  - Keyword n-grams (2-4 word non-stopword phrases)
  - Task-specific bonus keywords (explain_concept, compare, locate)
  - Concepts validated against sub-question text

The expanded query is the user's original query + injected bonus terms,
guaranteed not to exceed 500 characters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# English stopwords — common function words excluded from keyword extraction
# ---------------------------------------------------------------------------

STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "would", "could", "may", "might", "must", "shall",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "if", "as", "because", "until", "while", "also",
}

# ---------------------------------------------------------------------------
# Precompiled regex patterns
# ---------------------------------------------------------------------------

_QUOTED_RE = re.compile(r'"([^"]*)"')
_CURLY_QUOTED_RE = re.compile(r"“([^”]*)”")  # "smart" quotes
_CHAPTER_RE = re.compile(r"\b(Chapter|Ch\.)\s*\d+\b", re.IGNORECASE)
_TITLE_CASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_WORD_RE = re.compile(r"\b[a-zA-Z]+\b")

_QUESTION_PREFIXES: list[str] = [
    "what is", "what are", "what does", "what do",
    "explain", "define", "describe", "tell me about",
    "what is the definition of", "what is an",
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ExpandedQuery:
    """Result of query expansion.

    Attributes:
        original_query: The user's query exactly as provided.
        planner_query: The planner-rewritten query (may be empty).
        expanded_query: Full expanded query (original + injected terms), <= 500 chars.
        injected_terms: Comma-separated list of terms injected beyond the original.
    """

    original_query: str
    planner_query: str
    expanded_query: str
    injected_terms: str  # comma-separated


# ---------------------------------------------------------------------------
# QueryExpander
# ---------------------------------------------------------------------------


class QueryExpander:
    """Deterministic query expansion — zero LLM calls.

    Injects bonus keywords without any network, filesystem, or LLM access.
    Completely deterministic: same inputs always produce the same outputs.

    Usage::

        expander = QueryExpander()
        result = expander.expand(
            user_query='what is "opportunity cost"',
            plan_task="explain_concept",
            sub_questions=["What is the definition of opportunity cost?"],
        )
        print(result.expanded_query)
    """

    # Task → bonus keyword mapping
    _TASK_KEYWORDS: dict[str, list[str]] = {
        "explain_concept": [
            "definition", "overview", "introduction", "characteristics",
        ],
        "compare": [
            "difference", "similarity", "comparison", "contrast", "versus",
        ],
        "locate": [
            "section", "chapter", "heading", "page",
        ],
        "specific_search": [],  # rely entirely on user query
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(
        self,
        user_query: str,
        planner_query: str = "",
        plan_task: str = "",
        sub_questions: list[str] | None = None,
        section_hints: list[str] | None = None,
    ) -> ExpandedQuery:
        """Expand a user query with deterministic keyword injection.

        Args:
            user_query: The original user query string.
            planner_query: Optional planner-rewritten version of the query.
            plan_task: Optional task type (explain_concept, compare, locate, etc.).
            sub_questions: Optional list of sub-questions extracted by the planner.
            section_hints: Optional list of section heading hints.

        Returns:
            ExpandedQuery with the full expanded query (<= 500 chars).
        """
        user_query = (user_query or "").strip()
        planner_query = (planner_query or "").strip()
        sub_questions = sub_questions or []
        section_hints = section_hints or []

        # Guard: empty query
        if not user_query:
            return ExpandedQuery(
                original_query="",
                planner_query=planner_query,
                expanded_query="",
                injected_terms="",
            )

        bonus_parts: list[str] = []
        injected_parts: list[str] = []

        # --- Section hints active (BEFORE other expansions) ---
        if section_hints:
            bonus_parts.append(" ".join(section_hints))
            injected_parts.append(" ".join(section_hints))

        # --- Front matter keyword injection ---
        if self._has_front_matter_keywords(user_query):
            fm_kw = "preface foreword introduction front_matter"
            bonus_parts.append(fm_kw)
            injected_parts.append(fm_kw)

        # --- Concept phrase quoting ---
        if plan_task == "explain_concept":
            phrase = self._extract_concept_phrase(user_query)
            if phrase:
                quoted = f'"{phrase}"'
                bonus_parts.append(quoted)
                injected_parts.append(quoted)

        # --- High-signal fragments from user query ---
        high = self._extract_high_signal(user_query)
        bonus_parts.extend(high)

        # --- Keyword n-grams from user query ---
        bonus_parts.extend(self._extract_keywords(user_query))

        # --- Keyword n-grams from planner query (injected) ---
        if planner_query:
            pk = self._extract_keywords(planner_query)
            bonus_parts.extend(pk)
            injected_parts.extend(pk)

        # --- Task bonus keywords (injected) ---
        if plan_task:
            tk = self._task_keywords(plan_task)
            bonus_parts.extend(tk)
            injected_parts.extend(tk)

        # --- Validated concepts from sub-questions (injected) ---
        for sq in sub_questions:
            concepts = self._validated_concepts(sq)
            bonus_parts.extend(concepts)
            injected_parts.extend(concepts)

        # --- Merge and build the final query ---
        extra = self._merge(bonus_parts)

        expanded = f"{user_query} {extra}" if extra else user_query

        # Hard cap at 500 characters
        expanded = self._cap(expanded, limit=500)

        # Build comma-separated injected terms (deduplicated, order-preserved)
        injected_str = ",".join(dict.fromkeys(injected_parts))

        return ExpandedQuery(
            original_query=user_query,
            planner_query=planner_query,
            expanded_query=expanded,
            injected_terms=injected_str,
        )

    # ------------------------------------------------------------------
    # High-signal extraction
    # ------------------------------------------------------------------

    def _extract_high_signal(self, user_query: str) -> list[str]:
        """Extract quoted phrases, Title Case terms, and chapter references.

        Args:
            user_query: The raw user query string.

        Returns:
            List of high-signal term strings.
        """
        signals: list[str] = []

        # 1. Quoted phrases (straight and smart quotes)
        for pattern in (_QUOTED_RE, _CURLY_QUOTED_RE):
            for match in pattern.finditer(user_query):
                phrase = match.group(1).strip()
                if phrase and len(phrase.split()) >= 1:
                    signals.append(phrase.lower())

        # 2. Chapter references: "Chapter 20", "Ch. 5"
        for match in _CHAPTER_RE.finditer(user_query):
            signals.append(match.group(0).lower())

        # 3. Title Case terms: 2+ capitalized words in sequence
        for match in _TITLE_CASE_RE.finditer(user_query):
            term = match.group(1)
            words = term.split()
            # Only keep if at least one word is a non-stopword content word
            if any(w.lower() not in STOPWORDS and len(w) > 2 for w in words):
                signals.append(term.lower())

        return signals

    # ------------------------------------------------------------------
    # Keyword n-gram extraction
    # ------------------------------------------------------------------

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract 2-4 word non-stopword n-grams from *text*.

        Args:
            text: Input text (user query, planner query, etc.).

        Returns:
            List of n-gram strings (space-separated words).
        """
        if not text or not text.strip():
            return []

        # Tokenize and normalize
        words = _WORD_RE.findall(text.lower())

        # Remove stopwords
        content = [w for w in words if w not in STOPWORDS]

        # With fewer than 2 content words, return single words
        if len(content) < 2:
            return content

        # Build 2- to 4-word n-grams
        ngrams: list[str] = []
        max_n = min(4, len(content))
        for n in range(2, max_n + 1):
            for i in range(len(content) - n + 1):
                ngrams.append(" ".join(content[i:i + n]))

        return ngrams

    # ------------------------------------------------------------------
    # Task keyword injection
    # ------------------------------------------------------------------

    def _task_keywords(self, task: str) -> list[str]:
        """Return the bonus keyword list for *task*, or empty list if unknown."""
        return list(self._TASK_KEYWORDS.get(task, []))

    # ------------------------------------------------------------------
    # Concept validation from sub-questions
    # ------------------------------------------------------------------

    def _validated_concepts(self, sub_question: str) -> list[str]:
        """Extract content words from *sub_question* as validated concepts.

        Concepts are validated by construction: since every word is extracted
        directly from the sub-question text, it guaranteed appears there.

        Args:
            sub_question: A single sub-question string.

        Returns:
            List of content-word concepts (non-stopwords, >2 chars).
        """
        if not sub_question or not sub_question.strip():
            return []

        text = sub_question.lower()

        # Extract all words
        words = _WORD_RE.findall(text)

        # Filter: non-stopword, longer than 2 characters
        concepts = [
            w for w in words
            if w not in STOPWORDS and len(w) > 2
        ]

        return concepts

    # ------------------------------------------------------------------
    # Merge & cap
    # ------------------------------------------------------------------

    def _merge(self, parts: list[str]) -> str:
        """Deduplicate *parts* (case-insensitive) and join with spaces.

        The first occurrence of each unique (lowercased) part is kept,
        preserving its original casing.

        Args:
            parts: List of term/phrase strings.

        Returns:
            Space-joined, deduplicated string.
        """
        seen: set[str] = set()
        unique: list[str] = []

        for part in parts:
            key = part.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(part.strip())

        return " ".join(unique)

    def _cap(self, text: str, limit: int = 500) -> str:
        """Truncate *text* to *limit* characters, breaking at a word boundary.

        Args:
            text: The text to cap.
            limit: Maximum character length (default 500).

        Returns:
            Truncated text, ending on a complete word.
        """
        if len(text) <= limit:
            return text
        # Cut at the last word boundary within the limit
        truncated = text[:limit]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            return truncated[:last_space]
        return truncated

    # ------------------------------------------------------------------
    # Front matter / concept detection (Module B)
    # ------------------------------------------------------------------

    @staticmethod
    def _has_front_matter_keywords(query: str) -> bool:
        query_lower = query.lower()
        _keywords = frozenset({"preface", "foreword", "introduction", "intended audience", "preamble"})
        return any(kw in query_lower for kw in _keywords)

    @staticmethod
    def _extract_concept_phrase(query: str) -> str:
        cleaned = query.strip().rstrip("?").strip()
        cleaned_lower = cleaned.lower()

        for prefix in _QUESTION_PREFIXES:
            if cleaned_lower.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        cleaned = re.sub(
            r'^(what|who|when|where|why|how|is|are|does|do)\s+',
            '', cleaned, flags=re.IGNORECASE,
        ).strip()

        small_stop = frozenset({
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "and",
            "or", "not", "no", "it", "its", "this", "that", "these", "those",
            "what", "which", "who", "whom", "how", "why", "when", "where",
        })
        words = cleaned.split()
        content_words = [w for w in words if w.lower() not in small_stop and len(w) > 1]
        if not content_words:
            return ""
        return " ".join(content_words)
