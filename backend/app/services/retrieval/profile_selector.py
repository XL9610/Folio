from typing import Literal

ProfileName = Literal["baseline", "keyword_heavy", "dense_heavy", "balanced"]

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "so", "than", "too", "very", "just",
    "about", "what", "which", "who", "whom", "this", "that", "these",
    "those", "it", "its", "and", "but", "or", "not", "no", "nor"
})


class ProfileSelector:
    """Deterministic rule-based selection of RetrievalProfile per query.

    Stateless. No LLM, no I/O. Selection depends only on planner task label,
    user query text, and document metadata passed in.
    """

    def select(
        self,
        task: str,
        query: str,
        document_metadata: list[dict],
    ) -> ProfileName:
        """Map (task, query, doc_metadata) → profile name.

        Selection rules (ordered, first match wins):
          1. explain_concept + specific terms → keyword_heavy
          2. compare + multi-doc → dense_heavy
          3. locate / source_overview → keyword_heavy
          4. default → balanced

        Args:
            task: Normalized planner label.
            query: Raw user query string.
            document_metadata: List of doc metadata dicts with 'structure_status', 'id'.

        Returns:
            Profile name string. Always returns a valid profile.
        """
        if not task:
            return "balanced"

        if task == "explain_concept" and self._has_specific_terms(query):
            return "keyword_heavy"

        if task == "compare" and len(document_metadata) > 1:
            return "dense_heavy"

        if task in {"locate", "source_overview"}:
            return "keyword_heavy"

        return "balanced"

    def _has_specific_terms(self, query: str) -> bool:
        """Detect high-specificity terms: proper nouns, technical terms, named entities.

        Heuristic: query contains ≥2 Title Case words OR ≥3 non-stopword terms (len > 2).
        """
        words = query.split()
        title_case = [
            w for w in words
            if w and w[0].isupper() and w.lower() not in _STOPWORDS
        ]
        non_stop = [
            w for w in words
            if w.lower() not in _STOPWORDS and len(w) > 2
        ]
        return len(title_case) >= 2 or len(non_stop) >= 3
