"""Shared query-intent detection — deterministic regexes used by planner + evaluator."""

from __future__ import annotations

import re

# Overview/summary intent — asks what the document as a whole is about, not a
# specific fact. Its "content terms" are generic (article, talk, findings) and
# won't literally appear in the body, so the exact-match gate must not treat
# their absence as off-topic, and the planner must fetch body chunks.
OVERVIEW_QUERY_RE = re.compile(
    r"\b(?:summar(?:y|ize|ise|ization)|overview|synopsis)\b"
    r"|\bmain\s+(?:point|finding|idea|topic|theme|argument|concept|conclusion|takeaway|claim)s?\b"
    r"|\bwhat\s+(?:is|are|does)\b[^?.!]{0,80}\b(?:article|paper|document|book|text|chapter|section|source|study)\b[^?.!]{0,50}\b(?:about|talk|say|discuss|cover|mean|contain)\b",
    re.IGNORECASE,
)


def is_overview_query(query: str) -> bool:
    """Return True if *query* asks for an overview/summary of the document."""
    return bool(OVERVIEW_QUERY_RE.search(query or ""))
