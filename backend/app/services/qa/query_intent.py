"""Shared query-intent detection — deterministic regexes used by planner + evaluator."""

from __future__ import annotations

import re

# Overview/summary intent — asks what the document as a whole is about, not a
# specific fact. Its "content terms" are generic (article, talk, findings) and
# won't literally appear in the body, so the exact-match gate must not treat
# their absence as off-topic, and the planner must fetch body chunks.
#
# Broadened for Branch C (roadmap Phase 2): magazine/slide-deck coverage
# phrasings ("main stories covered in this issue", "what does Module 6
# cover") previously fell through to the similarity path.
OVERVIEW_QUERY_RE = re.compile(
    r"\b(?:summar(?:y|ize|ise|ization)|overview|synopsis)\b"
    r"|\bmain\s+(?:point|finding|idea|topic|theme|argument|concept|conclusion|takeaway|claim|stor(?:y|ies)|subject)s?\b"
    r"|\bkey\s+(?:point|finding|idea|topic|theme|concept|takeaway)s?\b"
    r"|\bwhat\s+(?:is|are|does)\b[^?.!]{0,80}\b(?:article|paper|document|book|text|chapter|section|source|study|issue|magazine|module|deck|slide(?:s|deck)?|presentation)\b[^?.!]{0,50}\b(?:about|talk|say|discuss|cover|mean|contain)\b",
    re.IGNORECASE,
)

# Compositional whole-source language that does not necessarily use the words
# "summary" or "overview".  Keep these as source-agnostic scope markers rather
# than enumerating evaluation questions or document titles.  This deliberately
# does not match a bare "high-level" modifier (for example, "explain binary
# search at a high level") unless it modifies an overview-shaped noun.
_HOLISTIC_DESCRIPTION_RE = re.compile(
    r"\b(?:high[-\s]level|overall|broad)\s+"
    r"(?:account|view|picture|description|outline|survey|treatment|perspective|"
    r"project|case)\b"
    r"|\b(?:book|work|novel|text|paper|study|dialogue|collection|narrative|"
    r"story|memoir|report|issue|magazine|module|deck|presentation)\s+"
    r"as\s+a\s+whole\b",
    re.IGNORECASE,
)

_WHOLE_SOURCE_SCOPE_RE = re.compile(
    r"\b(?:what|which|describe|outline|explain|give|provide|offer|trace|discuss)\b"
    r"[^?.!]{0,140}\b(?:across|throughout)\s+(?:this|the)\s+"
    r"(?:book|work|novel|text|paper|study|dialogue|collection|narrative|story|"
    r"memoir|report|issue|magazine|module|deck|presentation)\b",
    re.IGNORECASE,
)

_COLLECTION_COHERENCE_RE = re.compile(
    r"\b(?:subjects?|themes?|ideas?|arguments?|topics?|concerns?)\b"
    r"[^?.!]{0,100}\b(?:connect|unify|recur(?:ring)?|run\s+through)\b"
    r"[^?.!]{0,100}\b(?:essays?|chapters?|sections?|collection)\b"
    r"|\b(?:connect|unify|recur(?:ring)?|run\s+through)\b"
    r"[^?.!]{0,100}\b(?:essays?|chapters?|sections?|collection)\b",
    re.IGNORECASE,
)


def is_overview_query(query: str) -> bool:
    """Return True if *query* asks for an overview/summary of the document."""
    text = query or ""
    return any(
        pattern.search(text)
        for pattern in (
            OVERVIEW_QUERY_RE,
            _HOLISTIC_DESCRIPTION_RE,
            _WHOLE_SOURCE_SCOPE_RE,
            _COLLECTION_COHERENCE_RE,
        )
    )
