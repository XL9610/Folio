"""Content boundary detection for PDF chunks.

Identifies where body content starts and ends in a list of TOC units,
distinguishing synthetic wrappers, front matter, body, and back matter.

Goal: ensure only actual study content enters the chunk index,
not just for one book but generically across diverse textbook PDFs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from .models import DocumentPage, TOCUnit

# ---------------------------------------------------------------------------
# Content role classification
# ---------------------------------------------------------------------------


class ContentRole(str, Enum):
    WRAPPER = "wrapper"           # book title / root node, not real content
    FRONT_MATTER = "front_matter" # copyright, contents, preface, etc.
    BODY = "body"                 # actual study content
    BACK_MATTER = "back_matter"   # bibliography, index, glossary, credits
    UNKNOWN = "unknown"           # can't determine confidently


# ---------------------------------------------------------------------------
# Pattern sets (generic, not book-specific)
# ---------------------------------------------------------------------------

# Patterns that indicate a synthetic wrapper / book title node
WRAPPER_PATTERNS: list[re.Pattern] = [
    # Long title with mostly uppercase words (matched after lowercasing)
    re.compile(r"^(?:[a-z]\S*\s+){4,}[a-z]\S*$"),  # 5+ words (title-like)
    re.compile(r"^\w+\s+(?:history|introduction|guide|handbook|companion|textbook)"),
    re.compile(r"^a\s+history\s+of", re.IGNORECASE),
]

# Patterns that STRONGLY indicate front matter
STRONG_FRONT_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(?:front\s*)?cover$", re.IGNORECASE),
    re.compile(r"^title\s*page$", re.IGNORECASE),
    re.compile(r"^half[\s-]?title$", re.IGNORECASE),
    re.compile(r"^copyright$", re.IGNORECASE),
    re.compile(r"^colophon$", re.IGNORECASE),
    re.compile(r"^dedication$", re.IGNORECASE),
    re.compile(r"^(?:brief\s*|detailed\s*|expanded\s*)?contents$", re.IGNORECASE),
    re.compile(r"^table\s*of\s*contents$", re.IGNORECASE),
    re.compile(r"^editorial\s+(?:advisory\s+)?board$", re.IGNORECASE),
    re.compile(r"^advisory\s+board$", re.IGNORECASE),
    re.compile(r"^maps?$", re.IGNORECASE),
    re.compile(
        r"^guide\s+to\s+",
        re.IGNORECASE,
    ),  # Guide to NAWM, Guide to Recordings, etc.
    re.compile(
        r"^list\s*of\s*(?:illustrations|figures|tables|maps|examples|abbreviations|charts|diagrams|boxes|timelines|recordings|audio\s*(?:examples|selections|tracks)?|listening\s*(?:examples|selections)?)$",
        re.IGNORECASE,
    ),
    re.compile(r"^preface$", re.IGNORECASE),
    re.compile(r"^preface\s+to\s+(?:the\s+)?\w+\s+edition$", re.IGNORECASE),
    re.compile(r"^foreword$", re.IGNORECASE),
    re.compile(r"^prologue$", re.IGNORECASE),
    re.compile(r"^acknowledg?ments?$", re.IGNORECASE),
    re.compile(r".*\bcredits?\s*$", re.IGNORECASE),
    re.compile(
        r"^about\s*the\s*(?:author|authors|editor|editors|contributors?)$",
        re.IGNORECASE,
    ),
    re.compile(r"^also\s+by", re.IGNORECASE),
    re.compile(r"^praise\s+for", re.IGNORECASE),
    re.compile(r"^reviews?\s+for", re.IGNORECASE),
    re.compile(r"^how\s*to\s*use\s*this\s*book$", re.IGNORECASE),
    re.compile(r"^a\s*note\s*on\s*(?:translation|terminology|the\s*text|conventions|spelling|dates|pronunciation|usage|sources|citations?|references?)$", re.IGNORECASE),
    re.compile(r"^notes?\s*to\s*the\s*(?:reader|student|instructor|teacher)s?$", re.IGNORECASE),
]

# Patterns that STRONGLY indicate body content
STRONG_BODY_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(?:chapter|ch\.?)\s*\d+", re.IGNORECASE),
    re.compile(r"^(?:chapter)\s+[ivxlcdm]+\b", re.IGNORECASE),
    re.compile(r"^(?:part)\s*[ivxlcdm\d]+", re.IGNORECASE),
    re.compile(r"^part\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", re.IGNORECASE),
    re.compile(r"^(?:unit|module|week|lesson|lecture|topic|theme)\s*\d+", re.IGNORECASE),
    re.compile(r"^\d{1,3}[\.\s]\s*\S{3,}", re.IGNORECASE),  # "1 Music in Antiquity", "15. The Madrigal"
]

# Patterns that STRONGLY indicate back matter
STRONG_BACK_PATTERNS: list[re.Pattern] = [
    re.compile(r"^bibliography$", re.IGNORECASE),
    re.compile(r"^references?$", re.IGNORECASE),
    re.compile(r"^works?\s*cited$", re.IGNORECASE),
    re.compile(r"^further\s*reading$", re.IGNORECASE),
    re.compile(r"^glossary$", re.IGNORECASE),
    re.compile(r"^index\s*$", re.IGNORECASE),
    re.compile(r"^index\s+of\s+", re.IGNORECASE),
    re.compile(r"^credits$", re.IGNORECASE),
    re.compile(r"^colophon$", re.IGNORECASE),
    re.compile(r"^appendices$", re.IGNORECASE),
    re.compile(r"^appendix\s*[a-z]?$", re.IGNORECASE),
    re.compile(
        r"^about\s*the\s*(?:author|authors|editor|editors|contributors?)$",
        re.IGNORECASE,
    ),
    re.compile(r"^epilogue$", re.IGNORECASE),
    re.compile(r"^afterword$", re.IGNORECASE),
]

# Titles within body that are NOT study content on their own but are
# body-internal features (e.g. sidebars, timelines) — we DON'T skip these
# because they are inside the body window.
BODY_INTERNAL_SAFE = frozenset({
    "timeline", "chronology", "source reading", "in performance",
    "music in context", "innovations", "forms at a glance",
    "key concepts", "echoes of history", "echoes of the new art",
    "performing", "summary",
})


# ---------------------------------------------------------------------------
# Boundary decision result
# ---------------------------------------------------------------------------


@dataclass
class BoundaryDecision:
    """Result of content boundary detection."""
    filtered_units: list[TOCUnit] = field(default_factory=list)
    body_start_index: int = 0
    body_end_index: int = 0  # exclusive
    skipped_front_titles: list[str] = field(default_factory=list)
    skipped_back_titles: list[str] = field(default_factory=list)
    skipped_wrapper_titles: list[str] = field(default_factory=list)
    body_start_title: str = ""
    body_end_title: str = ""
    back_matter_start_title: str = ""
    wrapper_detected: bool = False
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _matches_any(title_lower: str, patterns: list[re.Pattern]) -> bool:
    return any(p.match(title_lower) for p in patterns)


def classify_toc_unit(
    unit: TOCUnit,
    total_units: int,
    page_count: int,
) -> ContentRole:
    """Classify a single TOC unit as wrapper, front_matter, body, or back_matter.

    Rules are generic — no book-specific hardcoding.
    """
    t = unit.entry.title.strip().lower()
    if not t:
        return ContentRole.UNKNOWN

    # --- Check back matter first (so it's never confused with front) ---
    # Back matter is ONLY classified when near the document end AND
    # the entry is a top-level boundary (not a mid-chapter subsection titled
    # "References" or "Further Reading").
    position_ratio = unit.index / max(total_units - 1, 1)
    if _matches_any(t, STRONG_BACK_PATTERNS):
        # Top-level entry (level <= 2) in the final 25% of the TOC
        # AND in the final 30% of pages — strong back-matter signal
        if position_ratio > 0.75 and unit.level <= 2:
            return ContentRole.BACK_MATTER
        # Also at moderate position but very late in pages
        if position_ratio > 0.5 and unit.level <= 2 and unit.entry.page_start > page_count * 0.8:
            return ContentRole.BACK_MATTER
        # If very early in document, it might be front-matter
        if position_ratio < 0.15 and unit.level <= 2:
            return ContentRole.FRONT_MATTER
        # Otherwise: mid-book "References"/"Further Reading" section —
        # treat as body (conservative, don't truncate)
        return ContentRole.UNKNOWN

    # --- Check strong front matter ---
    if _matches_any(t, STRONG_FRONT_PATTERNS):
        # "Credits" at the very end could be back matter
        if position_ratio > 0.85:
            return ContentRole.BACK_MATTER
        return ContentRole.FRONT_MATTER

    # --- Check wrapper (book title root) ---
    # A wrapper is level-1, has children, appears very early, and its title
    # is an all-caps or book-title pattern
    if unit.level == 1 and unit.has_children and unit.index <= 1:
        if _matches_any(t, WRAPPER_PATTERNS):
            return ContentRole.WRAPPER

    # --- Check strong body ---
    if _matches_any(t, STRONG_BODY_PATTERNS):
        # Bare numbers on very early pages might be page numbers, not chapters
        if re.match(r"^\d{1,3}\s", t):
            if unit.entry.page_start <= 5 and unit.index < 5:
                # Could be a page number like "1 2 3 4 5" — but still
                # lean towards body since we can't be sure
                pass
        return ContentRole.BODY

    # --- Unknown ---
    return ContentRole.UNKNOWN


# ---------------------------------------------------------------------------
# Body window detection
# ---------------------------------------------------------------------------

# After we find body_start, we stop scanning front matter and look for
# back matter boundaries.  Once we hit a strong back-matter top-level
# entry after the body_start, everything from that point is back matter.

def _count_descendants(units: list[TOCUnit], parent_idx: int) -> int:
    """Count how many subsequent entries have level > units[parent_idx].level."""
    parent_level = units[parent_idx].level
    count = 0
    for j in range(parent_idx + 1, len(units)):
        if units[j].level > parent_level:
            count += 1
        else:
            break
    return count


def _skip_subtree(units: list[TOCUnit], i: int) -> int:
    """Return the next index after skipping unit[i] + its descendants."""
    desc = _count_descendants(units, i)
    return i + desc + 1


def find_body_window(
    units: list[TOCUnit],
    page_count: int,
) -> BoundaryDecision:
    """Find the body content window in a list of TOC units.

    Returns a BoundaryDecision with only the body units (front/back/wrapper removed).
    """
    decision = BoundaryDecision(filtered_units=list(units))

    if not units:
        decision.warnings.append("No TOC units to analyze.")
        return decision

    total = len(units)
    body_start = -1
    body_end = total
    back_start = -1
    wrapper_detected = False

    i = 0
    # Phase 1: skip wrappers and front matter
    while i < total:
        role = classify_toc_unit(units[i], total, page_count)

        if role == ContentRole.WRAPPER:
            wrapper_detected = True
            decision.skipped_wrapper_titles.append(units[i].entry.title)
            # Skip only the wrapper node itself (NOT its subtree).
            # Its children (front matter / body) will be classified individually
            # as we continue scanning.
            i += 1
            continue

        if role == ContentRole.FRONT_MATTER:
            decision.skipped_front_titles.append(units[i].entry.title)
            # Skip this entry AND all its descendants (subtree)
            i = _skip_subtree(units, i)
            continue

        if role == ContentRole.BODY:
            body_start = i
            break

        # UNKNOWN: treat as body start (conservative)
        body_start = i
        break

    # Safety: if no body found, start from 0 (keep all)
    if body_start == -1:
        body_start = 0
        decision.warnings.append(
            "Could not confidently detect body start — kept all entries from index 0."
        )

    # Phase 2: find back matter boundary (scan from body_start through end)
    back_start = total
    for i in range(body_start, total):
        role = classify_toc_unit(units[i], total, page_count)
        if role == ContentRole.BACK_MATTER:
            back_start = i
            break

    body_end = back_start

    # Collect back-matter titles
    for i in range(back_start, total):
        decision.skipped_back_titles.append(units[i].entry.title)

    # Build filtered body units
    filtered = units[body_start:body_end]

    # Safety: if everything was filtered, keep all
    if not filtered:
        decision.warnings.append(
            "All TOC entries filtered — keeping all as fallback."
        )
        filtered = units
        body_start = 0
        body_end = total

    decision.filtered_units = filtered
    decision.body_start_index = body_start
    decision.body_end_index = body_end
    decision.wrapper_detected = wrapper_detected

    if filtered:
        decision.body_start_title = filtered[0].entry.title
        decision.body_end_title = filtered[-1].entry.title

    if back_start < total and back_start > 0:
        decision.back_matter_start_title = units[back_start].entry.title

    return decision


# ---------------------------------------------------------------------------
# Page-level body window (for token fallback, no TOC)
# ---------------------------------------------------------------------------


def _page_contains_front_matter_keywords(text: str) -> bool:
    """Heuristic: does page text strongly look like front matter?"""
    t = text.lower()[:2000]
    # Very strong signals
    strong = [
        "copyright ©",
        "all rights reserved",
        "isbn",
        "library of congress cataloging",
        "cataloging-in-publication",
        "printed in the united states",
        "table of contents",
    ]
    for s in strong:
        if s in t:
            return True
    return False


def _page_contains_back_matter_keywords(text: str) -> bool:
    """Heuristic: does page text strongly look like back matter?"""
    t = text.lower()[:2000]
    strong = [
        "bibliography",
        "works cited",
        "further reading",
        "references",
    ]
    # Check for index pattern: page starts with "Index" and has dense letter sections
    if t.strip().startswith("index") or t.strip().startswith("note:"):
        # Could be index
        return True
    for s in strong:
        if s in t:
            return True
    return False


def find_body_page_window(
    pages: list[DocumentPage],
) -> tuple[int, int, list[str]]:
    """Detect body page range for token fallback.

    Returns (start_page_index, end_page_index_exclusive, warnings).

    start_page_index and end_page_index are 0-based indices into `pages`.
    """
    warnings: list[str] = []
    total = len(pages)
    if total == 0:
        return 0, 0, warnings

    # Scan from front to find first non-front-matter page
    start = 0
    for i in range(min(total, 15)):  # only scan first 15 pages
        if _page_contains_front_matter_keywords(pages[i].text):
            start = i + 1
        else:
            # Once we hit a page without front-matter signals, stop scanning
            break

    # Scan from back to find first non-back-matter page
    end = total
    back_scan = min(total, max(10, total // 8))  # scan back ~12% of pages
    for i in range(total - 1, total - 1 - back_scan, -1):
        if i < start:
            break
        if _page_contains_back_matter_keywords(pages[i].text):
            end = i
        else:
            # Once we hit a page without back-matter signals, stop scanning
            break

    if start > 0:
        warnings.append(
            f"Skipped {start} likely front-matter page(s) in token fallback."
        )
    if end < total:
        warnings.append(
            f"Skipped {total - end} likely back-matter page(s) in token fallback."
        )

    # Safety: never return empty
    if start >= end:
        start = 0
        end = total

    return start, end, warnings
