"""Body-aware chunk sampling for generation (summary, flashcards).

Replaces naive head+tail sampling with distributed section-aware sampling
so that whole-document generation doesn't disproportionately pick up
front matter or back matter when ingestion filtering is imperfect.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from ..retrieval.base import RetrievedChunk

# ---------------------------------------------------------------------------
# Non-body patterns (generation layer safety net)
# ---------------------------------------------------------------------------

_NON_BODY_TITLE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(?:front\s*)?cover$", re.IGNORECASE),
    re.compile(r"^title\s*page$", re.IGNORECASE),
    re.compile(r"^copyright$", re.IGNORECASE),
    re.compile(r"^colophon$", re.IGNORECASE),
    re.compile(r"^dedication$", re.IGNORECASE),
    re.compile(r"^(?:brief\s*|detailed\s*|expanded\s*)?contents$", re.IGNORECASE),
    re.compile(r"^table\s*of\s*contents$", re.IGNORECASE),
    re.compile(r"^editorial\s+(?:advisory\s+)?board$", re.IGNORECASE),
    re.compile(r"^advisory\s+board$", re.IGNORECASE),
    re.compile(r"^maps?$", re.IGNORECASE),
    re.compile(r"^guide\s+to\s+", re.IGNORECASE),
    re.compile(
        r"^list\s*of\s*(?:illustrations|figures|tables|maps|examples|abbreviations|charts|diagrams|boxes|timelines|recordings|audio\s*(?:examples|selections|tracks)?|listening\s*(?:examples|selections)?|texts?)$",
        re.IGNORECASE,
    ),
    re.compile(r"^preface", re.IGNORECASE),
    re.compile(r"^foreword$", re.IGNORECASE),
    re.compile(r"^acknowledg?ments?$", re.IGNORECASE),
    re.compile(r".*\bcredits?\s*$", re.IGNORECASE),
    re.compile(
        r"^about\s*the\s*(?:author|authors|editor|editors|contributors?)$",
        re.IGNORECASE,
    ),
    re.compile(r"^bibliography$", re.IGNORECASE),
    re.compile(r"^references?$", re.IGNORECASE),
    re.compile(r"^works?\s*cited$", re.IGNORECASE),
    re.compile(r"^further\s*reading$", re.IGNORECASE),
    re.compile(r"^glossary$", re.IGNORECASE),
    re.compile(r"^index\s*$", re.IGNORECASE),
    re.compile(r"^index\s+of\s+", re.IGNORECASE),
    re.compile(r"^epilogue$", re.IGNORECASE),
    re.compile(r"^afterword$", re.IGNORECASE),
    re.compile(r"^notes?$", re.IGNORECASE),
    re.compile(r"^endnotes?$", re.IGNORECASE),
    re.compile(r"^biographical\s+notes?$", re.IGNORECASE),
    re.compile(r"^contributors?$", re.IGNORECASE),
]

# Exact-match blocklist for individual section path components
_NON_BODY_PATH_SET: set[str] = {
    "bibliography",
    "references",
    "works cited",
    "further reading",
    "glossary",
    "index",
    "index of",
    "credits",
    "colophon",
    "about the author",
    "about the authors",
    "also by",
    "praise for",
    "epilogue",
    "afterword",
    "notes",
    "endnotes",
    "list of texts",
    "list of illustrations",
    "biographical notes",
    "contributors",
}


def _is_non_body_title(title: str) -> bool:
    """Check if a chunk title strongly indicates front/back matter."""
    t = title.strip().lower()
    if not t:
        return False
    for pat in _NON_BODY_TITLE_PATTERNS:
        if pat.match(t):
            return True
    return False


def _is_non_body_section_path(section_path: str) -> bool:
    """Check if any component of the section path is an exact non-body marker.

    Uses component-level matching (not substring) to avoid false positives
    like "indexing algorithms" being flagged for containing "index".
    """
    if not section_path:
        return False
    parts = [p.strip().lower() for p in section_path.split(" > ")]
    for part in parts:
        if not part:
            continue
        # Exact match first
        if part in _NON_BODY_PATH_SET:
            return True
        # Also match prefixes for "index of ..." variants
        if "index of" in part:
            return True
    return False


def is_body_chunk(chunk: RetrievedChunk) -> bool:
    """Check if a chunk is likely body content (not front/back matter)."""
    # Check title first
    if chunk.title and _is_non_body_title(chunk.title):
        return False

    # Check section_path
    if chunk.section_path and _is_non_body_section_path(chunk.section_path):
        return False

    return True


# ---------------------------------------------------------------------------
# Distributed body-aware sampling
# ---------------------------------------------------------------------------


def sample_body_chunks(
    chunks: list[RetrievedChunk],
    max_chunks: int = 25,
) -> list[RetrievedChunk]:
    """Sample chunks for whole-document generation, favoring body content.

    Strategy:
    1. Filter obvious non-body chunks (front/back matter).
    2. If remaining chunks ≤ max_chunks, return all.
    3. Allocate the budget fairly across documents, so the first document cannot
       monopolize the window.
    4. Within each document, pick evenly-spaced chunks across the full text
       position, so the head sections cannot crowd out the rest.
    """
    if not chunks:
        return []

    body = [c for c in chunks if is_body_chunk(c)]
    if not body:
        body = chunks
    if len(body) <= max_chunks:
        return body

    by_doc: dict[str, list[RetrievedChunk]] = {}
    for c in body:
        by_doc.setdefault(c.document_id or "", []).append(c)

    doc_ids = list(by_doc.keys())
    budgets = _allocate_document_budget(
        [len(by_doc[d]) for d in doc_ids], max_chunks
    )

    sampled: list[RetrievedChunk] = []
    for did, k in zip(doc_ids, budgets):
        sampled.extend(_spread_evenly(by_doc[did], k))
    return sampled


def _allocate_document_budget(sizes: list[int], total: int) -> list[int]:
    """Fair per-document budget: at least one each, remainder by size."""
    n = len(sizes)
    if n == 0:
        return []
    budgets = [1] * n if total >= n else [0] * n
    remaining = total - sum(budgets)
    total_size = sum(sizes)
    if total_size > 0 and remaining > 0:
        exact = [remaining * s / total_size for s in sizes]
        floors = [int(math.floor(e)) for e in exact]
        for i in range(n):
            budgets[i] += floors[i]
        leftover = remaining - sum(floors)
        order = sorted(range(n), key=lambda i: exact[i] - floors[i], reverse=True)
        for i in order[:leftover]:
            budgets[i] += 1
    return budgets


def _spread_evenly(items: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
    """Pick *k* items spread evenly across *items* (beginning/middle/end)."""
    n = len(items)
    if k >= n:
        return list(items)
    if k <= 0:
        return []
    if k == 1:
        return [items[0]]
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


# ---------------------------------------------------------------------------
# Section-aware chunk grouping and budget allocation
# ---------------------------------------------------------------------------


@dataclass
class SectionGroup:
    """A group of chunks belonging to the same major section/chapter."""
    key: str           # display name (first section path component)
    chunks: list[RetrievedChunk]
    page_start: int = 0
    page_end: int = 0
    chunk_weight: int = 0  # number of body chunks in this section


def group_chunks_by_major_section(
    chunks: list[RetrievedChunk],
) -> list[SectionGroup]:
    """Group body chunks by major section (chapter-level, not just Part-level).

    Uses major_section_key() to determine the grouping key:
    - For "Chapter 1 Art as History > Introduction" → key = "Chapter 1 Art as History"
    - For "PART ONE > 1. Music in Antiquity" → key = "1. Music in Antiquity"
    - For "PART ONE > 2. The Christian Church" → key = "2. The Christian Church"

    Returns sorted list by page_start.
    """
    groups: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for c in chunks:
        if not is_body_chunk(c):
            continue
        key = major_section_key(c.section_path or c.title or "")
        groups[key].append(c)

    result: list[SectionGroup] = []
    for key, group_chunks in groups.items():
        pages = sorted(c.page_start for c in group_chunks if c.page_start > 0)
        result.append(SectionGroup(
            key=key,
            chunks=group_chunks,
            page_start=pages[0] if pages else 0,
            page_end=pages[-1] if pages else 0,
            chunk_weight=len(group_chunks),
        ))

    result.sort(key=lambda g: g.page_start)
    return result


def major_section_key(section_path: str) -> str:
    """Determine the major section grouping key from a section path.

    Priority:
      1. Find the most specific chapter-like component (numbered or "Chapter X")
      2. Include its immediate parent if the parent provides useful context
      3. Fall back to the top-level component

    Examples:
        "Chapter 1 Art as History > Introduction"   → "Chapter 1 Art as History"
        "PART ONE > 1. Music in Antiquity"           → "1. Music in Antiquity"
        "PART TWO > 7. Music and the Renaissance"    → "7. Music and the Renaissance"
        "Chapter 2 Aesthetics > Philosophy of Art"    → "Chapter 2 Aesthetics"
        "Introduction"                                → "Introduction"
    """
    if not section_path:
        return "_uncategorized"

    parts = [p.strip() for p in section_path.split(" > ") if p.strip()]
    if not parts:
        return "_uncategorized"

    # Chapter-like patterns
    chapter_patterns: list[re.Pattern] = [
        re.compile(r"^(?:chapter|ch\.?)\s*\d+", re.IGNORECASE),
        re.compile(r"^\d{1,3}[\.\s]", re.IGNORECASE),  # "1. Music", "10 Baroque"
        re.compile(r"^(?:part)\s+[ivxlcdm\d]+", re.IGNORECASE),
        re.compile(r"^(?:unit|module|week|lesson|lecture|topic|theme)\s*\d+", re.IGNORECASE),
    ]

    # Find the best chapter-level component (prefer deeper levels)
    best_idx = -1
    for i in range(len(parts) - 1, -1, -1):  # scan from deepest to shallowest
        part = parts[i]
        for pat in chapter_patterns:
            if pat.match(part):
                best_idx = i
                break
        if best_idx >= 0:
            break

    if best_idx >= 0:
        # Use the chapter component, optionally with parent for context
        if best_idx > 0 and best_idx < len(parts):
            parent = parts[best_idx - 1]
            # Only include parent if it's a structure label (Part, Unit, etc.)
            if re.match(r"^(?:part|unit|section|module)\s+", parent, re.IGNORECASE):
                return f"{parent} > {parts[best_idx]}"
        return parts[best_idx]

    # No chapter-like component found — use top-level
    return parts[0]


def sample_section_chunks(
    section: SectionGroup,
    max_chunks: int = 4,
) -> list[RetrievedChunk]:
    """Sample representative chunks from a section.

    Strategy: pick beginning, middle, and end chunks, plus
    distinct subsection paths when available.
    """
    chunks = section.chunks
    if len(chunks) <= max_chunks:
        return list(chunks)

    sampled: list[RetrievedChunk] = []

    # Always include first and last chunk
    sampled.append(chunks[0])
    if len(chunks) > 1:
        sampled.append(chunks[-1])

    # Include a middle chunk (by index)
    mid_idx = len(chunks) // 2
    mid_chunk = chunks[mid_idx]
    if mid_chunk not in sampled:
        sampled.append(mid_chunk)

    # Collect distinct subsection paths (2nd level)
    subsections: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for c in chunks:
        if c in sampled:
            continue
        parts = (c.section_path or "").split(" > ")
        sub = parts[1].strip() if len(parts) > 1 else "_leaf"
        subsections[sub].append(c)

    # Round-robin across subsections for remaining slots
    if subsections and len(sampled) < max_chunks:
        sub_keys = list(subsections.keys())
        idxs = {k: 0 for k in sub_keys}
        round_idx = 0
        while len(sampled) < max_chunks:
            k = sub_keys[round_idx % len(sub_keys)]
            if idxs[k] < len(subsections[k]):
                c = subsections[k][idxs[k]]
                if c not in sampled:
                    sampled.append(c)
                idxs[k] += 1
            round_idx += 1
            if all(idxs[k] >= len(subsections[k]) for k in sub_keys):
                break

    # If still not enough, fill from beginning
    for c in chunks:
        if len(sampled) >= max_chunks:
            break
        if c not in sampled:
            sampled.append(c)

    return sampled[:max_chunks]


def allocate_flashcard_budget(
    sections: list[SectionGroup],
    total_count: int,
) -> dict[str, int]:
    """Allocate flashcard budget across sections proportional to chunk weight.

    Each section gets at least 1 card (if total_count >= number of sections),
    or a proportional share plus remainder distributed by weight.

    Small sections (1-chunk) are NOT excluded — they still receive a minimum
    allocation when the total budget permits.

    Returns dict mapping section key → allocated count.
    """
    if not sections:
        return {}

    n = len(sections)
    if n == 0:
        return {}

    # Minimum 1 per section if total allows, otherwise allocate by weight
    if total_count >= n:
        allocation = {s.key: 1 for s in sections}
        remaining = total_count - n
    else:
        # Not enough for all — allocate by weight only
        allocation = {}
        remaining = total_count

    if remaining <= 0:
        return allocation

    # Proportional allocation by chunk_weight
    total_weight = sum(s.chunk_weight for s in sections)
    if total_weight == 0:
        return {s.key: total_count // n for s in sections}

    # Distribute proportionally
    for s in sections:
        prop = int(remaining * s.chunk_weight / total_weight)
        allocation[s.key] = allocation.get(s.key, 0) + prop

    # Distribute remainder to largest sections
    allocated_total = sum(allocation.values())
    remainder = total_count - allocated_total
    sorted_sections = sorted(sections, key=lambda s: s.chunk_weight, reverse=True)
    for s in sorted_sections:
        if remainder <= 0:
            break
        allocation[s.key] = allocation.get(s.key, 0) + 1
        remainder -= 1

    # Cap per-section max (no single section should dominate)
    max_per_section = max(2, math.ceil(total_count * 0.4))
    for s in sections:
        if allocation.get(s.key, 0) > max_per_section:
            overage = allocation[s.key] - max_per_section
            allocation[s.key] = max_per_section
            others = [o for o in sections if o.key != s.key]
            if others:
                dist_per = math.ceil(overage / len(others))
                for o in others:
                    if overage <= 0:
                        break
                    add = min(dist_per, overage)
                    allocation[o.key] = allocation.get(o.key, 0) + add
                    overage -= add

    return allocation
