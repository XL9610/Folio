"""Synthetic outline construction for sources without usable TOC.

The PDF pipeline should not fall back straight to anonymous token chunks when
bookmarks are missing. This module builds a conservative section structure from
page text, then lets the existing chunker preserve section_path/title metadata.
It operates on DocumentPage so future text/docx parsers can reuse it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .content_boundary import find_body_page_window
from .models import DocumentPage, TocEntry


MAX_TITLE_CHARS = 90

_SMALL_WORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or",
    "the", "to", "with",
}


@dataclass
class SyntheticOutlineResult:
    toc: list[TocEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mode: str = "none"
    body_start_page: int = 0
    body_end_page: int = 0


@dataclass
class _HeadingCandidate:
    title: str
    page: int
    level: int
    score: float


_HEADING_PATTERNS: list[tuple[re.Pattern, int, float]] = [
    (re.compile(r"^(?:part)\s+(?:[ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I), 1, 8.0),
    (re.compile(r"^(?:chapter|ch\.?)\s+\d+\b", re.I), 1, 8.0),
    (re.compile(r"^(?:unit|module|lesson|lecture)\s+\d+\b", re.I), 1, 7.0),
    (re.compile(r"^(?:section|sec\.?)\s+\d+(?:\.\d+)*\b", re.I), 2, 7.0),
    (re.compile(r"^\d{1,3}\.\d+(?:\.\d+)*\s+\S"), 2, 6.5),
    (re.compile(r"^\d{1,3}[\.\)]?\s+[A-Z][A-Za-z0-9,'’:-]+(?:\s+\S+){0,10}$"), 1, 6.0),
    (re.compile(r"^#{1,6}\s+\S+"), 1, 7.0),
]

_REJECT_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\d+$"),
    re.compile(r"^page\s+\d+", re.I),
    re.compile(r"^(?:copyright|all rights reserved|isbn|table of contents|contents)$", re.I),
    re.compile(r"^(?:bibliography|references|index|glossary|works cited)$", re.I),
    re.compile(r"^(?:acknowledg?e?ments?|foreword|preface|praise for|about the author|dedication|epigraph)$", re.I),
    re.compile(r"^to\s+[a-z][\w'\-]*$", re.I),  # dedication pages ("To Pamela")
    re.compile(r"^\W+$"),
]


def build_synthetic_outline(
    pages: list[DocumentPage],
    document_title: str = "",
) -> SyntheticOutlineResult:
    """Build a usable outline when a source has no usable embedded TOC.

    Strategy:
      1. Remove obvious front/back pages with the existing page-window detector.
      2. Find high-confidence heading-like lines from page text.
      3. If there are too few headings, create stable page-window sections.

    The result is intentionally conservative. It is better to have durable
    "Pages 12-18" sections than hallucinated chapter names.
    """
    result = SyntheticOutlineResult()
    if not pages:
        result.warnings.append("Synthetic outline skipped: no pages.")
        return result

    body_start_idx, body_end_idx, boundary_warnings = find_body_page_window(pages)
    result.warnings.extend(boundary_warnings)
    body_pages = pages[body_start_idx:body_end_idx]
    if not body_pages:
        body_pages = pages
        body_start_idx = 0
        body_end_idx = len(pages)

    result.body_start_page = body_pages[0].page_number
    result.body_end_page = body_pages[-1].page_number

    candidates = _find_heading_candidates(body_pages)
    if len(candidates) >= 2:
        result.toc = _toc_from_candidates(candidates, result.body_end_page)
        result.mode = "detected_headings"
        result.warnings.append(
            f"Synthetic outline built from {len(result.toc)} detected section heading(s)."
        )
        return result

    result.toc = _toc_from_page_windows(body_pages, document_title=document_title)
    result.mode = "page_windows"
    result.warnings.append(
        f"Synthetic outline used page windows because only {len(candidates)} heading candidate(s) were detected."
    )
    return result


def _find_heading_candidates(pages: list[DocumentPage]) -> list[_HeadingCandidate]:
    by_page: dict[int, _HeadingCandidate] = {}

    for page in pages:
        lines = _clean_lines(page.text)
        page_candidates: list[_HeadingCandidate] = []
        for idx, line in enumerate(lines[:18]):
            candidate = _score_heading_line(line, idx, page.page_number)
            if candidate:
                page_candidates.append(candidate)

        if page_candidates:
            page_candidates.sort(key=lambda c: (-c.score, c.level, len(c.title)))
            by_page[page.page_number] = page_candidates[0]

    candidates = sorted(by_page.values(), key=lambda c: c.page)
    return _dedupe_candidates(candidates)


def _clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line:
            lines.append(line)
    return lines


def _score_heading_line(line: str, line_index: int, page_number: int) -> _HeadingCandidate | None:
    title = _normalize_title(line)
    if not _is_plausible_title(title):
        return None

    for pattern, level, base_score in _HEADING_PATTERNS:
        if pattern.match(title):
            score = base_score + _position_bonus(line_index)
            return _HeadingCandidate(title=title, page=page_number, level=level, score=score)

    # Short title-case lines near the top of a page are often section titles.
    if line_index <= 5 and _looks_like_title_case(title):
        score = 4.0 + _position_bonus(line_index)
        if title.isupper():
            score += 0.5
        return _HeadingCandidate(title=title, page=page_number, level=1, score=score)

    # Short sentence-case lines near the top are also headings
    # ("The addition rule", "Counting club officer assignments I").
    if line_index <= 5 and _looks_like_short_heading(title):
        score = 3.0 + _position_bonus(line_index)
        return _HeadingCandidate(title=title, page=page_number, level=1, score=score)

    return None


def _normalize_title(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^#{1,6}\s+", "", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"[\s:;,.]+$", "", s)
    return s[:MAX_TITLE_CHARS].strip()


def _is_plausible_title(title: str) -> bool:
    if not title or len(title) < 4 or len(title) > MAX_TITLE_CHARS:
        return False
    if any(p.match(title) for p in _REJECT_PATTERNS):
        return False
    words = title.split()
    if len(words) > 14:
        return False
    # Long sentence-like lines are usually body text, not headings.
    if title.endswith((".", "?", "!")):
        return False
    if title.count(",") >= 3:
        return False
    alpha = sum(ch.isalpha() for ch in title)
    if alpha < max(3, len(title) * 0.45):
        return False
    return True


def _looks_like_title_case(title: str) -> bool:
    words = [w for w in re.split(r"\s+", title) if re.search(r"[A-Za-z]", w)]
    if not 2 <= len(words) <= 10:
        return False
    small = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
    strong = 0
    for word in words:
        cleaned = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", word)
        if not cleaned:
            continue
        if cleaned.lower() in small:
            continue
        if cleaned[0].isupper() or cleaned.isupper():
            strong += 1
    return strong >= max(2, math.ceil(len(words) * 0.55))


def _looks_like_short_heading(title: str) -> bool:
    """Detect short sentence-case headings like "The addition rule".

    Such headings are 2-5 words, contain at least one non-stopword, and are
    sentence case (not Title Case), so _looks_like_title_case misses them.
    """
    words = [w for w in re.split(r"\s+", title) if re.search(r"[A-Za-z]", w)]
    if not 2 <= len(words) <= 5:
        return False
    content = [w for w in words if w.lower() not in _SMALL_WORDS]
    return len(content) >= 1


def _position_bonus(line_index: int) -> float:
    if line_index <= 1:
        return 1.5
    if line_index <= 5:
        return 0.8
    return 0.0


def _dedupe_candidates(candidates: list[_HeadingCandidate]) -> list[_HeadingCandidate]:
    result: list[_HeadingCandidate] = []
    seen_titles: set[str] = set()
    for candidate in candidates:
        key = re.sub(r"\W+", " ", candidate.title.lower()).strip()
        # Repeated running headers often appear on many pages; keep only first.
        if key in seen_titles:
            continue
        seen_titles.add(key)
        result.append(candidate)
    return result


def _toc_from_candidates(
    candidates: list[_HeadingCandidate],
    body_end_page: int,
) -> list[TocEntry]:
    entries: list[TocEntry] = []
    stack: list[TocEntry] = []

    for i, candidate in enumerate(candidates):
        next_page = candidates[i + 1].page if i + 1 < len(candidates) else body_end_page + 1
        page_end = max(candidate.page, next_page - 1)
        entry = TocEntry(
            title=candidate.title,
            level=candidate.level,
            page_start=candidate.page,
            page_end=page_end,
        )

        while stack and stack[-1].level >= entry.level:
            stack.pop()
        if stack:
            stack[-1].children.append(entry)
        else:
            entries.append(entry)
        stack.append(entry)

    return entries


def _toc_from_page_windows(
    pages: list[DocumentPage],
    document_title: str = "",
) -> list[TocEntry]:
    total = len(pages)
    if total <= 0:
        return []

    # Aim for 6-10 sections in long works, but do not create tiny windows.
    target_sections = min(10, max(1, math.ceil(total / 8)))
    window_size = max(3, math.ceil(total / target_sections))
    entries: list[TocEntry] = []

    for start_idx in range(0, total, window_size):
        end_idx = min(total - 1, start_idx + window_size - 1)
        start_page = pages[start_idx].page_number
        end_page = pages[end_idx].page_number
        title = f"Pages {start_page}-{end_page}" if start_page != end_page else f"Page {start_page}"
        if document_title and len(entries) == 0:
            title = f"{document_title}: {title}"
            if len(title) > MAX_TITLE_CHARS:
                title = f"Pages {start_page}-{end_page}" if start_page != end_page else f"Page {start_page}"
        entries.append(TocEntry(
            title=title,
            level=1,
            page_start=start_page,
            page_end=end_page,
        ))

    return entries
