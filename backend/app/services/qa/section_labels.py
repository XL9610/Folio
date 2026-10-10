"""Section-label text utilities for the QA path.

Parses outline segments ("Chapter 20: X", "PART > 20. Music") into chapter
numbers and display labels, and maps chapter numbers back to page ranges.
"""

from __future__ import annotations

import re


_PAGE_WINDOW_LABEL_RE = re.compile(r"^pages?\s+\d+(?:\s*[-\u2013\u2014]\s*\d+)?$", re.IGNORECASE)


def resolve_chapter_page_range(
    chapter_number: int,
    sections: list[dict],
) -> tuple[int, int] | None:
    """Map chapter number → (page_start, page_end) from document sections.

    Resolution strategy (ordered, first match wins):
        1. Exact chapter number match in section_path: "Chapter 20" or "20. Title" or "Ch. 20"
        2. Exact chapter number in title
    """
    patterns = [
        re.compile(rf"\bChapter\s+{chapter_number}\b", re.IGNORECASE),
        re.compile(rf"^{chapter_number}\.\s"),
        re.compile(rf"\bCh\.\s*{chapter_number}\b", re.IGNORECASE),
    ]

    for section in sections:
        page_start = section.get("page_start", -1)
        page_end = section.get("page_end", -1)

        if page_start == -1 or page_end == -1:
            continue

        path = section.get("section_path", "")
        title = section.get("title", "")

        for pat in patterns:
            if pat.search(path) or pat.search(title):
                return (page_start, page_end)

    return None


def _parse_numbered_segment(segment: str) -> tuple[int | None, str]:
    """Parse one outline segment as a numbered chapter/section heading."""
    text = (segment or "").strip()
    if not text:
        return None, ""

    m = re.match(
        r"^(?:(?:Chapter|Ch|Section|Sec)\s*)?(\d{1,3})(?:[.:)]|\s+)(.+)$",
        text,
        re.IGNORECASE,
    )
    if not m:
        m = re.match(
            r"^(?:(?:Chapter|Ch|Section|Sec)\s*)?(\d{1,3})$",
            text,
            re.IGNORECASE,
        )
    if not m:
        return None, ""

    label = text
    # Avoid treating subsection headings like 1.1 or 20.2 as chapter 1/20.
    if re.match(r"^\d+\.\d+", label):
        return None, ""
    return int(m.group(1)), label


def extract_numbered_section_label(section_path: str, title: str = "") -> tuple[int | None, str]:
    """Return (number, label) for paths like 'Chapter 20: X' or 'PART > 20. X'.

    The parser intentionally looks at each path segment, so it works for books
    whose top level is a Part and whose chapters are bare numbered headings.
    """
    number, _prefix, label = extract_numbered_section_prefix(section_path, title)
    return number, label


def extract_numbered_section_prefix(
    section_path: str,
    title: str = "",
) -> tuple[int | None, str, str]:
    """Return (number, full_path_prefix, label) for the first numbered segment.

    Example:
      "PART FIVE > 20. Music > Introduction"
      -> (20, "PART FIVE > 20. Music", "20. Music")
    """
    candidates = [section_path or "", title or ""]
    for text in candidates:
        prefix_segments: list[str] = []
        for segment in [s.strip() for s in text.split(">") if s.strip()]:
            prefix_segments.append(segment)
            number, label = _parse_numbered_segment(segment)
            if number is not None:
                return number, " > ".join(prefix_segments), label
    return None, "", ""


def section_label_for_display(section_path: str, title: str = "") -> str:
    """Best human-facing label for a matched section."""
    _number, _prefix, label = extract_numbered_section_prefix(section_path, title)
    if label:
        return label
    if section_path:
        return section_path.split(">")[-1].strip()
    return title.strip()


def is_page_window_section_label(section_path: str, title: str = "") -> bool:
    """Return True for neutral fallback labels like 'Pages 20-27'."""
    candidates = [section_path or "", title or ""]
    for text in candidates:
        for segment in [s.strip() for s in text.split(">") if s.strip()] or [text.strip()]:
            if _PAGE_WINDOW_LABEL_RE.match(segment):
                return True
    return False
