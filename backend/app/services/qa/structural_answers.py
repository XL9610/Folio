"""Deterministic structural Q&A (chapter title / page / section-locate).

Answers "What is chapter 3 titled?", "Where does chapter 5 start?", and "which
section discusses X?" directly from find_sections metadata — no LLM. Also owns
the unresolved-section disclaimer prefix for content answers.
"""

from __future__ import annotations

import re

from ..pdf_ingestion.structure_metadata import is_unreliable_structure_status
from .section_labels import extract_numbered_section_prefix, section_label_for_display
from .tools import SectionMatch, ToolResults


_SECTION_REF_RE = re.compile(
    r"\b(?:chapter|ch|section|sec)\s*#?\s*(\d{1,3})\b",
    re.IGNORECASE,
)
_STRUCTURAL_TITLE_RE = re.compile(
    r"\b(title|titled|name|named|heading|called)\b",
    re.IGNORECASE,
)
_STRUCTURAL_PAGE_RE = re.compile(
    r"\b(page|pages|start|starts|begin|begins|where|located|location)\b",
    re.IGNORECASE,
)
_SECTION_CONTENT_RE = re.compile(
    r"\b(discuss|discussed|about|summari[sz]e|summary|explain|cover|covered|"
    r"content|main idea|theme|themes|say|describe|overview)\b",
    re.IGNORECASE,
)
_SECTION_DISCUSSION_QUERY_RE = re.compile(
    r"\bwhat\s+(?:section|part|chapter|heading)\b.*"
    r"\b(?:discuss|discusses|cover|covers|about|mentions?)\b",
    re.IGNORECASE,
)
_CHAPTER_LIST_QUERY_RE = re.compile(
    r"\b(?:what|which)\s+chapters?\b|\bchapters?\s+(?:are|does|in)\b",
    re.IGNORECASE,
)


def _page_span(match: SectionMatch) -> str:
    if match.page_start < 0:
        return ""
    if match.page_end > match.page_start:
        return f"pp. {match.page_start}-{match.page_end}"
    return f"p. {match.page_start}"


def _unique_section_matches(matches: list[SectionMatch]) -> list[SectionMatch]:
    by_key: dict[tuple[str, str], SectionMatch] = {}
    unique: list[SectionMatch] = []
    for match in sorted(matches, key=lambda m: (m.rank, m.section_path.count(">"), m.page_start)):
        _number, prefix, label = extract_numbered_section_prefix(
            match.section_path,
            match.title,
        )
        key = (match.document_id, prefix or match.section_path)
        if key in by_key:
            existing = by_key[key]
            if match.page_start >= 0 and (
                existing.page_start < 0 or match.page_start < existing.page_start
            ):
                existing.page_start = match.page_start
            if match.page_end > existing.page_end:
                existing.page_end = match.page_end
            existing.chunk_count += match.chunk_count
            continue
        normalized = SectionMatch(
            document_id=match.document_id,
            section_path=prefix or match.section_path,
            title=label or match.title,
            page_start=match.page_start,
            page_end=match.page_end,
            chunk_count=match.chunk_count,
            rank=match.rank,
            match_type=match.match_type,
        )
        by_key[key] = normalized
        unique.append(normalized)
    return unique


def _common_top_section_label(matches: list[SectionMatch]) -> str:
    labels: list[str] = []
    for match in matches:
        path = match.section_path or match.title
        if not path:
            continue
        top = path.split(">")[0].strip()
        if top:
            labels.append(top)
    if not labels:
        return ""
    first = labels[0]
    if all(label.lower() == first.lower() for label in labels):
        return first
    return ""


def _try_answer_structural_query(query: str, tool_results: ToolResults) -> dict | None:
    """Answer title/page queries directly from section matches."""
    matches = _unique_section_matches(tool_results.section_matches)
    if _CHAPTER_LIST_QUERY_RE.search(query or "") and _has_unreliable_structure(tool_results):
        return {
            "type": "section_not_found",
            "answer": (
                "There is not enough information to identify reliable chapter "
                "headings in the selected source. I cannot identify a trustworthy "
                "chapter outline, so I should not infer chapter titles from the "
                "available TOC or page-range metadata."
            ),
        }

    if _SECTION_DISCUSSION_QUERY_RE.search(query or "") and matches:
        common_label = _common_top_section_label(matches)
        if len(matches) == 1:
            match = matches[0]
            label = common_label or section_label_for_display(match.section_path, match.title)
            page = _page_span(match)
            suffix = f" ({page})." if page else "."
            return {
                "type": "section_match",
                "answer": f"The relevant section is **{label}**{suffix}",
            }
        lines = (
            [f"The relevant section is **{common_label}**. Subsections include:"]
            if common_label
            else ["Relevant sections include:"]
        )
        for match in matches[:5]:
            label = section_label_for_display(match.section_path, match.title)
            page = _page_span(match)
            suffix = f" ({page})" if page else ""
            lines.append(f"- **{label}**{suffix}")
        return {"type": "section_match", "answer": "\n".join(lines)}

    section_ref = _SECTION_REF_RE.search(query or "")
    if not section_ref:
        return None

    wants_title = bool(_STRUCTURAL_TITLE_RE.search(query or ""))
    wants_page = bool(_STRUCTURAL_PAGE_RE.search(query or ""))
    if not wants_title and not wants_page:
        return None

    target = f"Chapter {section_ref.group(1)}"
    if not matches:
        if _has_unreliable_structure(tool_results):
            return {
                "type": "section_not_found",
                "answer": (
                    f"I cannot identify a reliable **{target}** heading in the "
                    "selected source. This source does not expose a trustworthy "
                    "chapter outline, so I should not infer a chapter title or "
                    "chapter boundary from page-range labels."
                ),
            }
        return None

    if wants_title:
        if len(matches) == 1:
            match = matches[0]
            label = section_label_for_display(match.section_path, match.title)
            page = _page_span(match)
            suffix = f" ({page})." if page else "."
            return {
                "type": "section_title",
                "answer": f"{target} is titled **{label}**{suffix}",
            }
        lines = [f"I found multiple matches for {target}:"]
        for match in matches[:5]:
            label = section_label_for_display(match.section_path, match.title)
            page = _page_span(match)
            location = f", {page}" if page else ""
            lines.append(f"- **{label}** (`{match.document_id}`{location})")
        return {"type": "section_title", "answer": "\n".join(lines)}

    if wants_page:
        match = matches[0]
        label = section_label_for_display(match.section_path, match.title)
        if match.page_start < 0:
            return None
        page = f"p. {match.page_start}"
        span = _page_span(match)
        suffix = f" (section span: {span})" if span and span != page else ""
        return {
            "type": "section_page",
            "answer": f"{target}, **{label}**, starts at {page}{suffix}.",
        }

    return None


def _has_unreliable_structure(tool_results: ToolResults) -> bool:
    containers = list(tool_results.metadata) + list(tool_results.outline)
    if not containers:
        return False
    return any(
        is_unreliable_structure_status(item.get("structure_status"))
        for item in containers
    )


def _with_unresolved_section_disclaimer(
    query: str,
    tool_results: ToolResults,
    answer_text: str,
) -> str:
    """Prefix content answers when a requested chapter had no reliable match."""
    section_ref = _SECTION_REF_RE.search(query or "")
    if not section_ref:
        return answer_text
    if not _SECTION_CONTENT_RE.search(query or ""):
        return answer_text
    if tool_results.section_matches:
        return answer_text
    if not tool_results.chunks:
        return answer_text
    if not _has_unreliable_structure(tool_results):
        return answer_text

    target = f"Chapter {section_ref.group(1)}"
    prefix = (
        f"Note: I could not identify a reliable **{target}** boundary in the "
        "selected source, so this answer is based on semantically retrieved "
        "passages rather than a confirmed chapter section.\n\n"
    )
    if answer_text.startswith(prefix):
        return answer_text
    return prefix + answer_text
