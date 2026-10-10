"""Structure reliability metadata for ingested sources.

This module keeps "can we trust chapter/section structure?" separate from
"are the retrieval chunks usable?".  No-TOC documents can still have good
chunks, but QA must not infer chapter titles from page-window labels.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .models import ChunkingResult, TocEntry
from .structure_status import StructureStatus

STRUCTURE_FIELDS = (
    "chunking_mode",
    "structure_status",
    "embedded_toc_present",
    "synthetic_outline_mode",
    "structure_confidence",
    "structure_user_note",
    "structure_llm_instruction",
)

# Backward-compatible status sets (derived from the single source of truth).
# StructureStatus.is_reliable is the canonical check; these sets remain for
# internal use within this module only.
RELIABLE_STRUCTURE_STATUSES = {StructureStatus.EMBEDDED_TOC}
UNRELIABLE_STRUCTURE_STATUSES = {
    StructureStatus.PAGE_WINDOW_OUTLINE,
    StructureStatus.NO_RELIABLE_OUTLINE,
    StructureStatus.UNKNOWN_LEGACY,
    StructureStatus.AUTO_DETECTED_OUTLINE,
}

WRAPPER_TOC_TITLE_RE = re.compile(
    r"^(?:toc|table\s+of\s+contents|contents?|articles?|features?)$",
    re.IGNORECASE,
)

MIN_PAGES_FOR_TOC_DENSITY_GATE = 50
MIN_TOC_ENTRIES_PER_100_PAGES = 3.0
WRAPPER_SPAN_RATIO = 0.80


@dataclass(frozen=True)
class StructureMetadata:
    chunking_mode: str = ""
    structure_status: str = "unknown_legacy"
    embedded_toc_present: bool | None = None
    synthetic_outline_mode: str | None = None
    structure_confidence: str | None = None
    structure_user_note: str | None = None
    structure_llm_instruction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


UNKNOWN_LEGACY_STRUCTURE = StructureMetadata(
    structure_status=StructureStatus.UNKNOWN_LEGACY,
    structure_user_note="Structure unknown; reprocess to detect outline reliability.",
    structure_llm_instruction=(
        "This document's outline reliability is unknown. Do not assume chapter "
        "titles or chapter boundaries unless they are explicitly present in "
        "retrieved text or section metadata."
    ),
)


def build_structure_metadata(result: ChunkingResult) -> StructureMetadata:
    """Map chunker output to document-level structure reliability metadata."""
    mode = result.synthetic_outline_mode or None
    chunking_mode = result.chunking_mode or result.strategy_used or ""

    if result.strategy_used == "toc":
        if _has_degenerate_embedded_toc(result):
            return StructureMetadata(
                chunking_mode=chunking_mode,
                structure_status=StructureStatus.NO_RELIABLE_OUTLINE,
                embedded_toc_present=True,
                synthetic_outline_mode="degenerate_toc",
                structure_confidence="low",
                structure_user_note="Degenerate embedded TOC",
                structure_llm_instruction=(
                    "This document exposes an embedded TOC, but it is too sparse "
                    "or wrapper-like to be trusted as chapter structure. Do not "
                    "infer chapter titles or chapter boundaries from it; answer "
                    "only from retrieved passages and page metadata."
                ),
            )
        return StructureMetadata(
            chunking_mode=chunking_mode,
            structure_status=StructureStatus.EMBEDDED_TOC,
            embedded_toc_present=True,
            synthetic_outline_mode=None,
            structure_confidence="high",
            structure_user_note="Original table of contents",
            structure_llm_instruction=(
                "This document has a usable embedded PDF outline. Chapter and "
                "section headings are reliable."
            ),
        )

    if result.strategy_used == "synthetic_outline" and mode == "detected_headings":
        return StructureMetadata(
            chunking_mode=chunking_mode,
            structure_status=StructureStatus.AUTO_DETECTED_OUTLINE,
            embedded_toc_present=bool(result.embedded_toc_present),
            synthetic_outline_mode=mode,
            structure_confidence="medium",
            structure_user_note="Auto-detected outline",
            structure_llm_instruction=(
                "This document has no usable embedded TOC; its outline was "
                "inferred from visible headings. Treat section headings as "
                "mostly reliable, but allow for missed or mistaken headings."
            ),
        )

    if result.strategy_used == "synthetic_outline" and mode == "page_windows":
        return StructureMetadata(
            chunking_mode=chunking_mode,
            structure_status=StructureStatus.PAGE_WINDOW_OUTLINE,
            embedded_toc_present=bool(result.embedded_toc_present),
            synthetic_outline_mode=mode,
            structure_confidence="low",
            structure_user_note="Page-range outline",
            structure_llm_instruction=(
                "This document has no reliable chapter structure. Page-window "
                "labels such as 'Pages 12-19' are not chapters. Do not infer "
                "chapter titles or boundaries from page numbers; answer only "
                "from retrieved passages."
            ),
        )

    if result.strategy_used in ("token_fallback", "layout_hybrid"):
        return StructureMetadata(
            chunking_mode=chunking_mode,
            structure_status=StructureStatus.NO_RELIABLE_OUTLINE,
            embedded_toc_present=bool(result.embedded_toc_present),
            synthetic_outline_mode=mode or "none",
            structure_confidence="none",
            structure_user_note="No reliable outline",
            structure_llm_instruction=(
                "This document has no reliable outline. Do not infer chapter "
                "information; answer only from retrieved passages and page "
                "metadata."
            ),
        )

    return StructureMetadata(
        chunking_mode=chunking_mode,
        structure_status=StructureStatus.UNKNOWN_LEGACY,
        embedded_toc_present=(
            bool(result.embedded_toc_present)
            if result.embedded_toc_present is not None
            else None
        ),
        synthetic_outline_mode=mode,
        structure_confidence=None,
        structure_user_note=UNKNOWN_LEGACY_STRUCTURE.structure_user_note,
        structure_llm_instruction=UNKNOWN_LEGACY_STRUCTURE.structure_llm_instruction,
    )


def document_structure_fields(row: dict[str, Any] | None) -> dict[str, Any]:
    """Return stable structure fields for API/tool responses from a DB row."""
    if not row:
        return UNKNOWN_LEGACY_STRUCTURE.to_dict()

    data = UNKNOWN_LEGACY_STRUCTURE.to_dict()
    for field in STRUCTURE_FIELDS:
        if field not in row:
            continue
        value = row.get(field)
        if value is None:
            continue
        if field == "embedded_toc_present":
            value = bool(value)
        data[field] = value
    return data


def _flatten_toc(entries: list[TocEntry]) -> list[TocEntry]:
    flat: list[TocEntry] = []
    for entry in entries or []:
        flat.append(entry)
        flat.extend(_flatten_toc(entry.children))
    return flat


def _entry_page_span(entry: TocEntry, page_count: int) -> int:
    start = max(1, int(entry.page_start or 1))
    end = entry.page_end if entry.page_end is not None else page_count
    end = max(start, min(int(end), page_count))
    return end - start + 1


def _has_degenerate_embedded_toc(result: ChunkingResult) -> bool:
    """Return True when an embedded TOC is a wrapper, not a real outline."""
    page_count = len(result.pages)
    flat_entries = _flatten_toc(result.toc)
    entry_count = len(flat_entries) or result.toc_entries_total
    if entry_count <= 0 or page_count <= 0:
        return False

    top_level_count = len(result.toc)
    has_children = any(entry.children for entry in flat_entries)

    if entry_count == 1 and flat_entries:
        entry = flat_entries[0]
        title = (entry.title or "").strip()
        span_ratio = _entry_page_span(entry, page_count) / page_count
        if WRAPPER_TOC_TITLE_RE.match(title):
            return True
        if page_count >= MIN_PAGES_FOR_TOC_DENSITY_GATE and span_ratio >= WRAPPER_SPAN_RATIO:
            return True

    if page_count >= MIN_PAGES_FOR_TOC_DENSITY_GATE:
        entries_per_100_pages = entry_count / (page_count / 100.0)
        sparse = entries_per_100_pages < MIN_TOC_ENTRIES_PER_100_PAGES
        shallow = top_level_count <= 2 and not has_children
        wrapper_titles = bool(result.toc) and all(
            WRAPPER_TOC_TITLE_RE.match((entry.title or "").strip())
            for entry in result.toc
        )
        if sparse and (shallow or wrapper_titles):
            return True

    return False


def apply_structure_metadata_to_chunks(
    result: ChunkingResult,
    metadata: StructureMetadata | None = None,
) -> None:
    """Add v2 structure metadata to each chunk's JSON metadata."""
    metadata = metadata or build_structure_metadata(result)
    if metadata.structure_status == StructureStatus.EMBEDDED_TOC:
        outline_source = "embedded_toc"
    elif metadata.structure_status == StructureStatus.AUTO_DETECTED_OUTLINE:
        outline_source = "synthetic_heading"
    elif metadata.structure_status == StructureStatus.PAGE_WINDOW_OUTLINE:
        outline_source = "page_window"
    else:
        outline_source = "none"

    chunking_strategy = (
        "toc_based"
        if result.strategy_used in {"toc", "synthetic_outline"}
        else "token_fallback"
    )

    for chunk in result.chunks:
        section_label = " > ".join(chunk.section_path) if chunk.section_path else chunk.title
        chunk.metadata.setdefault("outline_source", outline_source)
        chunk.metadata.setdefault("chunking_strategy", chunking_strategy)
        chunk.metadata.setdefault("section_label", section_label)


def is_unreliable_structure_status(status: str | None) -> bool:
    """True when the structure status is not reliable for chapter inference.

    Delegates to StructureStatus.is_reliable (the single source of truth).
    Unknown / None values are treated as unreliable.
    """
    if not status:
        return True
    try:
        return not StructureStatus(status).is_reliable
    except ValueError:
        return True


def is_reliable_structure_status(status: str | None) -> bool:
    """True only when the structure status is reliable for chapter inference.

    Delegates to StructureStatus.is_reliable (the single source of truth).
    """
    if not status:
        return False
    try:
        return StructureStatus(status).is_reliable
    except ValueError:
        return False
