"""Extraction data contracts — the durable boundary between extractors and chunkers.

All Phase 1 extractors produce ExtractedDocument and all chunkers consume it.
These types live in extraction to avoid coupling chunking output types
(Chunk, ChunkingResult) into extractor modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..pdf_ingestion.models import DocumentPage, TocEntry  # noqa: F401  # re-exported

SourceType = Literal[
    "pdf", "url", "text", "docx", "pptx", "epub", "audio", "video", "unknown"
]


@dataclass(frozen=True)
class SourceInput:
    """Self-describing input for any Phase 1 extractor.

    At least one content carrier must be set:
      file_path   — for file-backed sources (PDF, EPUB, PPTX, DOCX, etc.)
      url         — for HTTP/HTTPS sources
      raw_text    — for pasted or programmatic text input
    """

    source_type: SourceType
    file_path: str | None = None
    url: str | None = None
    raw_text: str | None = None
    filename: str = ""
    notebook_id: str = "default"
    mime_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SlideContent:
    """Content extracted from a single PPTX slide."""

    slide_number: int
    title: str = ""
    body_text: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioSegment:
    """Pre-transcribed audio/video segment.

    Phase 1 contract only — no transcription service is implemented.
    TimestampChunker is deferred until real transcription output is available.
    """

    start_time: float
    end_time: float
    text: str
    speaker: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedDocument:
    """Durable extraction output consumed by chunkers and persisted by artifact store.

    This is the handoff boundary between extraction and chunking.
    Phase 3 retry will resume from the persisted artifact without re-extracting.
    """

    title: str
    full_text: str
    markdown: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[DocumentPage] | None = None
    raw_toc: list[TocEntry] | None = None
    slides: list[SlideContent] | None = None
    segments: list[AudioSegment] | None = None
    extraction_engine: str = ""
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------

_KNOWN_EXTENSIONS: dict[str, SourceType] = {
    ".pdf": "pdf",
    ".epub": "epub",
    ".pptx": "pptx",
    ".docx": "docx",
    ".md": "text",
    ".txt": "text",
}


def source_extension(source: SourceInput) -> str:
    """Return lowercase extension with leading dot, or empty string."""
    if source.filename:
        return Path(source.filename).suffix.lower()
    if source.file_path:
        return Path(source.file_path).suffix.lower()
    return ""


def source_display_name(source: SourceInput) -> str:
    """Human-readable name preferring filename, then URL, then 'pasted text'."""
    if source.filename:
        return source.filename
    if source.url:
        return source.url
    return "pasted text"


def extracted_structure_summary(doc: ExtractedDocument) -> dict[str, Any]:
    """Count-oriented summary used in extracted.json artifact metadata."""
    return {
        "page_count": len(doc.pages) if doc.pages else 0,
        "toc_count": len(doc.raw_toc) if doc.raw_toc else 0,
        "slide_count": len(doc.slides) if doc.slides else 0,
        "segment_count": len(doc.segments) if doc.segments else 0,
        "has_markdown": doc.markdown is not None,
        "full_text_chars": len(doc.full_text),
    }
