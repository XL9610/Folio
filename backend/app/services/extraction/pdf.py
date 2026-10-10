"""PDF textbook extractor — wraps existing PyMuPDF parsing and TOC extraction.

Produces ExtractedDocument with pages and raw TOC, preserving existing behavior
without calling PDFChunker, TOC filtering, or LLM classification.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

from ..pdf_ingestion.models import DocumentPage, TocEntry
from ..pdf_ingestion.parser import PDFParser, extract_display_title
from ..pdf_ingestion.toc_extractor import TOCExtractor
from .base import ContentExtractor, ExtractionError
from .models import ExtractedDocument, SourceInput


class PdfTextbookExtractor(ContentExtractor):
    """Extract pages, raw TOC, and metadata from PDF using existing PyMuPDF path.

    This extractor does NOT:
      - call PDFChunker
      - filter front matter
      - call TOCFilter
      - classify TOC entries with LLM
      - create chunks
      - write files or database rows
    """

    engine_name = "pymupdf_textbook"

    def __init__(
        self,
        parser: PDFParser | None = None,
        toc_extractor: TOCExtractor | None = None,
    ) -> None:
        self._parser = parser or PDFParser()
        self._toc_extractor = toc_extractor or TOCExtractor()

    # ------------------------------------------------------------------
    # ContentExtractor interface
    # ------------------------------------------------------------------

    def can_handle(self, source: SourceInput) -> bool:
        """Accept PDF sources by source_type or file extension."""
        if source.source_type == "pdf":
            return True
        ext = Path(source.filename or source.file_path or "").suffix.lower()
        return ext == ".pdf"

    async def extract(self, source: SourceInput) -> ExtractedDocument:
        """Extract pages, raw TOC, and metadata from a PDF file."""
        file_path = _require_file_path(source)

        # Parse pages
        pages = self._parser.parse(file_path)
        if not pages:
            raise ExtractionError(f"No pages extracted from PDF: {file_path}")

        # Extract raw TOC (best-effort; failure does not block extraction)
        raw_toc: list[TocEntry] = []
        with suppress(Exception):
            raw_toc = self._toc_extractor.extract(file_path)

        # Build full text and display title
        full_text = "\n".join(p.text for p in pages)
        display_title = extract_display_title(file_path, filename=source.filename)

        # Build metadata
        metadata = _build_pdf_metadata(source, pages, raw_toc)

        # Build warnings
        warnings: list[str] = []
        if not raw_toc:
            warnings.append("No embedded TOC found in PDF.")
        if is_text_poor_pdf(pages):
            warnings.append(
                "PDF has very low text content. Consider using Kreuzberg fallback."
            )

        return ExtractedDocument(
            title=display_title or source.filename,
            full_text=full_text,
            pages=pages,
            raw_toc=raw_toc if raw_toc else None,
            extraction_engine=self.engine_name,
            metadata=metadata,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def pdf_text_stats(pages: list[DocumentPage]) -> dict[str, Any]:
    """Compute text statistics for a list of DocumentPage objects."""
    total_chars = sum(len(p.text) for p in pages)
    blank_count = sum(1 for p in pages if not p.text.strip())
    return {
        "page_count": len(pages),
        "total_text_chars": total_chars,
        "blank_page_count": blank_count,
        "blank_page_ratio": blank_count / len(pages) if pages else 0.0,
    }


def is_text_poor_pdf(
    pages: list[DocumentPage],
    min_total_chars: int = 200,
    max_blank_page_ratio: float = 0.6,
) -> bool:
    """Return True if the PDF has too little text for meaningful extraction."""
    if not pages:
        return True
    stats = pdf_text_stats(pages)
    text_poor: bool = stats["total_text_chars"] < min_total_chars
    blank_poor: bool = stats["blank_page_ratio"] > max_blank_page_ratio
    return text_poor or blank_poor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_file_path(source: SourceInput) -> str:
    """Validate and return the file_path, raising ExtractionError if missing."""
    if not source.file_path:
        raise ExtractionError("PDF extraction requires a file_path")
    return source.file_path


def _build_pdf_metadata(
    source: SourceInput,
    pages: list[DocumentPage],
    raw_toc: list[TocEntry],
) -> dict[str, Any]:
    """Build the metadata dict for a PDF extraction."""
    stats = pdf_text_stats(pages)
    return {
        "source_type": "pdf",
        "original_filename": source.filename or Path(source.file_path or "").name,
        "page_count": stats["page_count"],
        "toc_count": len(raw_toc),
        "total_text_chars": stats["total_text_chars"],
        "blank_page_count": stats["blank_page_count"],
        "blank_page_ratio": stats["blank_page_ratio"],
    }
