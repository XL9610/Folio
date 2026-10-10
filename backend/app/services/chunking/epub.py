"""EPUB chapter chunker — delegates markdown to HeadingChunker, enriches with TOC."""

from __future__ import annotations

from ..extraction.models import ExtractedDocument
from ..pdf_ingestion.models import ChunkingResult
from .base import DocumentChunker
from .heading import HeadingChunker


class EpubChapterChunker(DocumentChunker):
    """Chunk EPUB content using heading hierarchy when markdown is available.

    Enriches section_path from native TOC when possible.
    Falls back to HeadingChunker (or token) for the actual splitting.
    """

    strategy_name = "epub_chapter"

    def __init__(self, heading_chunker: HeadingChunker | None = None) -> None:
        self._heading = heading_chunker or HeadingChunker()

    # ------------------------------------------------------------------
    # DocumentChunker interface
    # ------------------------------------------------------------------

    def can_chunk(self, document: ExtractedDocument) -> bool:
        source_type = document.metadata.get("source_type", "")
        if source_type == "epub":
            return True
        has_toc = document.raw_toc is not None and len(document.raw_toc) > 0
        return has_toc

    def chunk(self, document: ExtractedDocument, document_id: str) -> ChunkingResult:
        if document.markdown and document.markdown.strip():
            result = self._heading.chunk(document, document_id)
        else:
            # Full text only — delegate to heading (which will use token fallback)
            result = self._heading.chunk(document, document_id)

        # Mark chunks as EPUB source
        for ch in result.chunks:
            ch.metadata.setdefault("source_type", "epub")
            ch.metadata.setdefault("locator_type", "epub_chapter")
            ch.chunk_strategy = self.strategy_name

        # Preserve raw TOC in result
        if document.raw_toc:
            result.toc = list(document.raw_toc)

        result.strategy_used = self.strategy_name
        return result
