"""Chunker selector — pick the best chunker for an ExtractedDocument.

PDF compatibility: delegates PDF sources with pages+source_file_path to
the existing PDFChunker, preserving current chunking behavior.
"""

from __future__ import annotations

from ..extraction.models import ExtractedDocument
from ..pdf_ingestion.chunker import PDFChunker
from ..pdf_ingestion.models import ChunkingResult
from .base import DocumentChunker
from .epub import EpubChapterChunker
from .heading import HeadingChunker
from .pptx import SlideChunker
from .token import TokenFallbackChunker


class ChunkerSelector:
    """Select and apply the best chunker for an ExtractedDocument.

    Selection order:
      1. PDF file path + pages → existing PDFChunker
      2. slides → SlideChunker
      3. raw_toc + markdown → EpubChapterChunker
      4. markdown → HeadingChunker
      5. full_text → TokenFallbackChunker
    """

    def __init__(
        self,
        pdf_chunker: PDFChunker | None = None,
        epub_chunker: EpubChapterChunker | None = None,
        slide_chunker: SlideChunker | None = None,
        heading_chunker: HeadingChunker | None = None,
        token_chunker: TokenFallbackChunker | None = None,
    ) -> None:
        self._pdf_chunker = pdf_chunker or PDFChunker()
        self._epub_chunker = epub_chunker or EpubChapterChunker()
        self._slide_chunker = slide_chunker or SlideChunker()
        self._heading_chunker = heading_chunker or HeadingChunker()
        self._token_chunker = token_chunker or TokenFallbackChunker()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        document: ExtractedDocument,
        document_id: str,
        source_file_path: str | None = None,
    ) -> ChunkingResult:
        """Select and apply the best chunker for *document*."""

        # 1. PDF delegation — forward pre-extracted pages/TOC to avoid re-parsing
        if document.pages and source_file_path:
            return self._pdf_chunker.chunk(
                source_file_path,
                document_id=document_id,
                pages=document.pages,
                raw_toc=document.raw_toc,
            )

        # 2. Slides
        if self._slide_chunker.can_chunk(document):
            return self._slide_chunker.chunk(document, document_id)

        # 3. EPUB (TOC + markdown)
        if self._epub_chunker.can_chunk(document):
            return self._epub_chunker.chunk(document, document_id)

        # 4. Markdown headings
        if self._heading_chunker.can_chunk(document):
            return self._heading_chunker.chunk(document, document_id)

        # 5. Token fallback
        return self._token_chunker.chunk(document, document_id)
