"""Slide chunker — creates one Chunk per non-empty PPTX slide."""

from __future__ import annotations

from ..extraction.models import ExtractedDocument
from ..pdf_ingestion.models import Chunk, ChunkingResult
from ..ingestion.index_text import MAX_CHUNK_TEXT_TOKENS
from .base import DocumentChunker
from .token import _split_text_with_overlap


class SlideChunker(DocumentChunker):
    """Create one chunk per non-empty slide with locator metadata.

    Section path: [document.title, "Slide N", slide.title] without empty entries.
    page_start and page_end are -1 for non-page sources.
    """

    strategy_name = "slide"

    # ------------------------------------------------------------------
    # DocumentChunker interface
    # ------------------------------------------------------------------

    def can_chunk(self, document: ExtractedDocument) -> bool:
        return document.slides is not None and len(document.slides) > 0

    def chunk(self, document: ExtractedDocument, document_id: str) -> ChunkingResult:
        slides = document.slides or []
        chunks: list[Chunk] = []
        chunk_idx = 0

        for slide in slides:
            # Build slide text from title, body, and notes
            parts = [slide.title, slide.body_text]
            if slide.notes:
                parts.append(slide.notes)
            slide_text = "\n".join(p for p in parts if p)
            if not slide_text.strip():
                continue

            # Split over-long slides so no chunk exceeds the embedding window.
            segments = _split_text_with_overlap(
                slide_text,
                chunk_size=MAX_CHUNK_TEXT_TOKENS,
                overlap=40,
            )

            # Build section path
            section_path: list[str] = []
            if document.title:
                section_path.append(document.title)
            section_path.append(f"Slide {slide.slide_number}")
            if slide.title:
                section_path.append(slide.title)

            for segment in segments:
                chunks.append(
                    Chunk(
                        text=segment,
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        title=slide.title or f"Slide {slide.slide_number}",
                        section_path=section_path,
                        page_start=-1,
                        page_end=-1,
                        chunk_strategy=self.strategy_name,
                        metadata={
                            "source_type": "pptx",
                            "locator_type": "slide_number",
                            "slide_number": slide.slide_number,
                            "has_notes": bool(slide.notes),
                        },
                    )
                )
                chunk_idx += 1

        warnings: list[str] = []
        if not chunks:
            warnings.append("No non-empty slides found")

        return ChunkingResult(
            document_id=document_id,
            chunks=chunks,
            strategy_used=self.strategy_name,
            first_chunk_title=chunks[0].title if chunks else "",
            warnings=warnings,
        )
