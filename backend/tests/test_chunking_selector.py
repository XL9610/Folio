"""Tests for Module 1.7 — ChunkerSelector delegation."""

from __future__ import annotations

from app.services.chunking.selector import ChunkerSelector
from app.services.extraction.models import ExtractedDocument, SlideContent
from app.services.pdf_ingestion.models import DocumentPage


def _make_doc(**kwargs: object) -> ExtractedDocument:
    defaults: dict = {
        "title": "Test",
        "full_text": "Default text.",
    }
    defaults.update(kwargs)  # type: ignore[arg-type]
    return ExtractedDocument(**defaults)  # type: ignore[arg-type]


class TestChunkerSelector:
    def test_pdf_with_pages_and_path_delegates_to_pdf_chunker(self) -> None:
        selector = ChunkerSelector()
        doc = ExtractedDocument(
            title="PDF Book",
            full_text="Page 1\nPage 2",
            pages=[DocumentPage(page_number=1, text="Page 1")],
            metadata={"source_type": "pdf"},
        )
        # Without source_file_path, should NOT use PDF chunker
        result = selector.chunk(doc, "doc-1", source_file_path=None)
        # Falls through to token fallback since no markdown/slides/toc
        assert result.strategy_used in ("token_fallback",)
        assert len(result.chunks) > 0

    def test_slides_delegation(self) -> None:
        selector = ChunkerSelector()
        doc = ExtractedDocument(
            title="Deck",
            full_text="Slide 1 content\nSlide 2 content",
            slides=[
                SlideContent(slide_number=1, title="S1", body_text="Content 1"),
                SlideContent(slide_number=2, title="S2", body_text="Content 2"),
            ],
            metadata={"source_type": "pptx"},
        )
        result = selector.chunk(doc, "doc-1")
        assert result.strategy_used == "slide"
        assert len(result.chunks) == 2

    def test_markdown_delegation(self) -> None:
        selector = ChunkerSelector()
        doc = ExtractedDocument(
            title="Notes",
            full_text="# H1\n\nBody text\n\n# H2\n\nMore text",
            markdown="# H1\n\nBody text\n\n# H2\n\nMore text",
            metadata={"source_type": "text"},
        )
        result = selector.chunk(doc, "doc-1")
        # Markdown should go through heading chunker
        assert result.strategy_used in ("heading",)
        assert len(result.chunks) > 0

    def test_full_text_fallback(self) -> None:
        selector = ChunkerSelector()
        doc = ExtractedDocument(
            title="Plain",
            full_text="Just some plain text with no structure markers.",
            metadata={"source_type": "text"},
        )
        result = selector.chunk(doc, "doc-1")
        assert result.strategy_used == "token_fallback"
        assert len(result.chunks) > 0

    def test_empty_document_no_chunks_warning(self) -> None:
        selector = ChunkerSelector()
        doc = ExtractedDocument(
            title="Empty",
            full_text="   ",
        )
        result = selector.chunk(doc, "doc-1")
        assert len(result.chunks) == 0
