"""Tests for Module 1.1 extraction contracts — models, helpers, import boundaries."""

from __future__ import annotations

import pytest

from app.services.extraction.models import (
    AudioSegment,
    ExtractedDocument,
    SlideContent,
    SourceInput,
    extracted_structure_summary,
    source_display_name,
    source_extension,
)
from app.services.pdf_ingestion.models import DocumentPage, TocEntry

# ---------------------------------------------------------------------------
# SourceInput minimal construction
# ---------------------------------------------------------------------------

class TestSourceInput:
    def test_minimal_construction(self) -> None:
        s = SourceInput(source_type="pdf")
        assert s.source_type == "pdf"
        assert s.file_path is None
        assert s.filename == ""

    def test_file_backed_construction(self) -> None:
        s = SourceInput(
            source_type="pdf",
            file_path="/tmp/doc.pdf",
            filename="doc.pdf",
            notebook_id="nb1",
            mime_type="application/pdf",
        )
        assert s.file_path == "/tmp/doc.pdf"
        assert s.mime_type == "application/pdf"
        assert s.notebook_id == "nb1"

    def test_url_construction(self) -> None:
        s = SourceInput(source_type="url", url="https://example.com/article")
        assert s.url == "https://example.com/article"

    def test_raw_text_construction(self) -> None:
        s = SourceInput(source_type="text", raw_text="Hello world", filename="note.txt")
        assert s.raw_text == "Hello world"

    def test_metadata_defaults_to_dict(self) -> None:
        s = SourceInput(source_type="pdf")
        assert s.metadata == {}

    def test_custom_metadata(self) -> None:
        s = SourceInput(source_type="epub", metadata={"author": "Test"})
        assert s.metadata["author"] == "Test"

    def test_frozen(self) -> None:
        s = SourceInput(source_type="pdf")
        with pytest.raises(Exception):  # noqa: B017
            s.source_type = "epub"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SlideContent
# ---------------------------------------------------------------------------

class TestSlideContent:
    def test_minimal(self) -> None:
        sc = SlideContent(slide_number=1)
        assert sc.slide_number == 1
        assert sc.title == ""
        assert sc.body_text == ""
        assert sc.notes == ""

    def test_full(self) -> None:
        sc = SlideContent(
            slide_number=3,
            title="Key Findings",
            body_text="Some body text.",
            notes="Speaker notes.",
            metadata={"layout": "title_and_content"},
        )
        assert sc.slide_number == 3
        assert sc.title == "Key Findings"
        assert sc.body_text == "Some body text."
        assert sc.notes == "Speaker notes."


# ---------------------------------------------------------------------------
# AudioSegment
# ---------------------------------------------------------------------------

class TestAudioSegment:
    def test_minimal(self) -> None:
        seg = AudioSegment(start_time=0.0, end_time=10.5, text="Hello")
        assert seg.start_time == 0.0
        assert seg.end_time == 10.5
        assert seg.text == "Hello"
        assert seg.speaker == ""

    def test_with_speaker(self) -> None:
        seg = AudioSegment(start_time=1.0, end_time=5.0, text="Hi", speaker="Dr. Smith")
        assert seg.speaker == "Dr. Smith"


# ---------------------------------------------------------------------------
# ExtractedDocument
# ---------------------------------------------------------------------------

class TestExtractedDocument:
    def test_minimal_text_document(self) -> None:
        doc = ExtractedDocument(title="Note", full_text="Some text.")
        assert doc.title == "Note"
        assert doc.full_text == "Some text."
        assert doc.markdown is None
        assert doc.pages is None
        assert doc.extraction_engine == ""

    def test_pdf_document_with_pages(self) -> None:
        pages = [
            DocumentPage(page_number=1, text="Page 1 content"),
            DocumentPage(page_number=2, text="Page 2 content"),
        ]
        toc = [TocEntry(title="Chapter 1", level=1, page_start=1)]
        doc = ExtractedDocument(
            title="Book",
            full_text="Page 1 content\nPage 2 content",
            pages=pages,
            raw_toc=toc,
            extraction_engine="pymupdf_textbook",
            metadata={"source_type": "pdf", "original_filename": "book.pdf"},
            warnings=["Some TOC entries were filtered"],
        )
        assert doc.pages is not None
        assert len(doc.pages) == 2
        assert doc.raw_toc is not None
        assert len(doc.raw_toc) == 1
        assert doc.extraction_engine == "pymupdf_textbook"
        assert len(doc.warnings) == 1

    def test_markdown_document(self) -> None:
        doc = ExtractedDocument(
            title="Notes",
            full_text="Heading\nContent",
            markdown="# Heading\n\nContent",
            extraction_engine="plain_text",
        )
        assert doc.markdown == "# Heading\n\nContent"

    def test_slides_document(self) -> None:
        slides = [SlideContent(slide_number=1, title="Slide 1")]
        doc = ExtractedDocument(
            title="Deck", full_text="Slide 1", slides=slides, extraction_engine="python-pptx"
        )
        assert doc.slides is not None
        assert len(doc.slides) == 1

    def test_segments_document(self) -> None:
        segments = [AudioSegment(start_time=0.0, end_time=30.0, text="Intro")]
        doc = ExtractedDocument(
            title="Lecture",
            full_text="Intro",
            segments=segments,
            extraction_engine="whisper",
        )
        assert doc.segments is not None
        assert len(doc.segments) == 1

    def test_frozen(self) -> None:
        doc = ExtractedDocument(title="X", full_text="Y")
        with pytest.raises(Exception):  # noqa: B017
            doc.title = "Z"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# source_extension
# ---------------------------------------------------------------------------

class TestSourceExtension:
    def test_lowercase_pdf(self) -> None:
        assert source_extension(SourceInput(source_type="pdf", filename="Doc.PDF")) == ".pdf"

    def test_epub(self) -> None:
        assert source_extension(SourceInput(source_type="epub", filename="book.epub")) == ".epub"

    def test_no_extension(self) -> None:
        assert source_extension(SourceInput(source_type="text", filename="README")) == ""

    def test_from_file_path_fallback(self) -> None:
        s = SourceInput(source_type="pdf", file_path="/tmp/doc.pdf", filename="")
        assert source_extension(s) == ".pdf"

    def test_empty_when_no_filename_or_path(self) -> None:
        assert source_extension(SourceInput(source_type="url", url="https://example.com")) == ""


# ---------------------------------------------------------------------------
# source_display_name
# ---------------------------------------------------------------------------

class TestSourceDisplayName:
    def test_prefers_filename(self) -> None:
        s = SourceInput(source_type="pdf", filename="book.pdf", url="https://x.com")
        assert source_display_name(s) == "book.pdf"

    def test_falls_back_to_url(self) -> None:
        s = SourceInput(source_type="url", url="https://example.com/article")
        assert source_display_name(s) == "https://example.com/article"

    def test_pasted_text_fallback(self) -> None:
        s = SourceInput(source_type="text", raw_text="hello")
        assert source_display_name(s) == "pasted text"


# ---------------------------------------------------------------------------
# extracted_structure_summary
# ---------------------------------------------------------------------------

class TestExtractedStructureSummary:
    def test_empty_document(self) -> None:
        doc = ExtractedDocument(title="Empty", full_text="")
        summary = extracted_structure_summary(doc)
        assert summary == {
            "page_count": 0,
            "toc_count": 0,
            "slide_count": 0,
            "segment_count": 0,
            "has_markdown": False,
            "full_text_chars": 0,
        }

    def test_full_document(self) -> None:
        doc = ExtractedDocument(
            title="Full",
            full_text="A" * 100,
            markdown="# H1",
            pages=[DocumentPage(page_number=1, text="A" * 100)],
            raw_toc=[TocEntry(title="Ch1", level=1, page_start=1)],
            slides=[SlideContent(slide_number=1)],
            segments=[AudioSegment(start_time=0, end_time=1, text="x")],
        )
        summary = extracted_structure_summary(doc)
        assert summary["page_count"] == 1
        assert summary["toc_count"] == 1
        assert summary["slide_count"] == 1
        assert summary["segment_count"] == 1
        assert summary["has_markdown"] is True
        assert summary["full_text_chars"] == 100


# ---------------------------------------------------------------------------
# Import boundary checks — extraction models must not import forbidden modules
# ---------------------------------------------------------------------------

class TestImportBoundaries:
    def test_models_does_not_import_chunking(self) -> None:
        """Extraction models must not import chunking types as primary contracts."""
        # Verify that Chunk and ChunkingResult are NOT re-exported by extraction.models
        from app.services.extraction import models as m

        assert not hasattr(m, "Chunk")
        assert not hasattr(m, "ChunkingResult")

    def test_models_re_exports_pdf_ingestion_types(self) -> None:
        """DocumentPage and TocEntry should be accessible from extraction.models."""
        from app.services.extraction import models as m

        assert hasattr(m, "DocumentPage")
        assert hasattr(m, "TocEntry")


# ============================================================================
# Module 1.2 — Content Extractor Interface tests
# ============================================================================

import asyncio  # noqa: E402

from app.services.extraction.base import (  # noqa: E402
    ContentExtractor,
    ExtractionError,
    ExtractorDependencyError,
    UnsupportedSourceError,
)


class FakeExtractor(ContentExtractor):
    """Minimal concrete extractor for testing the interface contract."""

    engine_name = "fake_test"

    def can_handle(self, source: SourceInput) -> bool:
        return source.source_type == "text"

    async def extract(self, source: SourceInput) -> ExtractedDocument:
        return ExtractedDocument(
            title="Fake",
            full_text=source.raw_text or "",
            extraction_engine=self.engine_name,
        )


class TestContentExtractorInterface:
    def test_fake_extractor_is_selectable(self) -> None:
        extractor = FakeExtractor()
        assert extractor.engine_name == "fake_test"

    def test_fake_can_handle_matching_source(self) -> None:
        extractor = FakeExtractor()
        source = SourceInput(source_type="text", raw_text="hi")
        assert extractor.can_handle(source) is True

    def test_fake_rejects_non_matching_source(self) -> None:
        extractor = FakeExtractor()
        source = SourceInput(source_type="pdf")
        assert extractor.can_handle(source) is False

    def test_fake_extract_returns_extracted_document(self) -> None:
        extractor = FakeExtractor()
        source = SourceInput(source_type="text", raw_text="Hello, world!")
        doc = asyncio.run(extractor.extract(source))
        assert isinstance(doc, ExtractedDocument)
        assert doc.full_text == "Hello, world!"
        assert doc.extraction_engine == "fake_test"


class TestAsyncSyncWrapper:
    """Test that sync libraries can be wrapped via asyncio.to_thread."""

    def test_sync_helper_wrapped_through_to_thread(self) -> None:
        """Simulate the pattern: async extract calls asyncio.to_thread(sync_fn, source)."""

        def _sync_extract(source: SourceInput) -> ExtractedDocument:
            return ExtractedDocument(
                title="Sync",
                full_text=source.raw_text or "",
                extraction_engine="sync_wrapper",
            )

        class SyncWrappedExtractor(ContentExtractor):
            engine_name = "sync_wrapper"

            def can_handle(self, source: SourceInput) -> bool:
                return source.source_type == "text"

            async def extract(self, source: SourceInput) -> ExtractedDocument:
                return await asyncio.to_thread(_sync_extract, source)

        extractor = SyncWrappedExtractor()
        doc = asyncio.run(extractor.extract(SourceInput(source_type="text", raw_text="x")))
        assert doc.full_text == "x"
        assert doc.extraction_engine == "sync_wrapper"


class TestExtractionErrors:
    def test_extraction_error_is_catchable_as_base(self) -> None:
        with pytest.raises(ExtractionError):
            raise ExtractorDependencyError("missing python-docx")

    def test_unsupported_source_is_extraction_error(self) -> None:
        with pytest.raises(ExtractionError):
            raise UnsupportedSourceError("bad format")

    def test_extractor_dependency_error_message(self) -> None:
        err = ExtractorDependencyError(
            "python-docx is required for DOCX extraction. "
            "Install with: pip install -r backend/requirements-extractors.txt"
        )
        assert "python-docx" in str(err)
        assert "requirements-extractors.txt" in str(err)
