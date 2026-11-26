"""Tests for Module 1.5 — Extractor Registry."""

from __future__ import annotations

import pytest

from app.services.extraction.base import ContentExtractor
from app.services.extraction.models import ExtractedDocument, SourceInput
from app.services.extraction.registry import (
    ExtractorRegistry,
    ExtractorSelection,
    build_default_extractor_registry,
    infer_source_type,
)

# ---------------------------------------------------------------------------
# Fake extractors for deterministic testing
# ---------------------------------------------------------------------------


class FakePdfExtractor(ContentExtractor):
    engine_name = "fake_pdf"

    def can_handle(self, source: SourceInput) -> bool:
        return source.source_type == "pdf"

    async def extract(self, source: SourceInput) -> ExtractedDocument:
        return ExtractedDocument(title="pdf", full_text="")


class FakeEpubExtractor(ContentExtractor):
    engine_name = "fake_epub"

    def can_handle(self, source: SourceInput) -> bool:
        return source.source_type == "epub"

    async def extract(self, source: SourceInput) -> ExtractedDocument:
        return ExtractedDocument(title="epub", full_text="")


class FakeFallbackExtractor(ContentExtractor):
    engine_name = "fake_fallback"

    def can_handle(self, source: SourceInput) -> bool:
        return source.file_path is not None

    async def extract(self, source: SourceInput) -> ExtractedDocument:
        return ExtractedDocument(title="fallback", full_text="")


# ---------------------------------------------------------------------------
# ExtractorRegistry
# ---------------------------------------------------------------------------


class TestExtractorRegistry:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("pdf", "fake_pdf"),
            ("epub", "fake_epub"),
        ],
        ids=['selects_first_matching_extractor', 'selects_second_when_first_rejects'],
    )
    def test_extractor_registry(self, value, expected) -> None:
        registry = ExtractorRegistry(
            extractors=[FakePdfExtractor(), FakeEpubExtractor()],
        )
        selection = registry.select(SourceInput(source_type=value))
        assert selection.engine_name == expected

    def test_fallback_when_no_extractor_matches(self) -> None:
        registry = ExtractorRegistry(
            extractors=[FakePdfExtractor()],
            fallback_extractor=FakeFallbackExtractor(),
        )
        selection = registry.select(SourceInput(source_type="docx", file_path="/tmp/test.docx"))
        assert selection.engine_name == "fake_fallback"
        assert "fallback" in selection.fallback_reason.lower()

    def test_order_is_deterministic(self) -> None:
        # If both match, the first registered wins
        class MatchesEverything(ContentExtractor):
            engine_name = "greedy"

            def can_handle(self, source: SourceInput) -> bool:
                return True

            async def extract(self, source: SourceInput) -> ExtractedDocument:
                return ExtractedDocument(title="g", full_text="")

        registry = ExtractorRegistry(
            extractors=[FakePdfExtractor(), MatchesEverything()],
        )
        selection = registry.select(SourceInput(source_type="pdf"))
        assert selection.engine_name == "fake_pdf"  # PDF comes first

    def test_fallback_reason_empty_when_dedicated(self) -> None:
        registry = ExtractorRegistry(extractors=[FakePdfExtractor()])
        selection = registry.select(SourceInput(source_type="pdf"))
        assert selection.fallback_reason == ""


# ---------------------------------------------------------------------------
# build_default_extractor_registry
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    def test_creates_registry_with_all_extractors(self) -> None:
        registry = build_default_extractor_registry()
        assert registry is not None
        # Default registry should handle PDF via PdfTextbookExtractor
        selection = registry.select(SourceInput(source_type="pdf", file_path="/tmp/test.pdf"))
        assert selection.engine_name == "pymupdf_textbook"

    def test_pdf_selected_for_pdf(self) -> None:
        registry = build_default_extractor_registry()
        selection = registry.select(SourceInput(
            source_type="pdf", file_path="/tmp/doc.pdf", filename="doc.pdf",
        ))
        assert selection.engine_name == "pymupdf_textbook"

    def test_fallback_for_unknown(self) -> None:
        registry = build_default_extractor_registry()
        selection = registry.select(SourceInput(
            source_type="unknown", file_path="/tmp/file.xyz",
        ))
        # Kreuzberg fallback should handle unknown file-backed sources
        assert selection.engine_name in ("kreuzberg",)


# ---------------------------------------------------------------------------
# infer_source_type
# ---------------------------------------------------------------------------


class TestInferSourceType:
    def test_url_with_http(self) -> None:
        assert infer_source_type(url="https://example.com") == "url"

    @pytest.mark.parametrize(
        "value",
        [
            "hello",
            "content",
        ],
        ids=['raw_text_without_filename', 'text_priority_for_raw_text_no_file'],
    )
    def test_infer_source_type(self, value) -> None:
        assert infer_source_type(raw_text=value) == "text"

    def test_raw_text_with_filename_is_typed_by_extension(self) -> None:
        assert infer_source_type(filename="notes.md", raw_text="x") == "text"

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("book.pdf", "pdf"),
            ("book.epub", "epub"),
            ("deck.pptx", "pptx"),
            ("report.docx", "docx"),
            ("readme.txt", "text"),
            ("readme.md", "text"),
            ("DOC.PDF", "pdf"),
            ("data.xyz", "unknown"),
        ],
        ids=['pdf_extension', 'epub_extension', 'pptx_extension', 'docx_extension', 'txt_extension', 'md_extension', 'upper_case_extension', 'unknown_extension'],
    )
    def test_pdf_extension(self, value, expected) -> None:
        assert infer_source_type(filename=value) == expected

    def test_empty_all(self) -> None:
        assert infer_source_type() == "unknown"

    def test_url_priority_over_extension(self) -> None:
        # URL takes priority even if filename has an extension
        assert infer_source_type(url="https://example.com", filename="doc.pdf") == "url"


# ---------------------------------------------------------------------------
# ExtractorSelection
# ---------------------------------------------------------------------------


class TestExtractorSelection:
    def test_default_fallback_reason_empty(self) -> None:
        sel = ExtractorSelection(
            extractor=FakePdfExtractor(),
            engine_name="fake_pdf",
        )
        assert sel.fallback_reason == ""

    def test_fallback_reason_preserved(self) -> None:
        sel = ExtractorSelection(
            extractor=FakeFallbackExtractor(),
            engine_name="fake_fallback",
            fallback_reason="No match",
        )
        assert sel.fallback_reason == "No match"
