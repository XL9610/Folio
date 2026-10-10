"""Tests for Module 1.3 — PdfTextbookExtractor with fake parser and TOC extractor."""

from __future__ import annotations

import asyncio

import pytest

from app.services.extraction.base import ExtractionError
from app.services.extraction.models import ExtractedDocument, SourceInput
from app.services.extraction.pdf import (
    PdfTextbookExtractor,
    is_text_poor_pdf,
    pdf_text_stats,
)
from app.services.pdf_ingestion.models import DocumentPage, TocEntry

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakePDFParser:
    """Returns pre-configured DocumentPage lists."""

    def __init__(self, pages: list[DocumentPage] | None = None) -> None:
        self._pages = pages or []
        self.last_path: str | None = None

    def parse(self, file_path: str) -> list[DocumentPage]:
        self.last_path = file_path
        return list(self._pages)


class FakeTOCExtractor:
    """Returns pre-configured TocEntry lists."""

    def __init__(self, toc: list[TocEntry] | None = None, should_fail: bool = False) -> None:
        self._toc = toc or []
        self._should_fail = should_fail
        self.last_path: str | None = None

    def extract(self, file_path: str) -> list[TocEntry]:
        self.last_path = file_path
        if self._should_fail:
            raise RuntimeError("Simulated TOC extraction failure")
        return list(self._toc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pages(texts: list[str]) -> list[DocumentPage]:
    return [
        DocumentPage(page_number=i + 1, text=t)
        for i, t in enumerate(texts)
    ]


def _make_toc(entries: list[tuple[str, int, int]]) -> list[TocEntry]:
    return [
        TocEntry(title=title, level=level, page_start=page)
        for title, level, page in entries
    ]


# ---------------------------------------------------------------------------
# pdf_text_stats
# ---------------------------------------------------------------------------


class TestPdfTextStats:
    def test_normal_pages(self) -> None:
        pages = _make_pages(["Hello world", "Another page", ""])
        stats = pdf_text_stats(pages)
        assert stats["page_count"] == 3
        assert stats["total_text_chars"] == 23
        assert stats["blank_page_count"] == 1
        assert stats["blank_page_ratio"] == pytest.approx(1 / 3)

    def test_empty_list(self) -> None:
        stats = pdf_text_stats([])
        assert stats["page_count"] == 0
        assert stats["blank_page_ratio"] == 0.0


# ---------------------------------------------------------------------------
# is_text_poor_pdf
# ---------------------------------------------------------------------------


class TestIsTextPoorPdf:
    def test_empty_pages_is_poor(self) -> None:
        assert is_text_poor_pdf([]) is True

    def test_low_char_count_is_poor(self) -> None:
        pages = _make_pages(["ab"])
        assert is_text_poor_pdf(pages, min_total_chars=200) is True

    def test_high_blank_ratio_is_poor(self) -> None:
        pages = _make_pages([""] * 8 + ["Just one page with text" * 20])
        assert is_text_poor_pdf(pages, max_blank_page_ratio=0.6) is True

    def test_normal_pdf_is_not_poor(self) -> None:
        pages = _make_pages(["A" * 500] * 10)
        assert is_text_poor_pdf(pages) is False


# ---------------------------------------------------------------------------
# PdfTextbookExtractor
# ---------------------------------------------------------------------------


class TestPdfTextbookExtractor:
    def test_can_handle_pdf_source_type(self) -> None:
        extractor = PdfTextbookExtractor()
        source = SourceInput(source_type="pdf", file_path="/tmp/doc.pdf")
        assert extractor.can_handle(source) is True

    def test_can_handle_pdf_extension(self) -> None:
        extractor = PdfTextbookExtractor()
        source = SourceInput(source_type="unknown", filename="book.pdf", file_path="/tmp/book.pdf")
        assert extractor.can_handle(source) is True

    def test_rejects_non_pdf(self) -> None:
        extractor = PdfTextbookExtractor()
        source = SourceInput(source_type="epub", filename="book.epub")
        assert extractor.can_handle(source) is False

    def test_extract_returns_pages(self) -> None:
        pages = _make_pages(["Page 1 content", "Page 2 content"])
        parser = FakePDFParser(pages=pages)
        toc = _make_toc([("Chapter 1", 1, 1)])
        toc_extractor = FakeTOCExtractor(toc=toc)

        extractor = PdfTextbookExtractor(parser=parser, toc_extractor=toc_extractor)
        source = SourceInput(source_type="pdf", file_path="/tmp/test.pdf", filename="test.pdf")

        doc = asyncio.run(extractor.extract(source))
        assert isinstance(doc, ExtractedDocument)
        assert doc.pages is not None
        assert len(doc.pages) == 2
        assert doc.pages[0].page_number == 1
        assert doc.pages[0].text == "Page 1 content"

    def test_extract_maps_raw_toc(self) -> None:
        pages = _make_pages(["content"] * 5)
        toc = _make_toc([("Ch 1", 1, 1), ("Ch 2", 1, 3)])
        parser = FakePDFParser(pages=pages)
        toc_extractor = FakeTOCExtractor(toc=toc)

        extractor = PdfTextbookExtractor(parser=parser, toc_extractor=toc_extractor)
        source = SourceInput(source_type="pdf", file_path="/tmp/test.pdf", filename="test.pdf")

        doc = asyncio.run(extractor.extract(source))
        assert doc.raw_toc is not None
        assert len(doc.raw_toc) == 2
        assert doc.raw_toc[0].title == "Ch 1"
        assert doc.raw_toc[1].title == "Ch 2"

    def test_extract_metadata_contains_pdf_info(self) -> None:
        pages = _make_pages(["Hello"] * 3)
        parser = FakePDFParser(pages=pages)
        toc_extractor = FakeTOCExtractor(toc=[])

        extractor = PdfTextbookExtractor(parser=parser, toc_extractor=toc_extractor)
        source = SourceInput(source_type="pdf", file_path="/tmp/test.pdf", filename="test.pdf")

        doc = asyncio.run(extractor.extract(source))
        assert doc.metadata["source_type"] == "pdf"
        assert doc.metadata["original_filename"] == "test.pdf"
        assert doc.metadata["page_count"] == 3
        assert doc.metadata["toc_count"] == 0

    def test_extract_empty_pages_raises(self) -> None:
        parser = FakePDFParser(pages=[])
        extractor = PdfTextbookExtractor(parser=parser)
        source = SourceInput(source_type="pdf", file_path="/tmp/empty.pdf")

        with pytest.raises(ExtractionError, match="No pages extracted"):
            asyncio.run(extractor.extract(source))

    def test_extract_missing_file_path_raises(self) -> None:
        extractor = PdfTextbookExtractor()
        source = SourceInput(source_type="pdf")

        with pytest.raises(ExtractionError, match="file_path"):
            asyncio.run(extractor.extract(source))

    def test_toc_extraction_failure_is_non_fatal(self) -> None:
        pages = _make_pages(["content"])
        parser = FakePDFParser(pages=pages)
        toc_extractor = FakeTOCExtractor(should_fail=True)

        extractor = PdfTextbookExtractor(parser=parser, toc_extractor=toc_extractor)
        source = SourceInput(source_type="pdf", file_path="/tmp/test.pdf", filename="test.pdf")

        doc = asyncio.run(extractor.extract(source))
        assert doc.raw_toc is None
        assert len(doc.pages) == 1  # Pages still extracted

    def test_low_text_warning(self) -> None:
        pages = _make_pages(["x"])
        parser = FakePDFParser(pages=pages)
        extractor = PdfTextbookExtractor(parser=parser)
        source = SourceInput(source_type="pdf", file_path="/tmp/sparse.pdf", filename="sparse.pdf")

        doc = asyncio.run(extractor.extract(source))
        assert any("low text" in w.lower() or "fallback" in w.lower() for w in doc.warnings)

    def test_extraction_engine_is_set(self) -> None:
        pages = _make_pages(["text"])
        parser = FakePDFParser(pages=pages)
        extractor = PdfTextbookExtractor(parser=parser)
        source = SourceInput(source_type="pdf", file_path="/tmp/test.pdf")

        doc = asyncio.run(extractor.extract(source))
        assert doc.extraction_engine == "pymupdf_textbook"

    def test_full_text_joins_page_text(self) -> None:
        pages = _make_pages(["First page.", "Second page."])
        parser = FakePDFParser(pages=pages)
        extractor = PdfTextbookExtractor(parser=parser)
        source = SourceInput(source_type="pdf", file_path="/tmp/test.pdf")

        doc = asyncio.run(extractor.extract(source))
        assert doc.full_text == "First page.\nSecond page."
