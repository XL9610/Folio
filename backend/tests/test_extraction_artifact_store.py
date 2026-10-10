"""Tests for Module 1.6 — Extraction Artifact Store, repository, and schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.db.schema import create_tables
from app.db.session import get_connection
from app.repositories.extraction_repository import (
    delete_document_extraction,
    get_document_extraction,
    upsert_document_extraction,
)
from app.services.extraction.artifact_store import ExtractionArtifactStore
from app.services.extraction.models import (
    AudioSegment,
    ExtractedDocument,
    SlideContent,
)
from app.services.pdf_ingestion.models import DocumentPage, TocEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> ExtractionArtifactStore:
    return ExtractionArtifactStore(base_dir=tmp_path / "extraction_artifacts")


@pytest.fixture
def doc_id() -> str:
    return "test-doc-001"


@pytest.fixture
def text_document() -> ExtractedDocument:
    return ExtractedDocument(
        title="Readme",
        full_text="Hello, world!",
        extraction_engine="plain_text",
        metadata={"source_type": "text", "original_filename": "readme.txt"},
        warnings=["Truncated"],
    )


@pytest.fixture
def pdf_document() -> ExtractedDocument:
    pages = [
        DocumentPage(page_number=1, text="Page one content", metadata={"width": 612.0, "height": 792.0}),
        DocumentPage(page_number=2, text="Page two content"),
    ]
    toc = [
        TocEntry(title="Chapter 1", level=1, page_start=1, page_end=5, children=[
            TocEntry(title="Section 1.1", level=2, page_start=2, page_end=3),
        ]),
    ]
    return ExtractedDocument(
        title="Test Book",
        full_text="Page one content\nPage two content",
        pages=pages,
        raw_toc=toc,
        extraction_engine="pymupdf_textbook",
        metadata={"source_type": "pdf", "original_filename": "book.pdf", "page_count": 2},
    )


@pytest.fixture
def markdown_document() -> ExtractedDocument:
    return ExtractedDocument(
        title="Notes",
        full_text="# Header\n\nBody text.",
        markdown="# Header\n\nBody text.",
        extraction_engine="plain_text",
        metadata={"source_type": "text", "original_filename": "notes.md"},
    )


@pytest.fixture
def slides_document() -> ExtractedDocument:
    slides = [
        SlideContent(slide_number=1, title="Title Slide", body_text="Welcome", notes="Intro notes"),
        SlideContent(slide_number=2, title="Content", body_text="Details", notes=""),
    ]
    return ExtractedDocument(
        title="Deck",
        full_text="Title Slide\nWelcome\nContent\nDetails",
        slides=slides,
        extraction_engine="python-pptx",
        metadata={"source_type": "pptx"},
    )


@pytest.fixture
def segments_document() -> ExtractedDocument:
    segments = [
        AudioSegment(start_time=0.0, end_time=30.0, text="Introduction", speaker="Dr. Smith"),
    ]
    return ExtractedDocument(
        title="Lecture 1",
        full_text="Introduction",
        segments=segments,
        extraction_engine="whisper",
        metadata={"source_type": "audio"},
    )


# ---------------------------------------------------------------------------
# Save tests
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_writes_content_txt(self, store: ExtractionArtifactStore, doc_id: str, text_document: ExtractedDocument) -> None:
        paths = store.save(doc_id, text_document)
        assert paths.content_txt.is_file()
        assert paths.content_txt.read_text(encoding="utf-8") == "Hello, world!"

    def test_save_writes_extracted_json(self, store: ExtractionArtifactStore, doc_id: str, text_document: ExtractedDocument) -> None:
        paths = store.save(doc_id, text_document)
        assert paths.extracted_json.is_file()
        data = json.loads(paths.extracted_json.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert data["title"] == "Readme"
        assert data["extraction_engine"] == "plain_text"
        assert data["warnings"] == ["Truncated"]
        assert data["structure_summary"]["full_text_chars"] == 13

    def test_save_writes_content_md_when_markdown_present(self, store: ExtractionArtifactStore, doc_id: str, markdown_document: ExtractedDocument) -> None:
        paths = store.save(doc_id, markdown_document)
        assert paths.content_md is not None
        assert paths.content_md.is_file()
        assert paths.content_md.read_text(encoding="utf-8") == "# Header\n\nBody text."

    def test_save_no_content_md_when_markdown_absent(self, store: ExtractionArtifactStore, doc_id: str, text_document: ExtractedDocument) -> None:
        paths = store.save(doc_id, text_document)
        assert paths.content_md is None

    def test_save_writes_pages_jsonl_for_pdf(self, store: ExtractionArtifactStore, doc_id: str, pdf_document: ExtractedDocument) -> None:
        paths = store.save(doc_id, pdf_document)
        assert paths.pages_jsonl is not None
        assert paths.pages_jsonl.is_file()
        lines = paths.pages_jsonl.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        page1 = json.loads(lines[0])
        assert page1["page_number"] == 1
        assert page1["text"] == "Page one content"
        assert page1["metadata"]["width"] == 612.0

    def test_save_no_pages_jsonl_for_non_pdf(self, store: ExtractionArtifactStore, doc_id: str, text_document: ExtractedDocument) -> None:
        paths = store.save(doc_id, text_document)
        assert paths.pages_jsonl is None


# ---------------------------------------------------------------------------
# TOC serialization
# ---------------------------------------------------------------------------


class TestTocSerialization:
    def test_raw_toc_serializes_recursively(self, store: ExtractionArtifactStore, doc_id: str, pdf_document: ExtractedDocument) -> None:
        paths = store.save(doc_id, pdf_document)
        data = json.loads(paths.extracted_json.read_text(encoding="utf-8"))
        toc_list = data["raw_toc"]
        assert len(toc_list) == 1
        assert toc_list[0]["title"] == "Chapter 1"
        assert toc_list[0]["level"] == 1
        assert len(toc_list[0]["children"]) == 1
        assert toc_list[0]["children"][0]["title"] == "Section 1.1"
        assert toc_list[0]["children"][0]["level"] == 2


# ---------------------------------------------------------------------------
# Slide serialization
# ---------------------------------------------------------------------------


class TestSlideSerialization:
    def test_slides_serialize_with_slide_numbers(self, store: ExtractionArtifactStore, doc_id: str, slides_document: ExtractedDocument) -> None:
        paths = store.save(doc_id, slides_document)
        data = json.loads(paths.extracted_json.read_text(encoding="utf-8"))
        assert len(data["slides"]) == 2
        assert data["slides"][0]["slide_number"] == 1
        assert data["slides"][0]["title"] == "Title Slide"
        assert data["slides"][1]["slide_number"] == 2
        assert data["slides"][1]["notes"] == ""


# ---------------------------------------------------------------------------
# Segment serialization
# ---------------------------------------------------------------------------


class TestSegmentSerialization:
    def test_segments_serialize(self, store: ExtractionArtifactStore, doc_id: str, segments_document: ExtractedDocument) -> None:
        paths = store.save(doc_id, segments_document)
        data = json.loads(paths.extracted_json.read_text(encoding="utf-8"))
        assert len(data["segments"]) == 1
        assert data["segments"][0]["start_time"] == 0.0
        assert data["segments"][0]["speaker"] == "Dr. Smith"


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_passes_when_files_exist(self, store: ExtractionArtifactStore, doc_id: str, text_document: ExtractedDocument) -> None:
        store.save(doc_id, text_document)
        ok, errors = store.validate(doc_id)
        assert ok is True
        assert errors == []

    def test_validate_fails_when_directory_missing(self, store: ExtractionArtifactStore, doc_id: str) -> None:
        ok, errors = store.validate(doc_id)
        assert ok is False
        assert any("missing" in e.lower() for e in errors)

    def test_validate_fails_when_required_file_missing(self, store: ExtractionArtifactStore, doc_id: str, text_document: ExtractedDocument) -> None:
        store.save(doc_id, text_document)
        (store.artifact_dir(doc_id) / "content.txt").unlink()
        ok, errors = store.validate(doc_id)
        assert ok is False
        assert any("content.txt" in e for e in errors)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_artifact_directory(self, store: ExtractionArtifactStore, doc_id: str, text_document: ExtractedDocument) -> None:
        store.save(doc_id, text_document)
        assert store.artifact_dir(doc_id).is_dir()
        store.delete(doc_id)
        assert not store.artifact_dir(doc_id).exists()

    def test_delete_nonexistent_is_noop(self, store: ExtractionArtifactStore, doc_id: str) -> None:
        # Should not raise
        store.delete(doc_id)


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------


class TestArtifactPaths:
    def test_root_dir_matches_store_base(self, store: ExtractionArtifactStore, doc_id: str, text_document: ExtractedDocument) -> None:
        paths = store.save(doc_id, text_document)
        assert paths.root_dir == store.artifact_dir(doc_id)


# ---------------------------------------------------------------------------
# Repository tests (require schema)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _init_schema() -> None:
    """Ensure tables exist for repository tests."""
    conn = get_connection()
    create_tables(conn)


def _ensure_source_document(doc_id: str, notebook_id: str = "nb-test") -> None:
    """Create a minimal source_document row so FK constraints are satisfied."""
    conn = get_connection()
    # Ensure notebook exists
    conn.execute(
        "INSERT OR IGNORE INTO notebooks (id, name) VALUES (?, ?)",
        (notebook_id, "Test Notebook"),
    )
    # Ensure source document exists
    conn.execute(
        """INSERT OR IGNORE INTO source_documents
           (id, notebook_id, filename, display_title, page_count, content_hash)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (doc_id, notebook_id, "test.pdf", "Test Doc", 0, f"hash-{doc_id}"),
    )
    conn.commit()


class TestExtractionRepository:
    def test_upsert_and_get(self) -> None:
        _ensure_source_document("doc-1")
        row = upsert_document_extraction(
            document_id="doc-1",
            extraction_engine="pymupdf_textbook",
            full_text_path="artifacts/doc-1/content.txt",
            markdown_path=None,
            pages_path="artifacts/doc-1/pages.jsonl",
            metadata_json='{"source_type":"pdf"}',
            warnings_json='["warning 1"]',
        )
        assert row["document_id"] == "doc-1"
        assert row["extraction_engine"] == "pymupdf_textbook"

        fetched = get_document_extraction("doc-1")
        assert fetched is not None
        assert fetched["full_text_path"] == "artifacts/doc-1/content.txt"
        assert fetched["pages_path"] == "artifacts/doc-1/pages.jsonl"

    def test_upsert_overwrites_existing(self) -> None:
        _ensure_source_document("doc-2")
        upsert_document_extraction(
            document_id="doc-2",
            extraction_engine="ebooklib",
            full_text_path="old/path.txt",
            markdown_path=None,
            pages_path=None,
            metadata_json="{}",
            warnings_json="[]",
        )
        upsert_document_extraction(
            document_id="doc-2",
            extraction_engine="ebooklib_v2",
            full_text_path="new/path.txt",
            markdown_path="new/md.md",
            pages_path=None,
            metadata_json='{"v":2}',
            warnings_json='["w"]',
        )
        row = get_document_extraction("doc-2")
        assert row is not None
        assert row["extraction_engine"] == "ebooklib_v2"
        assert row["full_text_path"] == "new/path.txt"
        assert row["markdown_path"] == "new/md.md"

    def test_get_nonexistent_returns_none(self) -> None:
        assert get_document_extraction("nonexistent") is None

    def test_delete_removes_row(self) -> None:
        _ensure_source_document("doc-3")
        upsert_document_extraction(
            document_id="doc-3",
            extraction_engine="kreuzberg",
            full_text_path="p.txt",
            markdown_path=None,
            pages_path=None,
            metadata_json="{}",
            warnings_json="[]",
        )
        assert delete_document_extraction("doc-3") is True
        assert get_document_extraction("doc-3") is None

    def test_delete_nonexistent_returns_false(self) -> None:
        assert delete_document_extraction("never-existed") is False
