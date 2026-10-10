from __future__ import annotations

import pytest

from app.services.ingestion import service
from app.services.ingestion.index_text import (
    CONTEXT_MODE_METADATA,
    IndexText,
    build_index_text,
    compose_indexed_text,
)
from app.services.extraction.models import ExtractedDocument
from app.services.pdf_ingestion.models import Chunk, ChunkingResult, DocumentPage


@pytest.fixture
def _test_db():
    """Create an in-memory SQLite database for ingestion integration tests."""
    import sqlite3
    from app.db import session as db_session
    from app.db.schema import create_tables

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    original_conn = getattr(db_session._local, "conn", None)
    db_session._local.conn = conn
    create_tables(conn)
    yield conn
    db_session._local.conn = original_conn
    conn.close()


class _FakeChunker:
    def chunk(self, *_args, **_kwargs) -> ChunkingResult:
        return ChunkingResult(
            document_id="book.pdf",
            pages=[DocumentPage(page_number=1, text="page text")],
            chunks=[
                Chunk(
                    text="chunk text",
                    document_id="book.pdf",
                    chunk_index=0,
                    title="Chapter 1",
                    page_start=1,
                    page_end=1,
                )
            ],
            strategy_used="toc",
            chunking_mode="hierarchical_toc",
            toc_entries_total=11,
            toc_entries_used=9,
            first_chunk_title="Chapter 1",
            body_start_title="Chapter 1",
            body_end_title="Appendix",
            front_matter_skipped_count=2,
        )


def test_duplicate_ingest_preserves_chunker_toc_metadata(monkeypatch, tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_text("not used by fake chunker", encoding="utf-8")

    existing_doc = {
        "id": "existing-doc",
        "display_title": "Existing Book",
        "page_count": 123,
        "chunk_count": 45,
        "chunking_mode": "hierarchical_toc",
        "structure_status": "embedded_toc",
        "embedded_toc_present": 1,
        "synthetic_outline_mode": None,
        "structure_confidence": "high",
        "structure_user_note": "Original table of contents",
        "structure_llm_instruction": "Reliable outline.",
    }
    monkeypatch.setattr(service.document_repository, "get_document_by_hash", lambda *_args: existing_doc)

    # Provide pre-extracted document to bypass real PyMuPDF extraction
    fake_extracted = ExtractedDocument(
        title="Existing Book",
        full_text="page text",
        extraction_engine="pymupdf_textbook",
        metadata={"source_type": "pdf"},
    )

    result = service.ingest_source(
        source=service.SourceInput(
            source_type="pdf",
            file_path=str(pdf),
            filename="book.pdf",
            notebook_id="nb1",
        ),
        force_reingest=False,
        chunker_selector=service.ChunkerSelector(pdf_chunker=_FakeChunker()),
        embedder=object(),
        dense_store=object(),
        extracted_document=fake_extracted,
    )

    assert result.status == "already_ingested"
    assert result.document_id == "existing-doc"
    assert result.source_type == "pdf"
    assert result.extraction_engine == "pymupdf_textbook"


# ---------------------------------------------------------------------------
# M2 - Index Text Builder tests
# ---------------------------------------------------------------------------


def test_build_index_text_with_section_path():
    """Prefix emitted for non-empty section path."""
    result = build_index_text("My Book", "Chapter 1 > Section A", "hello world")
    assert result.context_prefix == "[My Book > Chapter 1 > Section A]"
    assert result.context_mode == "metadata"
    assert result.indexed_text == "[My Book > Chapter 1 > Section A]\n\nhello world"


def test_build_index_text_trims_title_and_path():
    """Title and path are trimmed, chunk text is preserved exactly."""
    result = build_index_text("  My Book  ", "  Ch 1  ", "  chunk text  ")
    assert result.context_prefix == "[My Book > Ch 1]"
    assert result.indexed_text == "[My Book > Ch 1]\n\n  chunk text  "


def test_build_index_text_blank_section_path():
    """Blank section path returns canonical chunk text unchanged."""
    result = build_index_text("My Book", "", "hello world")
    assert result.context_prefix == ""
    assert result.context_mode == "metadata"
    assert result.indexed_text == "hello world"


def test_build_index_text_whitespace_section_path():
    """Whitespace-only section path returns canonical chunk text unchanged."""
    result = build_index_text("My Book", "   ", "hello world")
    assert result.context_prefix == ""
    assert result.indexed_text == "hello world"


def test_build_index_text_preserves_chunk_whitespace():
    """Chunk text whitespace is preserved exactly."""
    result = build_index_text("B", "C", "\n\n  indented text\n\n")
    assert result.indexed_text.endswith("\n\n  indented text\n\n")


def test_build_index_text_empty_title():
    """Empty document title is trimmed to empty string in prefix."""
    result = build_index_text("", "Ch 1", "text")
    assert result.context_prefix == "[ > Ch 1]"


def test_compose_indexed_text_with_prefix():
    """Reconstructs indexed text from stored prefix and canonical text."""
    result = compose_indexed_text("[B > C]", "chunk text")
    assert result == "[B > C]\n\nchunk text"


def test_compose_indexed_text_empty_prefix():
    """Returns chunk text when prefix is empty string."""
    assert compose_indexed_text("", "chunk text") == "chunk text"


def test_compose_indexed_text_whitespace_prefix():
    """Returns chunk text when prefix is whitespace-only."""
    assert compose_indexed_text("   ", "chunk text") == "chunk text"


def test_compose_indexed_text_roundtrip():
    """Roundtrip: build_index_text → store → compose_indexed_text reconstructs."""
    result = build_index_text("My Book", "Ch 1", "the chunk text")
    reconstructed = compose_indexed_text(result.context_prefix, "the chunk text")
    assert reconstructed == result.indexed_text


# ---------------------------------------------------------------------------
# M3 - Ingestion Enrichment Orchestrator tests
# ---------------------------------------------------------------------------


class _RecordingEmbedder:
    """Fake embedder that records texts it receives."""

    def __init__(self, vector_size: int = 4):
        self.texts: list[str] = []
        self._dim = vector_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts = list(texts)
        return [[0.1] * self._dim for _ in texts]


class _RecordingKeywordBackend:
    """Fake keyword backend that records indexed chunks."""

    def __init__(self):
        self.indexed: list[list[dict]] = []

    def index_chunks(self, chunks: list[dict]) -> None:
        self.indexed.append(list(chunks))

    def search(self, *args, **kwargs):
        return []

    def delete_document(self, doc_id: str) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "fake"

    @staticmethod
    def is_available() -> bool:
        return True


class _RecordingDenseStore:
    """Fake dense store that records payloads."""

    def __init__(self):
        self.payloads: list[list[dict]] = []
        self.vectors: list[list[list[float]]] = []

    def health_check(self, timeout: float = 3.0):
        return True, "ok", 5.0

    def ensure_collection(self, vector_size: int) -> None:
        pass

    def upsert_chunks(self, payloads: list[dict], vectors: list[list[float]]) -> None:
        self.payloads.append(list(payloads))
        self.vectors.append(list(vectors))

    def delete_by_document_id(self, doc_id: str) -> None:
        pass


class _FakeChunkerWithSectionPath:
    """Chunker that produces chunks with section_path."""

    def chunk(self, *_args, **_kwargs) -> ChunkingResult:
        import app.services.ingestion.service as _svc
        return ChunkingResult(
            document_id="book.pdf",
            pages=[DocumentPage(page_number=1, text="page text")],
            chunks=[
                Chunk(
                    text="chunk one text",
                    document_id="book.pdf",
                    chunk_index=0,
                    title="Chapter 1",
                    section_path=["Chapter 1", "Section A"],
                    page_start=1,
                    page_end=1,
                ),
                Chunk(
                    text="chunk two text",
                    document_id="book.pdf",
                    chunk_index=1,
                    title="",  # empty title → section_path stays empty
                    section_path=[],
                    page_start=2,
                    page_end=2,
                ),
            ],
            strategy_used="toc",
            chunking_mode="hierarchical_toc",
            toc_entries_total=5,
            toc_entries_used=3,
            first_chunk_title="Chapter 1",
            body_start_title="Chapter 1",
            body_end_title="Appendix",
            front_matter_skipped_count=1,
        )


def test_embedder_receives_prefixed_text_for_chunk_with_section_path(
    monkeypatch, tmp_path, _test_db,
):
    """Fake embedder receives prefixed indexed text for chunk with section_path."""
    from app.repositories.notebook_repository import create_notebook
    nb = create_notebook("Test Notebook")

    pdf = tmp_path / "book.pdf"
    pdf.write_text("not used by fake chunker", encoding="utf-8")

    embedder = _RecordingEmbedder()
    dense = _RecordingDenseStore()
    kw = _RecordingKeywordBackend()

    monkeypatch.setattr(service, "SentenceTransformerEmbedder", lambda: embedder)
    monkeypatch.setattr(service, "SqliteVecStore", lambda: dense)
    monkeypatch.setattr(
        "app.services.retrieval.keyword_backend.get_keyword_backend", lambda: kw,
    )

    # Bypass dedup
    monkeypatch.setattr(
        service.document_repository, "get_document_by_hash", lambda *_args: None,
    )

    fake_extracted = ExtractedDocument(
        title="My Textbook",
        full_text="page text",
        pages=[DocumentPage(page_number=1, text="page text")],
        extraction_engine="test",
        metadata={"source_type": "pdf"},
    )

    result = service.ingest_source(
        source=service.SourceInput(
            source_type="pdf",
            file_path=str(pdf),
            filename="book.pdf",
            notebook_id=nb["id"],
        ),
        force_reingest=False,
        chunker_selector=service.ChunkerSelector(
            pdf_chunker=_FakeChunkerWithSectionPath(),
        ),
        embedder=embedder,
        dense_store=dense,
        extracted_document=fake_extracted,
    )

    assert result.status == "completed"

    # Chunk 0 has section_path → embedder should get prefixed text
    assert len(embedder.texts) == 2
    assert embedder.texts[0].startswith("[My Textbook > Chapter 1 > Section A]")
    assert embedder.texts[0].endswith("chunk one text")

    # Chunk 1 has no section_path → embedder should get canonical text
    assert embedder.texts[1] == "chunk two text"

    # Keyword backend receives both canonical text and indexed_text
    assert len(kw.indexed) == 1
    kw_chunks = kw.indexed[0]
    assert kw_chunks[0]["text"] == "chunk one text"  # canonical
    assert kw_chunks[0]["indexed_text"] == embedder.texts[0]
    assert kw_chunks[1]["text"] == "chunk two text"
    assert kw_chunks[1]["indexed_text"] == "chunk two text"

    # Dense payloads exclude synthetic fields
    assert len(dense.payloads) == 1
    for p in dense.payloads[0]:
        assert "indexed_text" not in p
        assert "context_prefix" not in p
        assert "context_mode" not in p
        assert "source_type" in p
        assert p["source_type"] == "pdf"


def test_sqlite_rows_keep_canonical_text_and_store_context_prefix(
    monkeypatch, tmp_path, _test_db,
):
    """SQLite rows keep canonical text and store context_prefix."""
    from app.repositories.notebook_repository import create_notebook
    nb = create_notebook("Test Notebook")

    pdf = tmp_path / "book.pdf"
    pdf.write_text("not used", encoding="utf-8")

    embedder = _RecordingEmbedder()
    dense = _RecordingDenseStore()
    kw = _RecordingKeywordBackend()

    monkeypatch.setattr(service, "SentenceTransformerEmbedder", lambda: embedder)
    monkeypatch.setattr(service, "SqliteVecStore", lambda: dense)
    monkeypatch.setattr(
        "app.services.retrieval.keyword_backend.get_keyword_backend", lambda: kw,
    )
    monkeypatch.setattr(
        service.document_repository, "get_document_by_hash", lambda *_args: None,
    )

    fake_extracted = ExtractedDocument(
        title="My Textbook",
        full_text="page text",
        pages=[DocumentPage(page_number=1, text="page text")],
        extraction_engine="test",
        metadata={"source_type": "pdf"},
    )

    result = service.ingest_source(
        source=service.SourceInput(
            source_type="pdf",
            file_path=str(pdf),
            filename="book.pdf",
            notebook_id=nb["id"],
        ),
        force_reingest=False,
        chunker_selector=service.ChunkerSelector(
            pdf_chunker=_FakeChunkerWithSectionPath(),
        ),
        embedder=embedder,
        dense_store=dense,
        extracted_document=fake_extracted,
    )

    # Verify SQLite rows via repository
    from app.repositories.chunk_repository import get_chunks_by_document
    db_chunks = get_chunks_by_document(result.document_id)
    assert len(db_chunks) == 2

    # Chunk 0: canonical text preserved, context_prefix stored
    assert db_chunks[0]["text"] == "chunk one text"
    assert db_chunks[0]["context_prefix"] == "[My Textbook > Chapter 1 > Section A]"
    assert db_chunks[0]["context_mode"] == "metadata"

    # Chunk 1: no section_path → context_prefix is empty
    assert db_chunks[1]["text"] == "chunk two text"
    assert db_chunks[1]["context_prefix"] == ""
    assert db_chunks[1]["context_mode"] == "metadata"
