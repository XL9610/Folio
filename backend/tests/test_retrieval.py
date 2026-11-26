"""Unit tests for RRF fusion and hybrid retriever."""

from __future__ import annotations

import sqlite3

import pytest

from app.services.retrieval.retriever import HybridRetriever


class MockEmbedder:
    """Embedder that returns fixed-size vectors."""
    def __init__(self, dim: int = 384):
        self.dim = dim
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]
    def embed_query(self, query: str) -> list[float]:
        return [0.1] * self.dim


class MockDenseStore:
    """Returns predefined dense results."""
    def __init__(self, results: list[dict] | None = None):
        self.results = results or []
        self.deleted_document_ids: list[str] = []
    def ensure_collection(self, vector_size: int):
        pass
    def upsert_chunks(self, chunks, vectors):
        pass
    def delete_by_document_id(self, document_id: str):
        self.deleted_document_ids.append(document_id)
    def search(self, query_vector, notebook_id, top_k, document_ids=None):
        return self.results


def test_rrf_fuse_dense_only():
    """Dense-only results should produce correct RRF scores."""
    embedder = MockEmbedder()
    dense = MockDenseStore([
        {"chunk_id": "a", "document_id": "d1", "title": "T1", "score": 0.9, "text": "hello"},
        {"chunk_id": "b", "document_id": "d1", "title": "T2", "score": 0.7, "text": "world"},
    ])
    retriever = HybridRetriever(embedder, dense)
    # Patch _fts_search to return empty (no SQLite needed)
    original_fts = retriever._fts_search
    retriever._fts_search = lambda *a, **kw: ([], None)
    try:
        results = retriever.search("nb1", "test", top_k=5)
        assert len(results) == 2
        # Check RRF: rank 1 → 1/(60+1)=0.01639, rank 2 → 1/(60+2)=0.01613
        assert results[0].chunk_id == "a"
        assert results[1].chunk_id == "b"
        assert results[0].rrf_score > results[1].rrf_score
        assert results[0].text == "hello"
    finally:
        retriever._fts_search = original_fts


def test_rrf_fuse_fts_only():
    """FTS-only results should work without dense."""
    embedder = MockEmbedder()
    retriever = HybridRetriever(embedder, dense_store=None)

    # Patch FTS to return predefined results
    def mock_fts(notebook_id, query, top_k, document_ids=None):
        return [
            {"chunk_id": "x", "document_id": "d1", "text": "fts text",
             "title": "T", "section_path": "S", "page_start": 1, "page_end": 1,
             "score": -1.0},
        ], None
    retriever._fts_search = mock_fts
    # Patch hydrate to skip DB
    retriever._hydrate_from_db = lambda results: None

    try:
        results = retriever.search("nb1", "keyword", top_k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "x"
        assert results[0].text == "fts text"
        assert results[0].rrf_score > 0
    finally:
        retriever._fts_search = retriever._fts_search  # restore later


def test_rrf_fuse_dense_fts_overlap():
    """When dense and FTS hit same chunk, text from FTS should win."""
    embedder = MockEmbedder()
    dense = MockDenseStore([
        {"chunk_id": "shared", "document_id": "d1", "title": "From dense", "score": 0.9, "text": ""},
    ])
    retriever = HybridRetriever(embedder, dense)

    def mock_fts(notebook_id, query, top_k, document_ids=None):
        return [
            {"chunk_id": "shared", "document_id": "d1", "text": "real text from FTS",
             "title": "From FTS", "section_path": "Ch1", "page_start": 5, "page_end": 5,
             "score": -1.5},
        ], None
    retriever._fts_search = mock_fts
    retriever._hydrate_from_db = lambda results: None

    try:
        results = retriever.search("nb1", "test", top_k=5)
        assert len(results) == 1
        r = results[0]
        assert r.chunk_id == "shared"
        assert r.text == "real text from FTS"  # FTS text wins
        assert r.title == "From FTS"  # FTS title wins
        assert r.dense_score == 0.9
        assert r.fts_score == -1.5
        assert r.rrf_score > 0  # both ranks contribute
    finally:
        retriever._fts_search = retriever._fts_search


def test_embedding_failure_falls_back_to_fts():
    """When embedding fails, search should still return FTS results."""
    embedder = MockEmbedder()
    retriever = HybridRetriever(embedder, dense_store=None)

    # Make embedding fail
    retriever._embed_query_safe = lambda q: (None, None)

    def mock_fts(notebook_id, query, top_k, document_ids=None):
        return [
            {"chunk_id": "f1", "document_id": "d1", "text": "fallback text",
             "title": "T", "section_path": "S", "page_start": 1, "page_end": 1,
             "score": -2.0},
        ], None
    retriever._fts_search = mock_fts
    retriever._hydrate_from_db = lambda results: None

    try:
        results = retriever.search("nb1", "anything", top_k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "f1"
    finally:
        retriever._fts_search = retriever._fts_search


def test_search_with_meta_records_backend_errors():
    """Dense/keyword errors are surfaced in meta, not silently swallowed."""
    from app.services.retrieval.retriever import HybridRetriever

    class FailingEmbedder:
        def embed_query(self, query):
            raise RuntimeError("embed failed")

    class FailingKeyword:
        def name(self):
            return "failing"
        def search(self, *a, **kw):
            raise RuntimeError("keyword failed")

    retriever = HybridRetriever(FailingEmbedder(), dense_store=None, keyword_backend=FailingKeyword())
    chunks, meta = retriever.search_with_meta("nb1", "q", top_k=5)
    assert chunks == []
    assert "embedding_error" in meta
    assert meta.get("keyword_error")


def test_search_with_meta_zero_hits_no_error():
    """Both backends empty (no exception) is a genuine zero-hit, not an error."""
    from app.services.retrieval.retriever import HybridRetriever

    class EmptyKeyword:
        def name(self):
            return "empty"
        def search(self, *a, **kw):
            return []

    retriever = HybridRetriever(MockEmbedder(), dense_store=None, keyword_backend=EmptyKeyword())
    chunks, meta = retriever.search_with_meta("nb1", "q", top_k=5)
    assert chunks == []
    assert "embedding_error" not in meta
    assert "keyword_error" not in meta


def test_dedupe_removes_duplicates():
    """Duplicate chunks (same doc+title+pages) should be collapsed to highest score."""
    from app.services.retrieval.retriever import HybridRetriever
    from app.services.retrieval.base import RetrievedChunk

    results = [
        RetrievedChunk(chunk_id="a", document_id="d1", text="same", title="Ch1", page_start=1, page_end=1, rrf_score=0.5),
        RetrievedChunk(chunk_id="b", document_id="d1", text="same-dup", title="Ch1", page_start=1, page_end=1, rrf_score=0.3),
        RetrievedChunk(chunk_id="c", document_id="d1", text="diff", title="Ch2", page_start=2, page_end=2, rrf_score=0.4),
    ]

    deduped = HybridRetriever._dedupe_results(results)
    assert len(deduped) == 2
    # The higher-scoring duplicate should win (within same document)
    assert deduped[0].chunk_id == "a"  # rrf_score=0.5
    assert deduped[1].chunk_id == "c"  # rrf_score=0.4


def test_dedupe_all_unique():
    """No duplicates should all pass through."""
    from app.services.retrieval.retriever import HybridRetriever
    from app.services.retrieval.base import RetrievedChunk

    results = [
        RetrievedChunk(chunk_id="a", document_id="d1", text="a", title="T1", page_start=1, page_end=1, rrf_score=0.5),
        RetrievedChunk(chunk_id="b", document_id="d1", text="b", title="T2", page_start=2, page_end=2, rrf_score=0.3),
    ]

    deduped = HybridRetriever._dedupe_results(results)
    assert len(deduped) == len(results)


# ---------------------------------------------------------------------------
# M4 - Dense Index Writer Adapter tests
# ---------------------------------------------------------------------------


@pytest.fixture
def _m4_test_db():
    """In-memory SQLite DB with chunks containing context_prefix for M4 tests."""
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


def test_fts_only_hit_returns_canonical_text(_m4_test_db):
    """FTS-only hit can match context_prefix but returns canonical text."""
    from app.repositories.notebook_repository import create_notebook
    from app.repositories.document_repository import create_source_document
    from app.repositories.chunk_repository import insert_chunks, search_chunks_fts

    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")

    insert_chunks(doc["id"], [{
        "chunk_index": 0,
        "text": "canonical chunk text",
        "title": "Chapter 1",
        "section_path": "Chapter 1",
        "page_start": 1,
        "page_end": 1,
        "chunk_strategy": "toc",
        "context_prefix": "[Book > Ch1 > UNIQUEPREFIX]",
        "context_mode": "metadata",
    }])

    # FTS can find the chunk via the context_prefix token
    results = search_chunks_fts(nb["id"], "UNIQUEPREFIX", top_k=5)
    assert len(results) >= 1
    # But the returned text is canonical
    assert results[0]["text"] == "canonical chunk text"
    assert not results[0]["text"].startswith("[")


def test_dense_only_hydration_returns_canonical_text(_m4_test_db):
    """Dense-only hit hydration returns canonical SQLite chunks.text, not prefixed."""
    from app.repositories.notebook_repository import create_notebook
    from app.repositories.document_repository import create_source_document
    from app.repositories.chunk_repository import insert_chunks

    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")

    insert_chunks(doc["id"], [{
        "chunk_index": 0,
        "text": "canonical chunk text",
        "title": "Chapter 1",
        "section_path": "Chapter 1 > Section A",
        "page_start": 1,
        "page_end": 1,
        "chunk_strategy": "toc",
        "context_prefix": "[My Book > Chapter 1 > Section A]",
        "context_mode": "metadata",
    }])

    # Get the inserted chunk's ID
    from app.repositories.chunk_repository import get_chunks_by_document
    db_chunks = get_chunks_by_document(doc["id"])
    chunk_id = db_chunks[0]["id"]

    embedder = MockEmbedder()
    # Dense store returns result with empty text — hydration must fill it
    dense = MockDenseStore([
        {
            "chunk_id": chunk_id,
            "document_id": doc["id"],
            "title": "",
            "section_path": "",
            "page_start": 0,
            "page_end": 0,
            "score": 0.95,
            "text": "",  # dense-only: no text in payload
        },
    ])
    retriever = HybridRetriever(embedder, dense)

    # Patch FTS to return empty — pure dense-only path
    retriever._fts_search = lambda *a, **kw: ([], None)

    results = retriever.search(nb["id"], "test query", top_k=5)

    assert len(results) == 1
    # Hydration must return canonical text, not the prefixed version
    assert results[0].text == "canonical chunk text"
    assert not results[0].text.startswith("[")
    assert "[My Book" not in results[0].text
