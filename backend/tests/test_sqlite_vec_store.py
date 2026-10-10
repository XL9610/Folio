"""Tests for the sqlite-vec dense index store."""

from __future__ import annotations

import sqlite3

import pytest

import sqlite_vec

from app.services.index.sqlite_vec_store import SqliteVecStore


@pytest.fixture
def vec_store():
    """In-memory SQLite with the vec extension loaded, wired into the store."""
    from app.db import session as db_session

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    original_conn = getattr(db_session._local, "conn", None)
    db_session._local.conn = conn
    yield SqliteVecStore()
    db_session._local.conn = original_conn
    conn.close()


def _chunk(cid: str, doc: str, nb: str = "nb1", page: int = 1) -> dict:
    return {
        "chunk_id": cid,
        "document_id": doc,
        "notebook_id": nb,
        "title": f"T-{cid}",
        "section_path": "1.1",
        "page_start": page,
        "page_end": page,
        "chunk_strategy": "pdf",
        "source_type": "pdf",
    }


def test_upsert_and_search_roundtrip(vec_store):
    vec_store.ensure_collection(4)
    chunks = [_chunk("a", "doc-1"), _chunk("b", "doc-1"), _chunk("c", "doc-2")]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    vec_store.upsert_chunks(chunks, vectors)

    results = vec_store.search([1.0, 0.0, 0.0, 0.0], "nb1", 2)
    assert [r["chunk_id"] for r in results] == ["a", "b"]
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert results[0]["document_id"] == "doc-1"
    assert results[0]["page_start"] == 1


def test_search_filters_by_document_ids(vec_store):
    vec_store.ensure_collection(4)
    vec_store.upsert_chunks(
        [_chunk("a", "doc-1"), _chunk("c", "doc-2")],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    )
    results = vec_store.search(
        [1.0, 0.0, 0.0, 0.0], "nb1", 10, document_ids=["doc-2"]
    )
    assert [r["chunk_id"] for r in results] == ["c"]


def test_upsert_is_idempotent(vec_store):
    vec_store.ensure_collection(4)
    vec_store.upsert_chunks([_chunk("a", "doc-1")], [[1.0, 0.0, 0.0, 0.0]])
    vec_store.upsert_chunks([_chunk("a", "doc-1")], [[0.0, 1.0, 0.0, 0.0]])
    results = vec_store.search([0.0, 1.0, 0.0, 0.0], "nb1", 10)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "a"


def test_delete_by_document_id(vec_store):
    vec_store.ensure_collection(4)
    vec_store.upsert_chunks(
        [_chunk("a", "doc-1"), _chunk("c", "doc-2")],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    )
    vec_store.delete_by_document_id("doc-1")
    results = vec_store.search([1.0, 0.0, 0.0, 0.0], "nb1", 10)
    assert [r["chunk_id"] for r in results] == ["c"]


def test_ensure_collection_dimension_mismatch(vec_store):
    vec_store.ensure_collection(4)
    with pytest.raises(ValueError):
        vec_store.ensure_collection(8)


def test_health_check(vec_store):
    healthy, msg, _latency = vec_store.health_check()
    assert healthy
    assert msg == "ok"
