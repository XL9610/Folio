"""Tests for the derived summary-node schema and repository (roadmap Phase 3).

Covers: idempotent schema creation, scoped reads, uniqueness rules, atomic
replacement (previous build survives a failed replace), and orphan cleanup
on document deletion. No LLM is involved.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.db.schema import create_tables, run_migrations
from app.db import session as db_session
from app.repositories.chunk_repository import insert_chunks
from app.repositories.document_repository import (
    create_source_document,
    delete_document,
)
from app.repositories.notebook_repository import create_notebook
from app.repositories.summary_node_repository import (
    count_document_nodes,
    get_build_signature,
    get_document_nodes,
    replace_document_nodes,
)


@pytest.fixture
def test_db():
    """Create an in-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    original_conn = getattr(db_session._local, 'conn', None)
    db_session._local.conn = conn
    create_tables(conn)
    yield conn
    db_session._local.conn = original_conn
    conn.close()


@pytest.fixture
def seeded(test_db):
    nb = create_notebook("NB A")
    nb2 = create_notebook("NB B")
    doc = create_source_document(nb["id"], "book.pdf", page_count=5,
                                 content_hash="doc-hash-1")
    other_doc = create_source_document(nb["id"], "other.pdf", page_count=2,
                                       content_hash="doc-hash-2")
    return {"nb": nb, "nb2": nb2, "doc": doc, "other_doc": other_doc}


def _node(level="section", key="Chapter 1", ordinal=0, **overrides):
    node = {
        "level": level,
        "section_key": key,
        "ordinal": ordinal,
        "summary_json": json.dumps({"title": key, "summary": "s", "points": []}),
        "source_chunk_ids": ["c1"],
        "child_node_ids": [],
        "build_signature": "sig-1",
        "prompt_version": "1",
        "model_id": "test-model",
    }
    node.update(overrides)
    return node


def _hierarchy():
    return [
        _node("section", "Chapter 1", 0),
        _node("section", "Chapter 2", 1),
        _node("document", "__document__", 2,
              source_chunk_ids=["c1"], child_node_ids=["n1", "n2"]),
    ]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_table_exists_and_is_idempotent(test_db):
    tables = {r[0] for r in test_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert "document_summary_nodes" in tables
    create_tables(test_db)  # second call must not fail
    run_migrations(test_db)  # migration path must not fail on a fresh DB
    run_migrations(test_db)


def test_unique_index_enforces_one_node_per_key(seeded):
    replace_document_nodes(seeded["nb"]["id"], seeded["doc"]["id"], _hierarchy())
    with pytest.raises(sqlite3.IntegrityError):
        test_db_conn = db_session.get_connection()
        test_db_conn.execute(
            "INSERT INTO document_summary_nodes (id, notebook_id, document_id,"
            " level, section_key, ordinal, summary_json, source_chunk_ids_json,"
            " child_node_ids_json, build_signature, prompt_version, model_id)"
            " VALUES ('dup', ?, ?, 'section', 'Chapter 1', 9, '{}', '[]', '[]',"
            " 'sig', '1', 'm')",
            (seeded["nb"]["id"], seeded["doc"]["id"]),
        )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


def test_replace_and_read_roundtrip(seeded):
    nb_id = seeded["nb"]["id"]
    doc_id = seeded["doc"]["id"]
    count = replace_document_nodes(nb_id, doc_id, _hierarchy())
    assert count == 3

    nodes = get_document_nodes(nb_id, doc_id)
    assert len(nodes) == 3
    # Document card first, then sections in ordinal order.
    assert [n["level"] for n in nodes] == ["document", "section", "section"]
    assert nodes[1]["section_key"] == "Chapter 1"
    assert nodes[1]["source_chunk_ids"] == ["c1"]
    assert get_build_signature(nb_id, doc_id) == "sig-1"
    assert count_document_nodes(nb_id, doc_id) == {"section": 2, "document": 1}


def test_reads_enforce_notebook_scope(seeded):
    nb_id = seeded["nb"]["id"]
    doc_id = seeded["doc"]["id"]
    replace_document_nodes(nb_id, doc_id, _hierarchy())
    # Same document id, wrong notebook -> empty.
    assert get_document_nodes(seeded["nb2"]["id"], doc_id) == []
    assert get_build_signature(seeded["nb2"]["id"], doc_id) is None
    # Right notebook, other document -> empty.
    assert get_document_nodes(nb_id, seeded["other_doc"]["id"]) == []


def test_replace_is_atomic_previous_build_survives_failure(seeded):
    nb_id = seeded["nb"]["id"]
    doc_id = seeded["doc"]["id"]
    replace_document_nodes(nb_id, doc_id, _hierarchy())

    # A replacement containing a duplicate (document_id, level, section_key)
    # must fail the whole transaction and leave the previous build intact.
    bad = _hierarchy() + [_node("section", "Chapter 1", 9)]
    with pytest.raises(sqlite3.IntegrityError):
        replace_document_nodes(nb_id, doc_id, bad)

    nodes = get_document_nodes(nb_id, doc_id)
    assert len(nodes) == 3
    assert sorted(n["section_key"] for n in nodes) == [
        "Chapter 1", "Chapter 2", "__document__",
    ]


def test_replace_succeeds_after_previous_build(seeded):
    nb_id = seeded["nb"]["id"]
    doc_id = seeded["doc"]["id"]
    replace_document_nodes(nb_id, doc_id, _hierarchy())
    # Replacing with a single section node removes the old build entirely.
    replace_document_nodes(nb_id, doc_id, [_node("section", "Window 1", 0)])
    nodes = get_document_nodes(nb_id, doc_id)
    assert len(nodes) == 1
    assert nodes[0]["section_key"] == "Window 1"


def test_delete_document_removes_nodes(seeded):
    nb_id = seeded["nb"]["id"]
    doc_id = seeded["doc"]["id"]
    replace_document_nodes(nb_id, doc_id, _hierarchy())
    insert_chunks(doc_id, [{
        "chunk_index": 0, "text": "hello world text",
        "title": "", "section_path": "Chapter 1",
        "page_start": 1, "page_end": 1, "chunk_strategy": "test",
    }])
    assert delete_document(doc_id) is True
    assert count_document_nodes(nb_id, doc_id) == {}
    assert get_document_nodes(nb_id, doc_id) == []
