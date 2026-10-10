"""Tests for SQLite database schema and repositories."""

from __future__ import annotations

import os
import tempfile

import pytest

from app.db.schema import create_tables
from app.repositories.notebook_repository import create_notebook, get_or_create_default, get_notebook
from app.repositories.document_repository import (
    create_source_document,
    update_document_status,
    get_document,
    get_document_by_hash,
    list_documents,
    delete_document,
)
from app.repositories.chunk_repository import insert_chunks, search_chunks_fts, get_chunks_by_document


@pytest.fixture
def test_db():
    """Create an in-memory SQLite database for testing."""
    import sqlite3
    from app.db import session as db_session

    # Override the thread-local connection to use in-memory DB
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    # Save original thread-local and replace
    original_conn = getattr(db_session._local, 'conn', None)
    db_session._local.conn = conn
    create_tables(conn)
    yield conn
    # Restore
    db_session._local.conn = original_conn
    conn.close()


def test_schema_tables_exist(test_db):
    """All tables should exist after initialisation."""
    cursor = test_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in cursor.fetchall()}
    expected = {
        "notebooks",
        "source_documents",
        "chunks",
        "ingestion_jobs",
        "review_items",
        "schema_version",
    }
    for name in expected:
        assert name in tables, f"Table '{name}' missing from schema"


def test_create_notebook(test_db):
    """create_notebook should insert and return the row."""
    nb = create_notebook("Test Book")
    assert nb["name"] == "Test Book"
    assert nb["id"]


def test_get_or_create_default(test_db):
    """get_or_create_default should be idempotent."""
    nb1 = get_or_create_default()
    nb2 = get_or_create_default()
    assert nb1["id"] == nb2["id"]
    assert nb1["name"] == "Default Notebook"


def test_create_source_document(test_db):
    """create_source_document should insert and return the row."""
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf", page_count=10)
    assert doc["filename"] == "test.pdf"
    assert doc["page_count"] == 10
    assert doc["status"] == "pending"
    assert doc["structure_status"] == "unknown_legacy"


def test_create_source_document_with_structure_metadata(test_db):
    """source_documents should persist structure reliability fields."""
    nb = create_notebook("Test")
    doc = create_source_document(
        nb["id"],
        "toc.pdf",
        page_count=10,
        chunking_mode="hierarchical_toc",
        structure_status="embedded_toc",
        embedded_toc_present=True,
        structure_confidence="high",
        structure_user_note="Original table of contents",
        structure_llm_instruction="Reliable outline.",
    )

    assert doc["chunking_mode"] == "hierarchical_toc"
    assert doc["structure_status"] == "embedded_toc"
    assert doc["embedded_toc_present"] == 1
    assert doc["structure_confidence"] == "high"


def test_update_document_status(test_db):
    """update_document_status should change status and chunk_count."""
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")
    update_document_status(doc["id"], "completed", chunk_count=5)
    updated = get_document(doc["id"])
    assert updated["status"] == "completed"
    assert updated["chunk_count"] == 5


def test_insert_and_search_chunks(test_db):
    """insert_chunks should write to SQLite; FTS should find them."""
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")

    chunks = [
        {
            "chunk_index": 0,
            "text": "Gradient descent minimizes the loss function.",
            "title": "Chapter 1",
            "section_path": "Chapter 1",
            "page_start": 1,
            "page_end": 1,
            "chunk_strategy": "toc",
            "metadata_json": "{}",
        },
        {
            "chunk_index": 1,
            "text": "The quick brown fox jumps over the lazy dog.",
            "title": "Chapter 2",
            "section_path": "Chapter 2",
            "page_start": 2,
            "page_end": 2,
            "chunk_strategy": "toc",
            "metadata_json": "{}",
        },
    ]
    count = insert_chunks(doc["id"], chunks)
    assert count == 2

    # Verify direct retrieval
    db_chunks = get_chunks_by_document(doc["id"])
    assert len(db_chunks) == 2

    # FTS search
    results = search_chunks_fts(nb["id"], '"gradient"', top_k=5)
    assert len(results) >= 1
    assert "gradient" in results[0]["text"].lower()


def test_insert_chunks_rolls_back_on_partial_failure(test_db):
    """A mid-batch insert failure must not leave earlier chunks committed."""
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")

    with pytest.raises(KeyError):
        insert_chunks(doc["id"], [
            {
                "chunk_index": 0,
                "text": "first chunk",
                "title": "Ch 1",
                "section_path": "Ch 1",
                "page_start": 1,
                "page_end": 1,
                "chunk_strategy": "toc",
            },
            {
                # Missing "text" key raises mid-batch.
                "chunk_index": 1,
                "title": "Ch 2",
            },
        ])

    # The first chunk must have been rolled back, not committed.
    assert len(get_chunks_by_document(doc["id"])) == 0


def test_get_document_by_hash(test_db):
    """get_document_by_hash should find document by notebook + content_hash."""
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf", content_hash="abc123")
    found = get_document_by_hash(nb["id"], "abc123")
    assert found is not None
    assert found["id"] == doc["id"]

    # Different hash should not match
    not_found = get_document_by_hash(nb["id"], "xyz789")
    assert not_found is None


def test_list_documents(test_db):
    """list_documents should return all docs in a notebook."""
    nb = create_notebook("Test")
    create_source_document(nb["id"], "a.pdf", content_hash="hash_a")
    create_source_document(nb["id"], "b.pdf", content_hash="hash_b")
    docs = list_documents(nb["id"])
    assert len(docs) == 2


def test_delete_document(test_db):
    """delete_document should remove doc and its chunks."""
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")
    insert_chunks(doc["id"], [{
        "chunk_index": 0, "text": "hello", "title": "T",
        "section_path": "T", "page_start": 1, "page_end": 1,
        "chunk_strategy": "toc", "metadata_json": "{}",
    }])

    assert delete_document(doc["id"]) is True
    assert get_document(doc["id"]) is None
    assert len(get_chunks_by_document(doc["id"])) == 0


def test_mark_interrupted_jobs_failed(test_db):
    """Startup recovery marks leftover queued/running jobs and documents failed."""
    from app.repositories.ingestion_job_repository import (
        create_job,
        update_job,
        get_job,
        mark_interrupted_jobs_failed,
    )

    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "stuck.pdf")
    test_db.execute(
        "UPDATE source_documents SET status = 'indexing' WHERE id = ?", (doc["id"],)
    )
    test_db.commit()

    stuck = create_job(nb["id"])
    update_job(stuck, status="running")
    done = create_job(nb["id"])
    update_job(done, status="completed", document_id=doc["id"])

    recovered = mark_interrupted_jobs_failed()

    assert recovered >= 1
    assert get_job(stuck)["status"] == "failed"
    assert get_job(done)["status"] == "completed"
    row = test_db.execute(
        "SELECT status FROM source_documents WHERE id = ?", (doc["id"],)
    ).fetchone()
    assert row["status"] == "failed"


def test_delete_vectors_when_table_missing_is_idempotent(test_db):
    """Dense cleanup succeeds even when the vec0 table hasn't been created yet."""
    from app.services.index.sqlite_vec_store import SqliteVecStore

    # create_tables does not create chunks_vec (it is lazily created on first
    # embed); an early-failed document must still be deletable/retryable.
    SqliteVecStore().delete_by_document_id("no-such-doc")


def test_display_title_stored_and_returned(test_db):
    """display_title should be stored and returned by get_document and list_documents."""
    nb = create_notebook("Test")
    doc = create_source_document(
        nb["id"], "ugly_name_z_library_sk.pdf",
        display_title="Beautiful Title: A Critical Anthology",
    )
    assert doc.get("display_title", "") == "Beautiful Title: A Critical Anthology"

    # get_document should return display_title
    fetched = get_document(doc["id"])
    assert fetched["display_title"] == "Beautiful Title: A Critical Anthology"

    # list_documents should include display_title
    docs = list_documents(nb["id"])
    assert docs[0].get("display_title", "") == "Beautiful Title: A Critical Anthology"


def test_display_title_empty_by_default(test_db):
    """display_title should be empty string for legacy insert without parameter."""
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")
    assert doc.get("display_title", "") == ""


def test_ingestion_result_has_display_title():
    """IngestionResult should include display_title field."""
    from app.services.ingestion.service import IngestionResult
    result = IngestionResult(
        document_id="test-id",
        notebook_id="nb1",
        filename="test.pdf",
        display_title="The Art of Art History: A Critical Anthology",
    )
    assert result.display_title == "The Art of Art History: A Critical Anthology"


# ---------------------------------------------------------------------------
# M1 - Chunk Context Storage Contract tests
# ---------------------------------------------------------------------------


def test_chunk_context_columns_defaults(test_db):
    """New chunks table defaults context_prefix='' and context_mode='metadata'."""
    from app.db.schema import _ensure_chunk_context_columns

    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")
    insert_chunks(doc["id"], [{
        "chunk_index": 0,
        "text": "canonical text",
        "title": "Ch 1",
        "section_path": "Ch 1",
        "page_start": 1,
        "page_end": 1,
        "chunk_strategy": "toc",
    }])

    rows = get_chunks_by_document(doc["id"])
    assert len(rows) == 1
    assert rows[0]["context_prefix"] == ""
    assert rows[0]["context_mode"] == "metadata"
    assert rows[0]["text"] == "canonical text"


def test_chunk_context_explicit_persistence(test_db):
    """Explicit context_prefix and context_mode values are persisted."""
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")
    insert_chunks(doc["id"], [{
        "chunk_index": 0,
        "text": "canonical text",
        "title": "Ch 1",
        "section_path": "Ch 1",
        "page_start": 1,
        "page_end": 1,
        "chunk_strategy": "toc",
        "context_prefix": "[My Book > Ch 1]",
        "context_mode": "metadata",
    }])

    rows = get_chunks_by_document(doc["id"])
    assert rows[0]["context_prefix"] == "[My Book > Ch 1]"
    assert rows[0]["context_mode"] == "metadata"


def test_chunk_context_migration_preserves_text(test_db):
    """Migration of old chunks table adds columns without rewriting chunk text."""
    # Simulate an old database: drop the newly created chunks table first
    test_db.execute("DROP TABLE IF EXISTS chunks")
    # Recreate WITHOUT context columns (old schema)
    test_db.execute("""
        CREATE TABLE chunks (
            id              TEXT PRIMARY KEY,
            document_id     TEXT NOT NULL,
            chunk_index     INTEGER NOT NULL,
            text            TEXT NOT NULL,
            title           TEXT NOT NULL DEFAULT '',
            section_path    TEXT NOT NULL DEFAULT '',
            page_start      INTEGER NOT NULL DEFAULT -1,
            page_end        INTEGER NOT NULL DEFAULT -1,
            chunk_strategy  TEXT NOT NULL DEFAULT 'unknown',
            metadata_json   TEXT NOT NULL DEFAULT '{}',
            content_hash    TEXT NOT NULL DEFAULT '',
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (document_id) REFERENCES source_documents(id)
        )
    """)
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")
    old_text = "original chunk text"

    # Insert directly via SQL (bypass chunk_repository which expects new columns)
    import uuid, hashlib
    chunk_id = str(uuid.uuid4())
    content_hash = hashlib.sha256(old_text.encode("utf-8")).hexdigest()
    test_db.execute(
        """INSERT INTO chunks
           (id, document_id, chunk_index, text, title, section_path,
            page_start, page_end, chunk_strategy, content_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (chunk_id, doc["id"], 0, old_text, "Ch 1", "Ch 1", 1, 1, "toc", content_hash),
    )
    test_db.commit()

    # Verify old schema: no context columns
    cols_before = {r[1] for r in test_db.execute("PRAGMA table_info('chunks')").fetchall()}
    assert "context_prefix" not in cols_before
    assert "context_mode" not in cols_before

    # Run migration
    from app.db.schema import _ensure_chunk_context_columns
    _ensure_chunk_context_columns(test_db)
    test_db.commit()

    # Verify new columns exist
    cols_after = {r[1] for r in test_db.execute("PRAGMA table_info('chunks')").fetchall()}
    assert "context_prefix" in cols_after
    assert "context_mode" in cols_after

    # Existing chunk text is unchanged
    rows = test_db.execute("SELECT text, context_prefix, context_mode FROM chunks").fetchall()
    assert rows[0]["text"] == old_text
    assert rows[0]["context_prefix"] == ""
    assert rows[0]["context_mode"] == "metadata"


def test_content_hash_uses_canonical_text(test_db):
    """content_hash still hashes canonical chunks.text, not indexed text."""
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")

    import hashlib
    canonical = "canonical chunk content"
    expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    insert_chunks(doc["id"], [{
        "chunk_index": 0,
        "text": canonical,
        "title": "Ch 1",
        "section_path": "Ch 1",
        "page_start": 1,
        "page_end": 1,
        "chunk_strategy": "toc",
        "context_prefix": "[Book > Ch 1 > Subsection]",
        "context_mode": "metadata",
    }])

    row = test_db.execute(
        "SELECT content_hash, text, context_prefix FROM chunks WHERE document_id = ?",
        (doc["id"],),
    ).fetchone()
    assert row["content_hash"] == expected_hash
    assert row["text"] == canonical


def test_old_caller_insert_without_context_keys(test_db):
    """Old callers still insert successfully when context keys are omitted."""
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")

    # Simulate an old caller that doesn't know about context keys
    count = insert_chunks(doc["id"], [{
        "chunk_index": 0,
        "text": "hello world",
        "title": "Intro",
        "section_path": "Intro",
        "page_start": 1,
        "page_end": 1,
        "chunk_strategy": "fixed",
    }])
    assert count == 1
    rows = get_chunks_by_document(doc["id"])
    assert rows[0]["text"] == "hello world"


# ---------------------------------------------------------------------------
# M5 - Keyword Index Writer Adapter tests
# ---------------------------------------------------------------------------


def test_fts_matches_context_prefix_token(test_db):
    """FTS can match a token present only in chunks.context_prefix."""
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")

    insert_chunks(doc["id"], [{
        "chunk_index": 0,
        "text": "ordinary text content",
        "title": "Chapter",
        "section_path": "Chapter > Section",
        "page_start": 1,
        "page_end": 1,
        "chunk_strategy": "toc",
        "context_prefix": "[My Textbook > Chapter > UNIQUEPREFIXTOKEN]",
        "context_mode": "metadata",
    }])

    # FTS should find the chunk by the prefix-only token
    results = search_chunks_fts(nb["id"], "UNIQUEPREFIXTOKEN", top_k=5)
    assert len(results) >= 1
    # Result text must be canonical, not prefixed
    assert results[0]["text"] == "ordinary text content"
    assert not results[0]["text"].startswith("[")
    # Ensure the indexed text is NOT leaked
    assert "UNIQUEPREFIXTOKEN" not in results[0]["text"]


def test_fts_result_text_is_canonical(test_db):
    """FTS search result text comes from chunks.text, never from chunks_fts.text."""
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")

    insert_chunks(doc["id"], [{
        "chunk_index": 0,
        "text": "canonical chunk body",
        "title": "Ch 1",
        "section_path": "Ch 1 > Sect",
        "page_start": 1,
        "page_end": 1,
        "chunk_strategy": "toc",
        "context_prefix": "[Book > Ch 1 > Sect]",
        "context_mode": "metadata",
    }])

    results = search_chunks_fts(nb["id"], "canonical", top_k=5)
    assert len(results) >= 1
    assert results[0]["text"] == "canonical chunk body"
    assert not results[0]["text"].startswith("[")


def test_fts_update_trigger_reflects_prefix_change(test_db):
    """FTS update triggers remove old prefix terms and index new ones."""
    nb = create_notebook("Test")
    doc = create_source_document(nb["id"], "test.pdf")

    insert_chunks(doc["id"], [{
        "chunk_index": 0,
        "text": "chunk content",
        "title": "Ch",
        "section_path": "Ch",
        "page_start": 1,
        "page_end": 1,
        "chunk_strategy": "toc",
        "context_prefix": "[Book > OLDTERMXYZ]",
        "context_mode": "metadata",
    }])

    # Old term is findable
    assert len(search_chunks_fts(nb["id"], "OLDTERMXYZ", top_k=5)) >= 1

    # Update context_prefix
    row = test_db.execute(
        "SELECT id FROM chunks WHERE document_id = ?", (doc["id"],)
    ).fetchone()
    test_db.execute(
        "UPDATE chunks SET context_prefix = ? WHERE id = ?",
        ("[Book > NEWTERMABC]", row["id"]),
    )
    test_db.commit()

    # Old term should no longer be found
    assert len(search_chunks_fts(nb["id"], "OLDTERMXYZ", top_k=5)) == 0
    # New term should be found
    assert len(search_chunks_fts(nb["id"], "NEWTERMABC", top_k=5)) >= 1
    # Text is still canonical
    results = search_chunks_fts(nb["id"], "NEWTERMABC", top_k=5)
    assert results[0]["text"] == "chunk content"
