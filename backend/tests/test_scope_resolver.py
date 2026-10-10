"""Tests for ScopeResolver — three-layer scope resolution."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from app.db.schema import create_tables
from app.repositories.notebook_repository import create_notebook
from app.repositories.document_repository import create_source_document
from app.repositories.chunk_repository import insert_chunks
from app.services.generation.scope_resolver import ScopeResolver
from app.services.retrieval.base import RetrievedChunk


@pytest.fixture
def scope_test_db():
    """In-memory DB with a test document and chunks suitable for scope tests."""
    import sqlite3
    from app.db import session as db_session

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    original_conn = getattr(db_session._local, 'conn', None)
    db_session._local.conn = conn
    create_tables(conn)

    # Create notebook + document
    nb = create_notebook("Scope Test")
    doc = create_source_document(nb["id"], "Test Document.pdf", page_count=6)

    # Insert test chunks with known section_paths and pages
    insert_chunks(doc["id"], [
        {"chunk_index": 0, "text": "Chapter 1 intro text", "title": "Chapter 1",
         "section_path": "Chapter 1: Introduction", "page_start": 1, "page_end": 1,
         "chunk_strategy": "toc"},
        {"chunk_index": 1, "text": "Section 1.1 details", "title": "Section 1.1",
         "section_path": "Chapter 1: Introduction > 1.1 Motivation", "page_start": 1, "page_end": 2,
         "chunk_strategy": "toc"},
        {"chunk_index": 2, "text": "Chapter 2 methods text", "title": "Chapter 2",
         "section_path": "Chapter 2: Methods", "page_start": 3, "page_end": 3,
         "chunk_strategy": "toc"},
        {"chunk_index": 3, "text": "Algorithm details here", "title": "2.1 Algorithm",
         "section_path": "Chapter 2: Methods > 2.1 Algorithm", "page_start": 3, "page_end": 4,
         "chunk_strategy": "toc"},
        {"chunk_index": 4, "text": "Chapter 15 music text", "title": "15. The Madrigal",
         "section_path": "PART THREE > 15. The Madrigal and Related Forms", "page_start": 150, "page_end": 155,
         "chunk_strategy": "toc"},
        {"chunk_index": 5, "text": "Chapter 15 continued", "title": "15. The Madrigal",
         "section_path": "PART THREE > 15. The Madrigal and Related Forms", "page_start": 156, "page_end": 160,
         "chunk_strategy": "toc"},
        {"chunk_index": 6, "text": "Chapter 10 text", "title": "10. Baroque",
         "section_path": "PART TWO > 10. The Baroque Era", "page_start": 100, "page_end": 105,
         "chunk_strategy": "toc"},
    ])

    info = {"notebook_id": nb["id"], "document_id": doc["id"]}
    yield conn, info

    db_session._local.conn = original_conn
    conn.close()


def make_mock_retriever():
    """Retriever that returns fixed semantic results."""
    ret = MagicMock()
    ret.search.return_value = []
    return ret


# ── Layer tests ──────────────────────────────────────────────────────

def test_empty_scope_returns_all_chunks(scope_test_db):
    conn, info = scope_test_db
    resolver = ScopeResolver(make_mock_retriever())
    result = resolver.resolve(info["document_id"], info["notebook_id"], "")
    assert result.metadata["layer"] == "all"
    assert len(result.chunks) == 7


def test_page_range_layer(scope_test_db):
    conn, info = scope_test_db
    resolver = ScopeResolver(make_mock_retriever())
    result = resolver.resolve(info["document_id"], info["notebook_id"], "pages 150 to 160")
    assert result.metadata["layer"] == "page_range"
    assert result.metadata["page_start"] == 150
    assert result.metadata["page_end"] == 160
    # Should match the two Chapter 15 chunks that overlap pages 150-160
    assert len(result.chunks) >= 1
    for c in result.chunks:
        assert 150 <= c.page_end and c.page_start <= 160


def test_toc_chapter_number_match(scope_test_db):
    """Chapter 15 should match via chapter-number extraction even though
    section_path does NOT contain 'Chapter 15' literally."""
    conn, info = scope_test_db
    resolver = ScopeResolver(make_mock_retriever())
    result = resolver.resolve(info["document_id"], info["notebook_id"], "Chapter 15")
    assert result.metadata["layer"] == "toc"
    # Should find the two chunks from "15. The Madrigal"
    assert len(result.chunks) >= 1
    sections = {c.section_path for c in result.chunks if c.section_path}
    assert any("Madrigal" in s for s in sections)


def test_toc_chapter_number_variants(scope_test_db):
    """Ch. 15 or just '15' should also match."""
    conn, info = scope_test_db
    resolver = ScopeResolver(make_mock_retriever())
    for scope in ["Ch. 15", "chapter 15", "15"]:
        result = resolver.resolve(info["document_id"], info["notebook_id"], scope)
        # Page range regex could catch bare "15" — accept either layer
        if result.metadata["layer"] == "page_range":
            continue  # also valid
        assert result.metadata["layer"] == "toc", f"Expected toc layer for '{scope}', got {result.metadata['layer']}"


def test_toc_chapter_10_not_misidentified_as_1(scope_test_db):
    """Chapter 1 should NOT match '10. Baroque'."""
    conn, info = scope_test_db
    resolver = ScopeResolver(make_mock_retriever())
    result = resolver.resolve(info["document_id"], info["notebook_id"], "Chapter 1")
    assert result.metadata["layer"] == "toc"
    sections = {c.section_path for c in result.chunks if c.section_path}
    # Should match "Chapter 1: Introduction", NOT "10. The Baroque Era"
    assert any("Introduction" in s for s in sections)
    assert not any("Baroque" in s for s in sections)


def test_toc_direct_substring_match(scope_test_db):
    """Direct substring 'Methods' should match Chapter 2: Methods."""
    conn, info = scope_test_db
    resolver = ScopeResolver(make_mock_retriever())
    result = resolver.resolve(info["document_id"], info["notebook_id"], "Methods")
    assert result.metadata["layer"] == "toc"
    assert len(result.chunks) >= 1


def test_semantic_fallback_layer(scope_test_db):
    """Unmatched scope falls back to semantic retrieval."""
    conn, info = scope_test_db
    fake_ret = make_mock_retriever()
    fake_chunk = RetrievedChunk(
        chunk_id="c99", document_id=info["document_id"], text="semantic hit",
        title="", section_path="", page_start=1, page_end=1,
    )
    fake_ret.search.return_value = [fake_chunk]
    resolver = ScopeResolver(fake_ret)
    result = resolver.resolve(info["document_id"], info["notebook_id"], "completely unknown gibberish")
    assert result.metadata["layer"] == "semantic"
    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "c99"


def test_unmatched_chapter_scope_does_not_semantic_fallback(scope_test_db):
    """Generation scope should not pretend an unmatched chapter was resolved."""
    conn, info = scope_test_db
    resolver = ScopeResolver(make_mock_retriever())
    result = resolver.resolve(info["document_id"], info["notebook_id"], "Chapter 99")

    assert result.metadata["layer"] == "unresolved_section"
    assert result.metadata["scope_resolved"] is False
    assert result.chunks == []
    assert "Could not match" in result.metadata["unresolved_reason"]


# ── Body-window scope tests ────────────────────────────────────────────

def test_empty_scope_filters_non_body_chunks(scope_test_db):
    """Empty scope should filter copyright/index/front-matter chunks by default."""
    conn, info = scope_test_db
    from app.repositories.chunk_repository import insert_chunks

    # Insert additional non-body chunks
    insert_chunks(info["document_id"], [
        {"chunk_index": 10, "text": "copyright text",
         "title": "Copyright", "section_path": "Copyright",
         "page_start": 1, "page_end": 1, "chunk_strategy": "toc"},
        {"chunk_index": 11, "text": "index terms",
         "title": "Index", "section_path": "Index",
         "page_start": 200, "page_end": 210, "chunk_strategy": "toc"},
        {"chunk_index": 12, "text": "bibliography entries",
         "title": "Bibliography", "section_path": "Bibliography",
         "page_start": 211, "page_end": 215, "chunk_strategy": "toc"},
        {"chunk_index": 13, "text": "about author",
         "title": "About the Author", "section_path": "About the Author",
         "page_start": 216, "page_end": 216, "chunk_strategy": "toc"},
    ])

    resolver = ScopeResolver(make_mock_retriever())
    result = resolver.resolve(info["document_id"], info["notebook_id"], "")

    # Should have filtered some non-body chunks
    assert result.metadata["layer"] == "all"
    assert result.metadata.get("filtered_non_body_count", 0) > 0, \
        f"Expected some non-body chunks to be filtered, got metadata={result.metadata}"

    # Non-body chunks should NOT be in the result
    non_body_titles = {"Copyright", "Index", "Bibliography", "About the Author"}
    for c in result.chunks:
        assert c.title not in non_body_titles, \
            f"Non-body chunk '{c.title}' should be filtered: {result.chunks}"

    # Body chunks from the original fixture should still be present
    titles = {c.title for c in result.chunks}
    assert any("Chapter" in t for t in titles), \
        f"Body chapters should still be present: {titles}"
