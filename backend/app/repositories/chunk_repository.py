"""Chunk repository — insert and search text chunks."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from ..db.session import get_connection


def insert_chunks(document_id: str, chunks: list[dict[str, Any]]) -> int:
    """Insert all chunks for a document. Returns count inserted.

    Each chunk dict should have:
      - chunk_index
      - text
      - title
      - section_path
      - page_start
      - page_end
      - chunk_strategy
      - metadata_json (optional)
      - context_prefix (optional)
      - context_mode (optional)
    """
    conn = get_connection()
    count = 0
    try:
        for c in chunks:
            chunk_id = str(uuid.uuid4())
            content_hash = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()
            conn.execute(
                """INSERT INTO chunks
                   (id, document_id, chunk_index, text, title, section_path,
                    page_start, page_end, chunk_strategy, metadata_json, content_hash,
                    context_prefix, context_mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk_id,
                    document_id,
                    c["chunk_index"],
                    c["text"],
                    c.get("title", ""),
                    c.get("section_path", ""),
                    c.get("page_start", -1),
                    c.get("page_end", -1),
                    c.get("chunk_strategy", "unknown"),
                    c.get("metadata_json", "{}"),
                    content_hash,
                    c.get("context_prefix", ""),
                    c.get("context_mode", "metadata"),
                ),
            )
            count += 1
        conn.commit()
        return count
    except Exception:
        conn.rollback()
        raise


def delete_chunks_by_document(document_id: str) -> None:
    """Delete all chunks for a document (clears partial inserts on failed ingestion)."""
    conn = get_connection()
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    conn.commit()


def search_chunks_fts(
    notebook_id: str,
    query: str,
    top_k: int = 10,
    document_ids: list[str] | None = None,
    chunk_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Full-text search using SQLite FTS5, filtered by notebook.

    Optionally restrict to a set of document_ids and/or an explicit set of
    chunk_ids (Phase 4 candidate-scoped retrieval). The MATCH still ranks
    against the corpus index, but only the named chunks are returned, so a
    candidate chunk is never truncated away by a global top-K.

    Returns list of matching chunks with rank info.
    """
    conn = get_connection()

    conds = ["d.notebook_id = ?", "chunks_fts MATCH ?"]
    params: list[Any] = [notebook_id, query]
    if document_ids and len(document_ids) > 0:
        conds.append(f"c.document_id IN ({', '.join('?' * len(document_ids))})")
        params.extend(document_ids)
    if chunk_ids is not None:
        if not chunk_ids:
            return []
        conds.append(f"c.id IN ({', '.join('?' * len(chunk_ids))})")
        params.extend(chunk_ids)
    params.append(top_k)

    sql = f"""
        SELECT
            c.id,
            c.document_id,
            c.chunk_index,
            c.text,
            c.title,
            c.section_path,
            c.page_start,
            c.page_end,
            c.chunk_strategy,
            rank
        FROM chunks c
        JOIN chunks_fts fts ON c.rowid = fts.rowid
        JOIN source_documents d ON c.document_id = d.id
        WHERE {" AND ".join(conds)}
        ORDER BY rank
        LIMIT ?
        """
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_chunk_by_id(chunk_id: str) -> dict[str, Any] | None:
    """Get a single chunk by ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
    ).fetchone()
    return dict(row) if row else None


def get_chunk_identity_by_document(document_id: str) -> list[dict[str, Any]]:
    """Ordered (id, content_hash) pairs for one document — no chunk text.

    Lightweight input for cache-signature computation and provenance
    validation on warm query paths.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, content_hash FROM chunks "
        "WHERE document_id = ? ORDER BY chunk_index",
        (document_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_chunks_by_document(document_id: str) -> list[dict[str, Any]]:
    """List all chunks for a document, ordered by chunk_index."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
        (document_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_document_sections(document_id: str) -> list[dict[str, Any]]:
    """Get distinct sections for a document — used for overview/table-of-contents.

    Returns list of { title, section_path, page_start, page_end, chunk_count }.
    """
    conn = get_connection()
    rows = conn.execute(
        """SELECT
             title,
             section_path,
             MIN(page_start) AS page_start,
             MAX(page_end)   AS page_end,
             COUNT(*)        AS chunk_count
         FROM chunks
         WHERE document_id = ?
         GROUP BY section_path
         ORDER BY MIN(page_start)""",
        (document_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_chunks_by_section_prefix(
    document_id: str,
    section_prefix: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Get chunks whose section_path starts with or equals the prefix.

    Matches BOTH exact section_path AND children (section_path LIKE prefix > %).
    Returns chunks ordered by page_start ASC, chunk_index ASC.

    Examples:
        "Chapter 1" matches "Chapter 1" and "Chapter 1 > Section 1.1"
        "Chapter 1: Java Primer" matches only that exact section + its children.
    """
    conn = get_connection()
    exact = section_prefix
    prefix_like = section_prefix + " > %"
    rows = conn.execute(
        """SELECT *
           FROM chunks
           WHERE document_id = ?
             AND (section_path = ? OR section_path LIKE ?)
           ORDER BY page_start ASC, chunk_index ASC
           LIMIT ?""",
        (document_id, exact, prefix_like, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_chunks_for_section_paths(
    document_id: str,
    section_paths: list[str],
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Get chunks for a list of exact section_paths.

    Returns chunks ordered by page_start ASC, chunk_index ASC.
    Useful when multiple sections need to be retrieved for side-by-side comparison.
    """
    conn = get_connection()
    placeholders = ",".join("?" * len(section_paths))
    rows = conn.execute(
        f"""SELECT *
            FROM chunks
            WHERE document_id = ?
              AND section_path IN ({placeholders})
            ORDER BY page_start ASC, chunk_index ASC
            LIMIT ?""",
        (document_id, *section_paths, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_chunks_by_page_range(
    document_id: str,
    page_start: int,
    page_end: int,
) -> list[dict[str, Any]]:
    """Get chunks whose page range overlaps with [page_start, page_end].

    Returns chunks ordered by page_start ASC, chunk_index ASC.
    """
    conn = get_connection()
    rows = conn.execute(
        """SELECT *
           FROM chunks
           WHERE document_id = ?
             AND page_start <= ?
             AND page_end >= ?
           ORDER BY page_start ASC, chunk_index ASC""",
        (document_id, page_end, page_start),
    ).fetchall()
    return [dict(r) for r in rows]
