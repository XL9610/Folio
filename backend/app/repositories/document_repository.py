"""Source document repository — track uploaded PDFs and their ingestion status."""

from __future__ import annotations

import uuid
from typing import Any

from ..db.session import get_connection


def create_source_document(
    notebook_id: str,
    filename: str,
    page_count: int = 0,
    content_hash: str = "",
    display_title: str = "",
    chunking_mode: str = "",
    structure_status: str = "unknown_legacy",
    embedded_toc_present: bool | None = None,
    synthetic_outline_mode: str | None = None,
    structure_confidence: str | None = None,
    structure_user_note: str | None = None,
    structure_llm_instruction: str | None = None,
) -> dict[str, Any]:
    """Insert a new source document row and return it."""
    conn = get_connection()
    doc_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO source_documents
           (id, notebook_id, filename, display_title, page_count, content_hash,
            chunking_mode, structure_status, embedded_toc_present,
            synthetic_outline_mode, structure_confidence, structure_user_note,
            structure_llm_instruction)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            doc_id,
            notebook_id,
            filename,
            display_title,
            page_count,
            content_hash,
            chunking_mode,
            structure_status,
            None if embedded_toc_present is None else int(bool(embedded_toc_present)),
            synthetic_outline_mode,
            structure_confidence,
            structure_user_note,
            structure_llm_instruction,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM source_documents WHERE id = ?", (doc_id,)
    ).fetchone()
    return dict(row)


def update_document_status(
    document_id: str,
    status: str,
    chunk_count: int | None = None,
) -> None:
    """Update the ingestion status and optionally the chunk count."""
    conn = get_connection()
    if chunk_count is not None:
        conn.execute(
            "UPDATE source_documents SET status = ?, chunk_count = ? WHERE id = ?",
            (status, chunk_count, document_id),
        )
    else:
        conn.execute(
            "UPDATE source_documents SET status = ? WHERE id = ?",
            (status, document_id),
        )
    conn.commit()


def get_document(document_id: str) -> dict[str, Any] | None:
    """Get a source document by ID, or None."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM source_documents WHERE id = ?", (document_id,)
    ).fetchone()
    return dict(row) if row else None


def list_documents(notebook_id: str) -> list[dict[str, Any]]:
    """List all source documents in a notebook."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM source_documents WHERE notebook_id = ? ORDER BY created_at DESC",
        (notebook_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_document_by_hash(notebook_id: str, content_hash: str) -> dict[str, Any] | None:
    """Find an existing document by notebook + content hash. Returns None if not found."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM source_documents WHERE notebook_id = ? AND content_hash = ?",
        (notebook_id, content_hash),
    ).fetchone()
    return dict(row) if row else None


def delete_document(document_id: str) -> bool:
    """Delete a document and all child records. Returns True if deleted."""
    conn = get_connection()
    # Delete child records first to satisfy foreign key constraints
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    conn.execute("DELETE FROM ingestion_jobs WHERE document_id = ?", (document_id,))
    conn.execute("DELETE FROM document_extractions WHERE document_id = ?", (document_id,))
    conn.execute("DELETE FROM source_documents WHERE id = ?", (document_id,))
    conn.commit()
    return conn.total_changes > 0
