"""Extraction artifact repository — SQLite metadata for persisted extraction checkpoints.

The heavy body text lives on disk under local-data/extraction_artifacts/.
This table stores only paths and JSON metadata so rows stay small.
"""

from __future__ import annotations

from typing import Any

from ..db.session import get_connection


def upsert_document_extraction(
    document_id: str,
    extraction_engine: str,
    full_text_path: str,
    markdown_path: str | None,
    pages_path: str | None,
    metadata_json: str,
    warnings_json: str,
) -> dict[str, Any]:
    """Insert or update a document_extractions row. Returns the row as a dict."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO document_extractions
           (document_id, extraction_engine, full_text_path, markdown_path, pages_path,
            metadata_json, warnings_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(document_id) DO UPDATE SET
               extraction_engine = excluded.extraction_engine,
               full_text_path    = excluded.full_text_path,
               markdown_path     = excluded.markdown_path,
               pages_path        = excluded.pages_path,
               metadata_json     = excluded.metadata_json,
               warnings_json     = excluded.warnings_json""",
        (
            document_id,
            extraction_engine,
            full_text_path,
            markdown_path,
            pages_path,
            metadata_json,
            warnings_json,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM document_extractions WHERE document_id = ?", (document_id,)
    ).fetchone()
    return dict(row)


def get_document_extraction(document_id: str) -> dict[str, Any] | None:
    """Return the document_extractions row for a document, or None."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM document_extractions WHERE document_id = ?", (document_id,)
    ).fetchone()
    return dict(row) if row else None


def delete_document_extraction(document_id: str) -> bool:
    """Delete a document_extractions row. Returns True if a row was deleted."""
    conn = get_connection()
    cursor = conn.execute(
        "DELETE FROM document_extractions WHERE document_id = ?", (document_id,)
    )
    conn.commit()
    return cursor.rowcount > 0
