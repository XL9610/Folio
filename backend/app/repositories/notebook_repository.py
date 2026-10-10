"""Notebook repository — CRUD for notebook entities."""

from __future__ import annotations

import uuid
from typing import Any

from ..db.session import get_connection


def _row_to_dict(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    # Convert integer counts that come back as strings from the LEFT JOIN
    for col in ("source_count", "ready_count", "processing_count", "failed_count", "empty_count"):
        if col in d and d[col] is not None:
            d[col] = int(d[col])
    return d


def _derive_status(
    source_count: int,
    ready_count: int,
    processing_count: int,
    failed_count: int,
) -> str:
    """Derive a notebook-level status from its source documents."""
    if source_count == 0:
        return "empty"
    if processing_count > 0:
        return "processing"
    if failed_count == source_count and failed_count > 0:
        return "failed"
    if ready_count == source_count:
        return "ready"
    return "partial"


_NOTEBOOK_WITH_STATS_SQL = """
SELECT
    n.id,
    n.name,
    n.description,
    n.created_at,
    n.last_opened_at,
    COUNT(sd.id) AS source_count,
    COALESCE(SUM(CASE WHEN sd.status IN ('completed', 'already_ingested', 'indexed_partial', 'fts_only') THEN 1 ELSE 0 END), 0) AS ready_count,
    COALESCE(SUM(CASE WHEN sd.status IN ('pending', 'processing') THEN 1 ELSE 0 END), 0) AS processing_count,
    COALESCE(SUM(CASE WHEN sd.status = 'failed' THEN 1 ELSE 0 END), 0) AS failed_count,
    MAX(sd.created_at) AS last_updated
FROM notebooks n
LEFT JOIN source_documents sd ON n.id = sd.notebook_id
"""


def create_notebook(name: str | None = None, description: str = "") -> dict[str, Any]:
    """Create a new notebook and return it as a dict.

    If name is None, defaults to 'Default Notebook'.
    """
    conn = get_connection()
    notebook_id = str(uuid.uuid4())
    display_name = name or "Default Notebook"
    conn.execute(
        "INSERT INTO notebooks (id, name, description) VALUES (?, ?, ?)",
        (notebook_id, display_name, description),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM notebooks WHERE id = ?", (notebook_id,)).fetchone()
    d = dict(row)
    d["source_count"] = 0
    d["ready_count"] = 0
    d["processing_count"] = 0
    d["failed_count"] = 0
    d["status"] = "empty"
    d["last_updated"] = d["created_at"]
    return d


def get_or_create_default() -> dict[str, Any]:
    """Return the default notebook, creating it if it doesn't exist.

    Uses INSERT OR IGNORE so concurrent callers always converge on the same row.
    """
    conn = get_connection()
    notebook_id = str(uuid.uuid4())
    conn.execute(
        "INSERT OR IGNORE INTO notebooks (id, name) VALUES (?, ?)",
        (notebook_id, "Default Notebook"),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM notebooks WHERE name = ?", ("Default Notebook",)).fetchone()
    if row is None:
        return create_notebook("Default Notebook")
    return dict(row)


def list_notebooks() -> list[dict[str, Any]]:
    """Return all notebooks with computed stats."""
    conn = get_connection()
    rows = conn.execute(
        _NOTEBOOK_WITH_STATS_SQL
        + "GROUP BY n.id ORDER BY COALESCE(n.last_opened_at, n.created_at) DESC"
    ).fetchall()
    results = []
    for row in rows:
        d = _row_to_dict(row)
        if d:
            d["status"] = _derive_status(
                d.get("source_count", 0),
                d.get("ready_count", 0),
                d.get("processing_count", 0),
                d.get("failed_count", 0),
            )
            results.append(d)
    return results


def get_notebook(notebook_id: str) -> dict[str, Any] | None:
    """Get a notebook by ID, or None if not found."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM notebooks WHERE id = ?", (notebook_id,)).fetchone()
    return dict(row) if row else None


def get_notebook_with_stats(notebook_id: str) -> dict[str, Any] | None:
    """Get a notebook by ID with computed stats, or None if not found."""
    conn = get_connection()
    row = conn.execute(
        _NOTEBOOK_WITH_STATS_SQL + "WHERE n.id = ? GROUP BY n.id",
        (notebook_id,),
    ).fetchone()
    d = _row_to_dict(row)
    if d is None:
        return None
    d["status"] = _derive_status(
        d.get("source_count", 0),
        d.get("ready_count", 0),
        d.get("processing_count", 0),
        d.get("failed_count", 0),
    )
    return d


def touch_notebook(notebook_id: str) -> bool:
    """Record that a notebook was just opened (for recently-opened ordering)."""
    conn = get_connection()
    cur = conn.execute(
        "UPDATE notebooks SET last_opened_at = datetime('now') WHERE id = ?",
        (notebook_id,),
    )
    conn.commit()
    return cur.rowcount > 0


def update_notebook(
    notebook_id: str,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any] | None:
    """Update notebook name and/or description. Returns the updated row or None."""
    conn = get_connection()
    existing = get_notebook(notebook_id)
    if existing is None:
        return None

    new_name = name if name is not None else existing["name"]
    new_desc = description if description is not None else existing.get("description", "")

    conn.execute(
        "UPDATE notebooks SET name = ?, description = ? WHERE id = ?",
        (new_name, new_desc, notebook_id),
    )
    conn.commit()
    return get_notebook_with_stats(notebook_id)


def delete_notebook(notebook_id: str) -> bool:
    """Delete a notebook and all its child rows. Returns True on success."""
    conn = get_connection()
    existing = get_notebook(notebook_id)
    if existing is None:
        return False

    try:
        # Notebook-scoped children first. review_items references both notebooks
        # and saved_artifacts, so it must go before saved_artifacts; the
        # chunks_ad trigger cleans chunks_fts automatically when chunks go.
        conn.execute("DELETE FROM review_items WHERE notebook_id = ?", (notebook_id,))
        conn.execute("DELETE FROM saved_artifacts WHERE notebook_id = ?", (notebook_id,))
        conn.execute("DELETE FROM saved_conversations WHERE notebook_id = ?", (notebook_id,))
        conn.execute("DELETE FROM ingestion_jobs WHERE notebook_id = ?", (notebook_id,))

        # Document-scoped children, then the documents, then the notebook itself.
        for doc_id in (
            row[0]
            for row in conn.execute(
                "SELECT id FROM source_documents WHERE notebook_id = ?",
                (notebook_id,),
            ).fetchall()
        ):
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM document_extractions WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM source_documents WHERE notebook_id = ?", (notebook_id,))
        conn.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
