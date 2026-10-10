"""Saved artifacts repository — CRUD for generated study materials."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..db.session import get_connection


def create_artifact(
    notebook_id: str,
    kind: str,
    title: str,
    content: dict[str, Any],
    source_document_ids: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert a new saved artifact and return it."""
    conn = get_connection()
    artifact_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO saved_artifacts
           (id, notebook_id, kind, title, content_json, source_document_ids, params_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            artifact_id,
            notebook_id,
            kind,
            title,
            json.dumps(content, ensure_ascii=False),
            json.dumps(source_document_ids or []),
            json.dumps(params or {}),
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM saved_artifacts WHERE id = ?", (artifact_id,)
    ).fetchone()
    return _row_to_dict(row)


def list_artifacts(notebook_id: str, kind: str | None = None) -> list[dict[str, Any]]:
    """List all saved artifacts in a notebook, optionally filtered by kind."""
    conn = get_connection()
    if kind:
        rows = conn.execute(
            """SELECT * FROM saved_artifacts
               WHERE notebook_id = ? AND kind = ?
               ORDER BY created_at DESC""",
            (notebook_id, kind),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM saved_artifacts
               WHERE notebook_id = ?
               ORDER BY created_at DESC""",
            (notebook_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_artifact(artifact_id: str) -> dict[str, Any] | None:
    """Get a single artifact by ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM saved_artifacts WHERE id = ?", (artifact_id,)
    ).fetchone()
    return _row_to_dict(row) if row else None


def delete_artifact(artifact_id: str) -> bool:
    """Delete an artifact by ID. Returns True if deleted."""
    conn = get_connection()
    conn.execute("DELETE FROM review_items WHERE artifact_id = ?", (artifact_id,))
    cur = conn.execute("DELETE FROM saved_artifacts WHERE id = ?", (artifact_id,))
    conn.commit()
    return cur.rowcount > 0


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a sqlite3.Row to a dict with parsed JSON fields."""
    d = dict(row)
    # Parse JSON fields for ergonomic consumption
    for field in ("content_json", "source_document_ids", "params_json"):
        raw = d.get(field, "{}")
        if isinstance(raw, str):
            try:
                d[field] = json.loads(raw)
            except json.JSONDecodeError:
                pass
    return d
