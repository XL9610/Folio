"""Ingestion job repository — track async ingestion progress."""

from __future__ import annotations

import uuid
from typing import Any

from ..db.session import get_connection


def create_job(
    notebook_id: str,
    document_id: str = "",
    status: str = "queued",
    stage: str = "queued",
) -> str:
    """Create a new ingestion job and return its id."""
    job_id = str(uuid.uuid4())
    conn = get_connection()
    conn.execute(
        """INSERT INTO ingestion_jobs (id, notebook_id, document_id, status, stage)
           VALUES (?, ?, ?, ?, ?)""",
        (job_id, notebook_id, document_id, status, stage),
    )
    # Commit immediately: the async worker thread reads/writes this row on a
    # separate connection, and an uncommitted INSERT would hold the write lock
    # and deadlock it with "database is locked".
    conn.commit()
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    """Update fields on a job (status, stage, progress_pct, document_id, error, warnings)."""
    allowed = {"status", "stage", "progress_pct", "document_id", "error_message", "warnings_json"}
    updates: list[str] = []
    values: list[Any] = []
    for key, value in fields.items():
        if key in allowed:
            updates.append(f"{key} = ?")
            values.append(value)
    if not updates:
        return
    updates.append("updated_at = datetime('now')")
    values.append(job_id)
    conn = get_connection()
    conn.execute(f"UPDATE ingestion_jobs SET {', '.join(updates)} WHERE id = ?", values)
    # Commit immediately so the polling endpoint sees the change without waiting
    # for an implicit transaction to be closed by a later SELECT/DDL.
    conn.commit()


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return a job by id, or None."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(notebook_id: str) -> list[dict[str, Any]]:
    """Return jobs for a notebook, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM ingestion_jobs WHERE notebook_id = ? ORDER BY created_at DESC",
        (notebook_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def mark_interrupted_jobs_failed() -> int:
    """Mark jobs and documents left non-terminal by a previous run as failed.

    Called on startup: an app closed mid-ingestion leaves a queued/running job
    (and possibly a pending/indexing source_document) behind. Failing them here
    lets the existing failed-retry path (content-hash dedup) recover cleanly.
    """
    conn = get_connection()
    cur = conn.execute(
        """UPDATE ingestion_jobs
           SET status = 'failed',
               error_message = 'Interrupted by app restart',
               updated_at = datetime('now')
           WHERE status IN ('queued', 'running')"""
    )
    conn.execute(
        """UPDATE source_documents
           SET status = 'failed'
           WHERE status IN ('pending', 'indexing')"""
    )
    conn.commit()
    return cur.rowcount
