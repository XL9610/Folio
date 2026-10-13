"""Summary-node repository — derived hierarchical overview cards.

Roadmap Phase 3: stores the two-level summary hierarchy (section cards and
one document card) built above the canonical chunks. Generated summaries are
navigation/synthesis aids, never citations; every provenance ID must resolve
to a raw chunk in the same document.

All reads enforce both notebook and document scope.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..db.session import get_connection

DOCUMENT_SECTION_KEY = "__document__"

_INSERT_SQL = """
    INSERT INTO document_summary_nodes
        (id, notebook_id, document_id, level, section_key, ordinal,
         summary_json, source_chunk_ids_json, child_node_ids_json,
         build_signature, prompt_version, model_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def replace_document_nodes(
    notebook_id: str,
    document_id: str,
    nodes: list[dict[str, Any]],
) -> int:
    """Atomically replace all summary nodes for one document.

    Deletes the previous build and inserts ``nodes`` in a single
    transaction: readers never observe half of a rebuilt hierarchy, and a
    failed insert leaves the previous build intact (rollback).

    Each node dict needs: level ("section"|"document"), section_key,
    ordinal, summary_json (str), source_chunk_ids (list[str]),
    child_node_ids (list[str]), build_signature, prompt_version, model_id.
    An optional "id" is stored as-is (the builder assigns node ids up front
    so document-card provenance can reference them); otherwise one is
    generated. Returns the number of nodes inserted.
    """
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM document_summary_nodes WHERE document_id = ?",
            (document_id,),
        )
        for node in nodes:
            conn.execute(
                _INSERT_SQL,
                (
                    node.get("id") or str(uuid.uuid4()),
                    notebook_id,
                    document_id,
                    node["level"],
                    node["section_key"],
                    node["ordinal"],
                    node["summary_json"],
                    json.dumps(node["source_chunk_ids"], ensure_ascii=False),
                    json.dumps(node["child_node_ids"], ensure_ascii=False),
                    node["build_signature"],
                    node["prompt_version"],
                    node["model_id"],
                ),
            )
        conn.commit()
        return len(nodes)
    except Exception:
        conn.rollback()
        raise


def get_document_nodes(
    notebook_id: str,
    document_id: str,
) -> list[dict[str, Any]]:
    """Read all summary nodes for a document, scoped to its notebook.

    Returns rows ordered document card first, then section cards by ordinal.
    """
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT * FROM document_summary_nodes
        WHERE notebook_id = ? AND document_id = ?
        ORDER BY CASE level WHEN 'document' THEN 0 ELSE 1 END, ordinal
        """,
        (notebook_id, document_id),
    ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        d["source_chunk_ids"] = json.loads(d.pop("source_chunk_ids_json") or "[]")
        d["child_node_ids"] = json.loads(d.pop("child_node_ids_json") or "[]")
        result.append(d)
    return result


def get_build_signature(notebook_id: str, document_id: str) -> str | None:
    """Return the stored build signature, or None when no build exists."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT build_signature FROM document_summary_nodes
        WHERE notebook_id = ? AND document_id = ?
        LIMIT 1
        """,
        (notebook_id, document_id),
    ).fetchone()
    return row[0] if row else None


def count_document_nodes(notebook_id: str, document_id: str) -> dict[str, int]:
    """Count nodes by level for one document (scoped read)."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT level, COUNT(*) AS n FROM document_summary_nodes
        WHERE notebook_id = ? AND document_id = ?
        GROUP BY level
        """,
        (notebook_id, document_id),
    ).fetchall()
    return {r[0]: r[1] for r in rows}
