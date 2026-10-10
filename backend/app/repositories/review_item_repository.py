"""Review item repository - saved flashcards and quiz questions for review."""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from ..db.session import get_connection

ReviewFilter = Literal["all", "mistakes", "starred", "flashcards", "quiz"]


def upsert_review_item(
    notebook_id: str,
    artifact_id: str,
    item_type: str,
    item_index: int,
    prompt: str,
    answer: str = "",
    choices: list[str] | None = None,
    explanation: str = "",
    source_document_id: str = "",
    source_chunk_id: str = "",
    section_path: str = "",
    page_start: int = -1,
    page_end: int = -1,
    is_starred: bool = False,
    is_mistake: bool = False,
    last_user_answer: str = "",
) -> dict[str, Any]:
    """Create or update a review item.

    Repeated saves preserve any existing starred/mistake flags so a wrong answer
    can later be starred without losing its mistake status, and vice versa.
    """
    conn = get_connection()
    item_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO review_items (
            id, notebook_id, artifact_id, item_type, item_index, prompt, answer,
            choices_json, explanation, source_document_id, source_chunk_id,
            section_path, page_start, page_end, is_starred, is_mistake,
            last_user_answer
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(notebook_id, artifact_id, item_type, item_index) DO UPDATE SET
            prompt = excluded.prompt,
            answer = excluded.answer,
            choices_json = excluded.choices_json,
            explanation = excluded.explanation,
            source_document_id = excluded.source_document_id,
            source_chunk_id = excluded.source_chunk_id,
            section_path = excluded.section_path,
            page_start = excluded.page_start,
            page_end = excluded.page_end,
            is_starred = MAX(review_items.is_starred, excluded.is_starred),
            is_mistake = MAX(review_items.is_mistake, excluded.is_mistake),
            last_user_answer = CASE
                WHEN excluded.last_user_answer != '' THEN excluded.last_user_answer
                ELSE review_items.last_user_answer
            END,
            updated_at = datetime('now')
        """,
        (
            item_id,
            notebook_id,
            artifact_id,
            item_type,
            item_index,
            prompt,
            answer,
            json.dumps(choices or [], ensure_ascii=False),
            explanation,
            source_document_id,
            source_chunk_id,
            section_path,
            page_start,
            page_end,
            1 if is_starred else 0,
            1 if is_mistake else 0,
            last_user_answer,
        ),
    )
    conn.commit()
    row = conn.execute(
        """
        SELECT * FROM review_items
        WHERE notebook_id = ? AND artifact_id = ? AND item_type = ? AND item_index = ?
        """,
        (notebook_id, artifact_id, item_type, item_index),
    ).fetchone()
    return _row_to_dict(row)


def list_review_items(notebook_id: str, filter_by: ReviewFilter = "all") -> list[dict[str, Any]]:
    """List review items for a notebook, optionally filtered for the UI tabs."""
    conn = get_connection()
    where = ["notebook_id = ?"]
    params: list[Any] = [notebook_id]
    if filter_by == "mistakes":
        where.append("is_mistake = 1")
    elif filter_by == "starred":
        where.append("is_starred = 1")
    elif filter_by == "flashcards":
        where.append("item_type = 'flashcard'")
    elif filter_by == "quiz":
        where.append("item_type = 'quiz_question'")
    rows = conn.execute(
        f"""
        SELECT * FROM review_items
        WHERE {" AND ".join(where)}
        ORDER BY updated_at DESC, created_at DESC
        """,
        params,
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_review_item(item_id: str) -> bool:
    conn = get_connection()
    cur = conn.execute("DELETE FROM review_items WHERE id = ?", (item_id,))
    conn.commit()
    return cur.rowcount > 0


def _row_to_dict(row: Any) -> dict[str, Any]:
    d = dict(row)
    raw_choices = d.get("choices_json", "[]")
    if isinstance(raw_choices, str):
        try:
            d["choices"] = json.loads(raw_choices)
        except json.JSONDecodeError:
            d["choices"] = []
    else:
        d["choices"] = raw_choices or []
    d["is_starred"] = bool(d.get("is_starred"))
    d["is_mistake"] = bool(d.get("is_mistake"))
    return d
