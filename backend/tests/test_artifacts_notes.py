"""Tests for saved note artifacts."""

from __future__ import annotations

import sqlite3

from app.db.schema import create_tables
from app.repositories.artifact_repository import create_artifact, delete_artifact, get_artifact, list_artifacts
from app.repositories.notebook_repository import create_notebook


def test_saved_note_artifact_round_trip():
    from app.db import session as db_session

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    original_conn = getattr(db_session._local, "conn", None)
    db_session._local.conn = conn
    create_tables(conn)
    try:
        nb = create_notebook("Notes")
        payload = {
            "messages": [
                {"role": "user", "text": "What is X?"},
                {"role": "assistant", "text": "X is a concept.", "citations": []},
            ],
            "quiz_context": {
                "question": "What is X?",
                "choices": ["A. X", "B. Y"],
                "selected_answer": "B: Y",
                "correct_answer": "A: X",
                "was_correct": False,
                "explanation": "X is supported by the source.",
            },
        }

        created = create_artifact(
            notebook_id=nb["id"],
            kind="note",
            title="What is X?",
            content=payload,
            source_document_ids=["d1"],
            params={"origin": "chat_save"},
        )

        listed = list_artifacts(nb["id"])
        fetched = get_artifact(created["id"])
        assert listed[0]["kind"] == "note"
        assert fetched["content_json"]["messages"][0]["text"] == "What is X?"
        assert fetched["content_json"]["quiz_context"]["question"] == "What is X?"
        assert fetched["source_document_ids"] == ["d1"]

        assert delete_artifact(created["id"]) is True
        assert get_artifact(created["id"]) is None
    finally:
        db_session._local.conn = original_conn
        conn.close()
