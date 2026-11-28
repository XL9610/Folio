"""Tests for conversation save/upsert and notebook-ownership enforcement."""

from __future__ import annotations

import sqlite3
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.db.schema import create_tables
from app.main import app


@pytest.fixture
def conv_client() -> TestClient:
    """TestClient with conversations_router's connection pinned to in-memory SQLite."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    create_tables(conn)
    conn.execute("INSERT INTO notebooks (id, name) VALUES ('nb-a', 'A')")
    conn.execute("INSERT INTO notebooks (id, name) VALUES ('nb-b', 'B')")
    conn.commit()
    with patch("app.routers.conversations_router.get_connection", return_value=conn), \
         TestClient(app) as c:
        yield c


def test_conversation_upsert_checks_notebook_ownership(conv_client: TestClient) -> None:
    """Same-notebook upsert updates in place; cross-notebook id is rejected with 409."""
    create = conv_client.post(
        "/api/conversations",
        json={"notebook_id": "nb-a", "title": "chat", "messages": [{"role": "user", "text": "hi"}]},
    )
    assert create.status_code == 200
    conv_id = create.json()["id"]

    # Upsert into the same notebook: same id, updated messages.
    upd = conv_client.post(
        "/api/conversations",
        json={"id": conv_id, "notebook_id": "nb-a", "title": "chat", "messages": [{"role": "user", "text": "hello"}]},
    )
    assert upd.status_code == 200
    assert upd.json()["id"] == conv_id
    assert upd.json()["messages"] == [{"role": "user", "text": "hello"}]

    # Upsert the same id into a different notebook: rejected, not overwritten.
    cross = conv_client.post(
        "/api/conversations",
        json={"id": conv_id, "notebook_id": "nb-b", "title": "chat", "messages": [{"role": "user", "text": "x"}]},
    )
    assert cross.status_code == 409


def test_sessions_list_most_recently_saved_first(conv_client: TestClient) -> None:
    """A notebook holds several sessions; re-saving one moves it to the top."""
    ids = []
    for title in ("first", "second"):
        r = conv_client.post(
            "/api/conversations",
            json={"notebook_id": "nb-a", "title": title, "messages": [{"id": title, "role": "user"}]},
        )
        ids.append(r.json()["id"])
    time.sleep(0.02)  # updated_at has millisecond precision
    conv_client.post(
        "/api/conversations",
        json={"id": ids[0], "notebook_id": "nb-a", "title": "renamed", "messages": [{"id": "first", "role": "user"}]},
    )
    listed = conv_client.get("/api/conversations", params={"notebook_id": "nb-a"}).json()
    assert [c["id"] for c in listed][0] == ids[0]
    assert {c["title"] for c in listed} == {"renamed", "second"}


def test_dedupe_drops_stale_snapshots_and_keeps_distinct_sessions() -> None:
    import json

    from app.db.schema import _dedupe_conversation_snapshots

    conn = sqlite3.connect(":memory:")
    create_tables(conn)
    conn.execute("INSERT INTO notebooks (id, name) VALUES ('nb-a', 'A')")
    conn.execute("INSERT INTO notebooks (id, name) VALUES ('nb-b', 'B')")

    def add(row_id: str, notebook: str, message_ids: list, created: str) -> None:
        conn.execute(
            "INSERT INTO saved_conversations (id, notebook_id, messages_json, created_at) VALUES (?, ?, ?, ?)",
            (row_id, notebook, json.dumps([{"id": i} for i in message_ids]), created),
        )

    add("snap1", "nb-a", ["m1"], "2026-05-20 10:00:00")
    add("snap2", "nb-a", ["m1", "m2"], "2026-05-20 10:01:00")
    add("full", "nb-a", ["m1", "m2", "m3"], "2026-05-20 10:02:00")
    add("copy", "nb-a", ["m1", "m2", "m3"], "2026-05-20 10:03:00")  # equal: newest wins
    add("other", "nb-a", ["x1"], "2026-05-21 10:00:00")              # a different session
    add("elsewhere", "nb-b", ["m1"], "2026-05-20 10:00:00")          # other notebook: kept

    # the oldest messages carry no id: they match by role and text
    for row_id, texts in (("old1", ["hi"]), ("old2", ["hi", "more"]), ("old-other", ["bye"])):
        conn.execute(
            "INSERT INTO saved_conversations (id, notebook_id, messages_json) VALUES (?, 'nb-b', ?)",
            (row_id, json.dumps([{"role": "user", "text": t} for t in texts])),
        )

    _dedupe_conversation_snapshots(conn)
    _dedupe_conversation_snapshots(conn)  # idempotent
    left = {r[0] for r in conn.execute("SELECT id FROM saved_conversations")}
    assert left == {"copy", "other", "elsewhere", "old2", "old-other"}
