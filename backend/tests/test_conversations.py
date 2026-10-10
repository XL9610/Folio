"""Tests for conversation save/upsert and notebook-ownership enforcement."""

from __future__ import annotations

import sqlite3
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
