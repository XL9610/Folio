"""Tests for Review Basket repository and API."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.db.schema import create_tables
from app.main import app
from app.repositories.artifact_repository import create_artifact, delete_artifact
from app.repositories.notebook_repository import create_notebook
from app.repositories.review_item_repository import list_review_items, upsert_review_item


@pytest.fixture
def review_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    create_tables(conn)
    with (
        patch("app.repositories.notebook_repository.get_connection", return_value=conn),
        patch("app.repositories.artifact_repository.get_connection", return_value=conn),
        patch("app.repositories.review_item_repository.get_connection", return_value=conn),
        patch("app.db.session.get_connection", return_value=conn),
    ):
        yield conn
    conn.close()


@pytest.fixture
def review_context(review_db):
    nb = create_notebook("Review")
    artifact = create_artifact(
        notebook_id=nb["id"],
        kind="quiz",
        title="Chapter Quiz",
        content={"questions": []},
        source_document_ids=["doc-1"],
        params={},
    )
    return nb, artifact


def test_upsert_review_item_updates_without_duplicate(review_context):
    nb, artifact = review_context

    created = upsert_review_item(
        notebook_id=nb["id"],
        artifact_id=artifact["id"],
        item_type="flashcard",
        item_index=0,
        prompt="Binary search",
        answer="Search over sorted data.",
        is_starred=True,
    )
    updated = upsert_review_item(
        notebook_id=nb["id"],
        artifact_id=artifact["id"],
        item_type="flashcard",
        item_index=0,
        prompt="Binary search",
        answer="Divide-and-conquer search over sorted data.",
        is_mistake=True,
    )

    items = list_review_items(nb["id"])
    assert len(items) == 1
    assert updated["id"] == created["id"]
    assert items[0]["answer"] == "Divide-and-conquer search over sorted data."
    assert items[0]["is_starred"] is True
    assert items[0]["is_mistake"] is True


def test_review_item_filters(review_context):
    nb, artifact = review_context

    upsert_review_item(
        notebook_id=nb["id"],
        artifact_id=artifact["id"],
        item_type="flashcard",
        item_index=0,
        prompt="Term",
        answer="Definition",
        is_starred=True,
    )
    upsert_review_item(
        notebook_id=nb["id"],
        artifact_id=artifact["id"],
        item_type="quiz_question",
        item_index=1,
        prompt="Question",
        answer="A. Correct",
        is_mistake=True,
    )

    assert len(list_review_items(nb["id"], "all")) == 2
    assert len(list_review_items(nb["id"], "starred")) == 1
    assert len(list_review_items(nb["id"], "mistakes")) == 1
    assert len(list_review_items(nb["id"], "flashcards")) == 1
    assert len(list_review_items(nb["id"], "quiz")) == 1


def test_delete_artifact_removes_review_items(review_context):
    nb, artifact = review_context
    upsert_review_item(
        notebook_id=nb["id"],
        artifact_id=artifact["id"],
        item_type="quiz_question",
        item_index=0,
        prompt="Question",
        is_mistake=True,
    )

    assert len(list_review_items(nb["id"])) == 1
    assert delete_artifact(artifact["id"]) is True
    assert len(list_review_items(nb["id"])) == 0


def test_artifact_delete_endpoint_removes_review_items_and_404s(review_context):
    nb, artifact = review_context
    client = TestClient(app)
    upsert_review_item(
        notebook_id=nb["id"],
        artifact_id=artifact["id"],
        item_type="quiz_question",
        item_index=0,
        prompt="Question",
        is_mistake=True,
    )

    resp = client.delete(f"/api/artifacts/{artifact['id']}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["artifact_id"] == artifact["id"]
    assert len(list_review_items(nb["id"])) == 0

    missing = client.delete(f"/api/artifacts/{artifact['id']}")
    assert missing.status_code == 404


def test_review_items_api_create_and_filter(review_context):
    nb, artifact = review_context
    client = TestClient(app)

    resp = client.post(
        "/api/review-items",
        json={
            "notebook_id": nb["id"],
            "artifact_id": artifact["id"],
            "item_type": "quiz_question",
            "item_index": 3,
            "prompt": "What is binary search?",
            "answer": "B. Search sorted data",
            "choices": ["A. Linear", "B. Search sorted data"],
            "explanation": "Binary search halves the search space.",
            "source_document_id": "doc-1",
            "source_chunk_id": "chunk-1",
            "section_path": "Chapter 4 > Search",
            "page_start": 82,
            "page_end": 83,
            "is_mistake": True,
            "last_user_answer": "A. Linear",
        },
    )

    assert resp.status_code == 200, resp.text
    item = resp.json()
    assert item["is_mistake"] is True
    assert item["choices"] == ["A. Linear", "B. Search sorted data"]

    resp = client.get(f"/api/review-items?notebook_id={nb['id']}&filter=mistakes")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["review_items"]) == 1
    assert data["review_items"][0]["prompt"] == "What is binary search?"
