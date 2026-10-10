"""Tests for /api/generate endpoint (HTTP layer, DB mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def fake_llm_config(is_configured=True):
    mock_cfg = MagicMock()
    mock_cfg.is_configured = is_configured
    mock_cfg.model = "test-model"
    mock_cfg.max_tokens = 1024
    mock_cfg.temperature = 0.1
    return mock_cfg


# ── /api/generate tests ──────────────────────────────────────────────

def test_generate_llm_not_configured(client):
    with patch("app.routers.generate_router.LLMConfig",
               return_value=fake_llm_config(is_configured=False)):
        resp = client.post("/api/generate", json={
            "kind": "summary",
            "notebook_id": "default",
            "document_ids": ["d1"],
            "params": {},
        })
        assert resp.status_code == 503
        assert resp.json()["detail"]["code"] == "llm_not_configured"


def test_generate_unsupported_kind_returns_501(client):
    with patch("app.routers.generate_router.LLMConfig",
               return_value=fake_llm_config(is_configured=True)):
        resp = client.post("/api/generate", json={
            "kind": "diagram",
            "notebook_id": "default",
            "document_ids": ["d1"],
            "params": {},
        })
        assert resp.status_code == 501
        assert resp.json()["detail"]["code"] == "not_implemented"


def test_generate_summary_happy_path(client):
    """Full happy path: LLM configured, document found, summary generated, artifact persisted."""
    mock_llm = MagicMock()
    mock_llm.complete_chat.return_value = "# Test Summary\n\nGenerated content."

    fake_artifact = {
        "id": "art-1", "notebook_id": "nb1", "kind": "summary",
        "title": "Test Summary", "content_json": '{"markdown":"# Test"}',
        "source_document_ids": '["d1"]', "params_json": "{}",
    }

    with patch("app.routers.generate_router.LLMConfig",
               return_value=fake_llm_config(is_configured=True)):
        with patch("app.routers.generate_router.OpenAICompatibleClient",
                   return_value=mock_llm):
            with patch("app.routers.generate_router.get_or_create_default",
                       return_value={"id": "nb1"}):
                with patch("app.routers.generate_router.list_documents",
                           return_value=[{"id": "d1", "notebook_id": "nb1"}]):
                    with patch("app.routers.generate_router.get_document",
                               return_value={"id": "d1", "notebook_id": "nb1", "filename": "test.pdf"}):
                        with patch("app.routers.generate_router.create_artifact",
                                   return_value=fake_artifact):
                            resp = client.post("/api/generate", json={
                                "kind": "summary",
                                "notebook_id": "default",
                                "document_ids": ["d1"],
                                "params": {"scope": "Chapter 1", "length": "short", "style": "bullets"},
                            })
                            assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
                            data = resp.json()
                            assert data["kind"] == "summary"
                            assert len(data["markdown"]) > 0
                            assert data["artifact_id"] == "art-1"
                            assert "stats" in data


def test_generate_summary_empty_scope(client):
    """Empty scope should work (all chunks mode)."""
    mock_llm = MagicMock()
    mock_llm.complete_chat.return_value = "# Overview\n\nWhole document summary."

    fake_artifact = {
        "id": "art-2", "notebook_id": "nb1", "kind": "summary",
        "title": "Overview", "content_json": "{}",
        "source_document_ids": "[]", "params_json": "{}",
    }

    with patch("app.routers.generate_router.LLMConfig",
               return_value=fake_llm_config(is_configured=True)):
        with patch("app.routers.generate_router.OpenAICompatibleClient",
                   return_value=mock_llm):
            with patch("app.routers.generate_router.get_or_create_default",
                       return_value={"id": "nb1"}):
                with patch("app.routers.generate_router.list_documents",
                           return_value=[{"id": "d1", "notebook_id": "nb1"}]):
                    with patch("app.routers.generate_router.get_document",
                               return_value={"id": "d1", "notebook_id": "nb1", "filename": "test.pdf"}):
                        with patch("app.routers.generate_router.create_artifact",
                                   return_value=fake_artifact):
                            resp = client.post("/api/generate", json={
                                "kind": "summary",
                                "notebook_id": "default",
                                "document_ids": ["d1"],
                                "params": {"scope": "", "length": "detailed", "style": "prose"},
                            })
                            assert resp.status_code == 200
                            data = resp.json()
                            assert "stats" in data


def test_generate_document_not_found(client):
    with patch("app.routers.generate_router.LLMConfig",
               return_value=fake_llm_config(is_configured=True)):
        resp = client.post("/api/generate", json={
            "kind": "summary",
            "notebook_id": "default",
            "document_ids": ["nonexistent-id"],
            "params": {},
        })
        assert resp.status_code == 404


# ── /api/generate flashcards tests ──────────────────────────────────

def test_generate_flashcards_happy_path(client):
    mock_llm = MagicMock()
    mock_llm.complete_chat.return_value = '{"cards": [{"term": "T", "definition": "D", "source": 1}]}'
    fake_artifact = {"id": "art-fc", "notebook_id": "nb1", "kind": "flashcards",
                     "title": "FC", "content_json": "{}", "source_document_ids": "[]", "params_json": "{}"}

    from app.services.generation.scope_resolver import ScopeResult, ScopeResolver
    from app.services.generation.flashcards_service import FlashcardsGenerator
    from app.services.retrieval.base import RetrievedChunk
    fake_chunk = RetrievedChunk(chunk_id="c1", document_id="d1", text="test text",
                                title="", section_path="Ch 1", page_start=1, page_end=1)
    fake_resolver = MagicMock(spec=ScopeResolver)
    fake_resolver.resolve.return_value = ScopeResult(chunks=[fake_chunk], metadata={"layer": "toc"})
    fake_gen = FlashcardsGenerator(mock_llm, fake_resolver)

    with patch("app.routers.generate_router._get_flashcards_generator", return_value=fake_gen):
        with patch("app.routers.generate_router.LLMConfig", return_value=fake_llm_config(is_configured=True)):
            with patch("app.routers.generate_router.get_or_create_default", return_value={"id": "nb1"}):
                with patch("app.routers.generate_router.list_documents", return_value=[{"id": "d1", "notebook_id": "nb1"}]):
                    with patch("app.routers.generate_router.get_document",
                               return_value={"id": "d1", "notebook_id": "nb1", "filename": "test.pdf"}):
                        with patch("app.routers.generate_router.create_artifact", return_value=fake_artifact):
                            resp = client.post("/api/generate", json={
                                "kind": "flashcards",
                                "notebook_id": "default",
                                "document_ids": ["d1"],
                                "params": {"scope": "", "count": 10},
                            })
                            assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
                            data = resp.json()
                            assert data["kind"] == "flashcards"
                            assert len(data["cards"]) >= 1
                            assert data["artifact_id"] == "art-fc"


def test_generate_mindmap_persists_with_json_serializable_sources(client):
    """Mind Map artifact persistence must not fail on citation serialization.

    Regression: sources were CitationItem dataclasses that json.dumps() could not
    serialize, causing a 500 at create_artifact. This test exercises the real
    create_artifact path (not mocked), so it fails if sources aren't plain dicts.
    """
    import sqlite3

    from app.db.schema import create_tables

    mock_llm = MagicMock()
    mock_llm.complete_chat.return_value = (
        '{"label": "Topic", "children": ['
        '{"label": "Concept A", "children": [{"label": "A1"}, {"label": "A2"}]},'
        '{"label": "Concept B", "children": [{"label": "B1"}, {"label": "B2"}]},'
        '{"label": "Concept C", "children": [{"label": "C1"}]},'
        '{"label": "Concept D", "children": [{"label": "D1"}]}'
        "]}"
    )

    from app.services.generation.scope_resolver import ScopeResult, ScopeResolver
    from app.services.generation.mindmap_service import MindmapGenerator
    from app.services.retrieval.base import RetrievedChunk
    fake_chunk = RetrievedChunk(chunk_id="c1", document_id="d1", text="test text about the topic",
                                title="", section_path="Ch 1", page_start=1, page_end=1)
    fake_resolver = MagicMock(spec=ScopeResolver)
    fake_resolver.resolve.return_value = ScopeResult(chunks=[fake_chunk], metadata={"layer": "toc"})
    fake_gen = MindmapGenerator(mock_llm, fake_resolver)

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    create_tables(conn)
    conn.execute("INSERT INTO notebooks (id, name) VALUES ('nb1', 'Test Notebook')")
    conn.commit()

    with patch("app.routers.generate_router._get_mindmap_generator", return_value=fake_gen):
        with patch("app.routers.generate_router.LLMConfig", return_value=fake_llm_config(is_configured=True)):
            with patch("app.routers.generate_router.get_or_create_default", return_value={"id": "nb1"}):
                with patch("app.routers.generate_router.list_documents", return_value=[{"id": "d1", "notebook_id": "nb1"}]):
                    with patch("app.routers.generate_router.get_document",
                               return_value={"id": "d1", "notebook_id": "nb1", "filename": "test.pdf"}):
                        with patch("app.repositories.artifact_repository.get_connection", return_value=conn):
                            resp = client.post("/api/generate", json={
                                "kind": "mindmap",
                                "notebook_id": "default",
                                "document_ids": ["d1"],
                                "params": {"scope": ""},
                            })
                            assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
                            data = resp.json()
                            assert data["kind"] == "mindmap"
                            assert data["artifact_id"]


def test_generate_flashcards_count_out_of_range(client):
    with patch("app.routers.generate_router.LLMConfig", return_value=fake_llm_config(is_configured=True)):
        resp = client.post("/api/generate", json={
            "kind": "flashcards",
            "notebook_id": "default",
            "document_ids": ["d1"],
            "params": {"count": 100},  # exceeds max 50
        })
        assert resp.status_code == 422  # pydantic validation error


def test_generate_flashcards_count_below_min(client):
    """count=5 should fail Pydantic validation (ge=10)."""
    with patch("app.routers.generate_router.LLMConfig", return_value=fake_llm_config(is_configured=True)):
        resp = client.post("/api/generate", json={
            "kind": "flashcards",
            "notebook_id": "default",
            "document_ids": ["d1"],
            "params": {"count": 5},
        })
        assert resp.status_code == 422


def test_generate_quiz_happy_path(client):
    """Full happy path: LLM configured, quiz generated, artifact persisted."""
    mock_llm = MagicMock()
    mock_llm.complete_chat.return_value = '{"questions":[{"question":"What is X?","choices":["A. Yes","B. No","C. Maybe","D. None"],"answer":"A","explanation":"Because...","source":1}]}'
    fake_artifact = {"id": "art-quiz", "notebook_id": "nb1", "kind": "quiz",
                     "title": "Quiz", "content_json": "{}", "source_document_ids": "[]", "params_json": "{}"}

    from app.services.generation.scope_resolver import ScopeResult, ScopeResolver
    from app.services.generation.quiz_service import QuizGenerator
    from app.services.retrieval.base import RetrievedChunk
    fake_chunk = RetrievedChunk(chunk_id="c1", document_id="d1", text="test text",
                                title="", section_path="Ch 1", page_start=1, page_end=1)
    fake_resolver = MagicMock(spec=ScopeResolver)
    fake_resolver.resolve.return_value = ScopeResult(chunks=[fake_chunk], metadata={"layer": "toc"})
    fake_gen = QuizGenerator(mock_llm, fake_resolver)

    with patch("app.routers.generate_router._get_quiz_generator", return_value=fake_gen):
        with patch("app.routers.generate_router.LLMConfig", return_value=fake_llm_config(is_configured=True)):
            with patch("app.routers.generate_router.get_or_create_default", return_value={"id": "nb1"}):
                with patch("app.routers.generate_router.list_documents", return_value=[{"id": "d1", "notebook_id": "nb1"}]):
                    with patch("app.routers.generate_router.get_document",
                               return_value={"id": "d1", "notebook_id": "nb1", "filename": "test.pdf"}):
                        with patch("app.routers.generate_router.create_artifact", return_value=fake_artifact):
                            resp = client.post("/api/generate", json={
                                "kind": "quiz",
                                "notebook_id": "default",
                                "document_ids": ["d1"],
                                "params": {"quiz_count": 5, "quiz_difficulty": "medium"},
                            })
                            assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
                            data = resp.json()
                            assert data["kind"] == "quiz"
                            assert len(data["questions"]) >= 1
