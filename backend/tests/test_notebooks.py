"""Tests for notebook repository and router.

Uses in-memory SQLite for all tests -- no disk I/O.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.db.schema import create_tables
from app.main import app
from app.repositories.notebook_repository import (
    _derive_status,
    create_notebook,
    delete_notebook,
    get_notebook,
    get_notebook_with_stats,
    list_notebooks,
    touch_notebook,
    update_notebook,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn() -> sqlite3.Connection:
    """Create a fresh in-memory SQLite connection with standard settings."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_db() -> Generator[sqlite3.Connection, None, None]:
    """Provide a standalone in-memory SQLite connection for repository tests.

    Patches ``get_connection`` where the repository functions actually
    reference it (they import it via ``from ..db.session import
    get_connection`` so the local name in the repository module must be
    patched, not the original in ``app.db.session``).
    """
    conn = _make_conn()
    create_tables(conn)
    with patch(
        "app.repositories.notebook_repository.get_connection",
        return_value=conn,
    ):
        yield conn


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient with an in-memory database override.

    Two patches are needed:

    1. ``app.repositories.notebook_repository.get_connection``
       -- the repository functions are imported into the router module
       via a ``from`` import and call ``get_connection`` through their
       own local reference.

    2. ``app.db.session.get_connection``
       -- the FastAPI lifespan calls ``init_db()`` which uses
       ``get_connection`` from its own module.
    """
    conn = _make_conn()
    create_tables(conn)
    with (
        patch(
            "app.repositories.notebook_repository.get_connection",
            return_value=conn,
        ),
        patch(
            "app.repositories.document_repository.get_connection",
            return_value=conn,
        ),
        patch(
            "app.db.session.get_connection",
            return_value=conn,
        ),
        TestClient(app) as c,
    ):
        yield c


# ---------------------------------------------------------------------------
# Repository tests -- _derive_status
# ---------------------------------------------------------------------------


def test_derive_status_empty() -> None:
    assert _derive_status(0, 0, 0, 0) == "empty"


def test_derive_status_processing() -> None:
    assert _derive_status(1, 0, 1, 0) == "processing"


def test_derive_status_failed() -> None:
    assert _derive_status(3, 0, 0, 3) == "failed"


def test_derive_status_ready() -> None:
    assert _derive_status(2, 2, 0, 0) == "ready"


def test_derive_status_partial() -> None:
    assert _derive_status(4, 2, 0, 2) == "partial"


# ---------------------------------------------------------------------------
# Repository tests -- create_notebook
# ---------------------------------------------------------------------------


def test_create_notebook(test_db: sqlite3.Connection) -> None:
    nb = create_notebook("My Notebook", description="A test notebook")
    assert nb["name"] == "My Notebook"
    assert nb["description"] == "A test notebook"
    assert nb["id"]
    assert nb["source_count"] == 0
    assert nb["status"] == "empty"


def test_create_notebook_defaults(test_db: sqlite3.Connection) -> None:
    nb = create_notebook()
    assert nb["name"] == "Default Notebook"
    assert nb["description"] == ""


# ---------------------------------------------------------------------------
# Repository tests -- list_notebooks
# ---------------------------------------------------------------------------


def test_list_notebooks_empty(test_db: sqlite3.Connection) -> None:
    notebooks = list_notebooks()
    assert notebooks == []


def test_list_notebooks_with_stats(test_db: sqlite3.Connection) -> None:
    nb = create_notebook("Stats Notebook")
    conn = test_db

    conn.execute(
        "INSERT INTO source_documents (id, notebook_id, filename, status) VALUES (?, ?, ?, ?)",
        ("doc-1", nb["id"], "a.pdf", "completed"),
    )
    conn.execute(
        "INSERT INTO source_documents (id, notebook_id, filename, status) VALUES (?, ?, ?, ?)",
        ("doc-2", nb["id"], "b.pdf", "processing"),
    )
    conn.execute(
        "INSERT INTO source_documents (id, notebook_id, filename, status) VALUES (?, ?, ?, ?)",
        ("doc-3", nb["id"], "c.pdf", "failed"),
    )
    conn.commit()

    notebooks = list_notebooks()
    assert len(notebooks) == 1
    result = notebooks[0]
    assert result["name"] == "Stats Notebook"
    assert result["source_count"] == 3
    assert result["ready_count"] == 1
    assert result["processing_count"] == 1
    assert result["failed_count"] == 1
    assert result["status"] == "processing"  # processing_count > 0 takes priority


# ---------------------------------------------------------------------------
# Repository tests -- get_notebook
# ---------------------------------------------------------------------------


def test_get_notebook(test_db: sqlite3.Connection) -> None:
    nb = create_notebook("Find Me")
    found = get_notebook(nb["id"])
    assert found is not None
    assert found["name"] == "Find Me"


def test_get_notebook_missing(test_db: sqlite3.Connection) -> None:
    found = get_notebook("non-existent-id")
    assert found is None


def test_get_notebook_with_stats(test_db: sqlite3.Connection) -> None:
    nb = create_notebook("Stats")
    conn = test_db
    conn.execute(
        "INSERT INTO source_documents (id, notebook_id, filename, status) VALUES (?, ?, ?, ?)",
        ("doc-a", nb["id"], "x.pdf", "completed"),
    )
    conn.execute(
        "INSERT INTO source_documents (id, notebook_id, filename, status) VALUES (?, ?, ?, ?)",
        ("doc-b", nb["id"], "y.pdf", "completed"),
    )
    conn.commit()

    result = get_notebook_with_stats(nb["id"])
    assert result is not None
    assert result["source_count"] == 2
    assert result["ready_count"] == 2
    assert result["status"] == "ready"


# ---------------------------------------------------------------------------
# Repository tests -- update_notebook
# ---------------------------------------------------------------------------


def test_update_notebook_name(test_db: sqlite3.Connection) -> None:
    nb = create_notebook("Original")
    updated = update_notebook(nb["id"], name="Renamed")
    assert updated is not None
    assert updated["name"] == "Renamed"
    assert updated.get("description", "") == ""


def test_update_notebook_description(test_db: sqlite3.Connection) -> None:
    nb = create_notebook("Keep Name", description="Old Desc")
    updated = update_notebook(nb["id"], description="New Desc")
    assert updated is not None
    assert updated["name"] == "Keep Name"
    assert updated.get("description", "") == "New Desc"


def test_update_notebook_missing(test_db: sqlite3.Connection) -> None:
    result = update_notebook("nonexistent", name="X")
    assert result is None


# ---------------------------------------------------------------------------
# Repository tests -- delete_notebook
# ---------------------------------------------------------------------------


def test_delete_notebook_empty(test_db: sqlite3.Connection) -> None:
    nb = create_notebook("To Delete")
    result = delete_notebook(nb["id"])
    assert result is True
    assert get_notebook(nb["id"]) is None


def test_delete_notebook_with_sources(test_db: sqlite3.Connection) -> None:
    nb = create_notebook("Has Sources")
    conn = test_db
    conn.execute(
        "INSERT INTO source_documents (id, notebook_id, filename, status) VALUES (?, ?, ?, ?)",
        ("doc-x", nb["id"], "file.pdf", "completed"),
    )
    conn.commit()

    assert delete_notebook(nb["id"]) is True
    # Cascade: the notebook and its source document are both gone.
    assert conn.execute(
        "SELECT COUNT(*) FROM notebooks WHERE id = ?", (nb["id"],)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM source_documents WHERE notebook_id = ?", (nb["id"],)
    ).fetchone()[0] == 0


def test_delete_notebook_missing(test_db: sqlite3.Connection) -> None:
    result = delete_notebook("no-such-id")
    assert result is False


# ---------------------------------------------------------------------------
# Repository tests -- touch_notebook (recently-opened ordering)
# ---------------------------------------------------------------------------


def test_touch_notebook_reorders(test_db: sqlite3.Connection) -> None:
    a = create_notebook("A")
    b = create_notebook("B")
    conn = test_db
    # Force distinct created_at values so the default ordering is deterministic.
    conn.execute("UPDATE notebooks SET created_at = '2020-01-01 00:00:00' WHERE id = ?", (a["id"],))
    conn.execute("UPDATE notebooks SET created_at = '2020-01-02 00:00:00' WHERE id = ?", (b["id"],))
    conn.commit()

    # Default: newer created_at first (B, then A).
    assert [n["id"] for n in list_notebooks()] == [b["id"], a["id"]]

    # Touching A floats it to the top.
    assert touch_notebook(a["id"]) is True
    assert [n["id"] for n in list_notebooks()] == [a["id"], b["id"]]


def test_touch_notebook_exposes_last_opened_at(test_db: sqlite3.Connection) -> None:
    nb = create_notebook("Touchable")
    assert touch_notebook(nb["id"]) is True
    match = [n for n in list_notebooks() if n["id"] == nb["id"]][0]
    assert match.get("last_opened_at")


def test_touch_notebook_missing(test_db: sqlite3.Connection) -> None:
    assert touch_notebook("no-such-id") is False


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


def test_list_notebooks_endpoint(client: TestClient) -> None:
    resp = client.get("/api/notebooks")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_notebook_endpoint(client: TestClient) -> None:
    resp = client.post(
        "/api/notebooks",
        json={"name": "API Notebook", "description": "from endpoint"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "API Notebook"
    assert data["description"] == "from endpoint"
    assert data["id"]
    assert data["source_count"] == 0
    assert data["status"] == "empty"


def test_create_notebook_validation(client: TestClient) -> None:
    resp = client.post("/api/notebooks", json={"name": ""})
    assert resp.status_code == 422

    resp = client.post("/api/notebooks", json={})
    assert resp.status_code == 422


def test_get_notebook_endpoint(client: TestClient) -> None:
    create = client.post("/api/notebooks", json={"name": "Get Test"})
    nb_id = create.json()["id"]

    resp = client.get(f"/api/notebooks/{nb_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Get Test"


def test_get_notebook_default(client: TestClient) -> None:
    resp = client.get("/api/notebooks/default")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Default Notebook"

    resp2 = client.get(f"/api/notebooks/{data['id']}")
    assert resp2.status_code == 200
    assert resp2.json()["id"] == data["id"]


def test_update_notebook_endpoint(client: TestClient) -> None:
    create = client.post("/api/notebooks", json={"name": "Before Update"})
    nb_id = create.json()["id"]

    resp = client.put(
        f"/api/notebooks/{nb_id}",
        json={"name": "After Update", "description": "new desc"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "After Update"
    assert data["description"] == "new desc"


def test_delete_notebook_endpoint(client: TestClient) -> None:
    create = client.post("/api/notebooks", json={"name": "Delete Me"})
    nb_id = create.json()["id"]

    resp = client.delete(f"/api/notebooks/{nb_id}")
    assert resp.status_code == 204

    resp2 = client.get(f"/api/notebooks/{nb_id}")
    assert resp2.status_code == 404


def test_delete_notebook_with_sources_cascades(client: TestClient) -> None:
    create = client.post("/api/notebooks", json={"name": "Has Source"})
    nb_id = create.json()["id"]

    # Directly insert a source document using the SAME patched connection
    import app.db.session  # use the session-level patch from client fixture

    conn = app.db.session.get_connection()
    conn.execute(
        "INSERT INTO source_documents (id, notebook_id, filename, status) VALUES (?, ?, ?, ?)",
        ("doc-z", nb_id, "z.pdf", "completed"),
    )
    conn.commit()

    class _FakeVecStore:
        def __init__(self, *a, **k): pass
        def delete_by_document_id(self, doc_id: str) -> None:
            pass

    class _FakeArtifactStore:
        def __init__(self, *a, **k): pass
        def delete(self, doc_id: str) -> None:
            pass

    with (
        patch("app.routers.notebooks_router.SqliteVecStore", _FakeVecStore),
        patch("app.routers.notebooks_router.ExtractionArtifactStore", _FakeArtifactStore),
    ):
        resp = client.delete(f"/api/notebooks/{nb_id}")
    assert resp.status_code == 204
    # Cascade: the notebook is gone from the listing.
    assert client.get("/api/notebooks").json() == []


def test_delete_notebook_removes_extraction_artifacts(client: TestClient) -> None:
    """Deleting a notebook also deletes each document's on-disk artifacts."""
    create = client.post("/api/notebooks", json={"name": "Artifact Cleanup"})
    nb_id = create.json()["id"]

    import app.db.session
    conn = app.db.session.get_connection()
    conn.execute(
        "INSERT INTO source_documents (id, notebook_id, filename, status) VALUES (?, ?, ?, ?)",
        ("doc-art", nb_id, "a.pdf", "completed"),
    )
    conn.commit()

    artifact_deletes = []

    class _FakeArtifactStore:
        def __init__(self, *a, **k): pass
        def delete(self, doc_id: str) -> None:
            artifact_deletes.append(doc_id)

    class _FakeVecStore:
        def __init__(self, *a, **k): pass
        def delete_by_document_id(self, doc_id: str) -> None:
            pass

    with (
        patch("app.routers.notebooks_router.ExtractionArtifactStore", _FakeArtifactStore),
        patch("app.routers.notebooks_router.SqliteVecStore", _FakeVecStore),
    ):
        resp = client.delete(f"/api/notebooks/{nb_id}")

    assert resp.status_code == 204
    assert artifact_deletes == ["doc-art"]


def test_api_token_enforced_when_configured(client: TestClient, monkeypatch) -> None:
    """/api/* requires X-Auth-Token when FOLIO_API_TOKEN is set (401/401/200)."""
    import app.main as main
    monkeypatch.setattr(main, "API_TOKEN", "secret-token")

    assert client.get("/api/notebooks").status_code == 401
    assert client.get("/api/notebooks", headers={"X-Auth-Token": "wrong"}).status_code == 401
    assert client.get("/api/notebooks", headers={"X-Auth-Token": "secret-token"}).status_code == 200
    # OPTIONS preflight is exempt from token enforcement.
    assert client.options("/api/notebooks").status_code != 401


def test_delete_notebook_fails_closed_on_artifact_error(client: TestClient) -> None:
    """Notebook delete keeps DB records when artifact cleanup fails."""
    create = client.post("/api/notebooks", json={"name": "NB Fail"})
    nb_id = create.json()["id"]

    import app.db.session
    conn = app.db.session.get_connection()
    conn.execute(
        "INSERT INTO source_documents (id, notebook_id, filename, status) VALUES (?, ?, ?, ?)",
        ("doc-art", nb_id, "a.pdf", "completed"),
    )
    conn.commit()

    class _FakeVecStore:
        def __init__(self, *a, **k): pass
        def delete_by_document_id(self, doc_id: str) -> None:
            pass

    class _FailingArtifactStore:
        def __init__(self, *a, **k): pass
        def delete(self, doc_id: str) -> None:
            raise PermissionError("file locked")

    with (
        patch("app.routers.notebooks_router.SqliteVecStore", _FakeVecStore),
        patch("app.routers.notebooks_router.ExtractionArtifactStore", _FailingArtifactStore),
    ):
        resp = client.delete(f"/api/notebooks/{nb_id}")

    assert resp.status_code == 503
    # Notebook (and its document) still present for retry.
    assert client.get(f"/api/notebooks/{nb_id}").status_code == 200


def test_delete_document_removes_artifacts(client: TestClient) -> None:
    """DELETE /api/documents/{id} also deletes the on-disk artifacts."""
    create = client.post("/api/notebooks", json={"name": "Doc Delete"})
    nb_id = create.json()["id"]

    import app.db.session
    conn = app.db.session.get_connection()
    conn.execute(
        "INSERT INTO source_documents (id, notebook_id, filename, status) VALUES (?, ?, ?, ?)",
        ("doc-del", nb_id, "a.pdf", "completed"),
    )
    conn.commit()

    artifact_deletes = []

    class _FakeVecStore:
        def __init__(self, *a, **k): pass
        def delete_by_document_id(self, doc_id: str) -> None:
            pass

    class _FakeArtifactStore:
        def __init__(self, *a, **k): pass
        def delete(self, doc_id: str) -> None:
            artifact_deletes.append(doc_id)

    with (
        patch("app.routers.documents_router.SqliteVecStore", _FakeVecStore),
        patch("app.routers.documents_router.ExtractionArtifactStore", _FakeArtifactStore),
    ):
        resp = client.delete("/api/documents/doc-del")

    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert artifact_deletes == ["doc-del"]
    assert client.get("/api/documents/doc-del").status_code == 404


def test_delete_document_fails_closed_on_artifact_error(client: TestClient) -> None:
    """Document delete keeps the DB record when artifact cleanup fails."""
    create = client.post("/api/notebooks", json={"name": "Doc Delete Fail"})
    nb_id = create.json()["id"]

    import app.db.session
    conn = app.db.session.get_connection()
    conn.execute(
        "INSERT INTO source_documents (id, notebook_id, filename, status) VALUES (?, ?, ?, ?)",
        ("doc-del", nb_id, "a.pdf", "completed"),
    )
    conn.commit()

    class _FakeVecStore:
        def __init__(self, *a, **k): pass
        def delete_by_document_id(self, doc_id: str) -> None:
            pass

    class _FailingArtifactStore:
        def __init__(self, *a, **k): pass
        def delete(self, doc_id: str) -> None:
            raise PermissionError("file locked")

    with (
        patch("app.routers.documents_router.SqliteVecStore", _FakeVecStore),
        patch("app.routers.documents_router.ExtractionArtifactStore", _FailingArtifactStore),
    ):
        resp = client.delete("/api/documents/doc-del")

    assert resp.status_code == 503
    # DB record retained so the user can retry.
    assert client.get("/api/documents/doc-del").status_code == 200


def test_mark_opened_endpoint(client: TestClient) -> None:
    create = client.post("/api/notebooks", json={"name": "Touch Me"})
    nb_id = create.json()["id"]

    resp = client.put(f"/api/notebooks/{nb_id}/opened")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    listed = client.get("/api/notebooks").json()
    match = [n for n in listed if n["id"] == nb_id][0]
    assert match["last_opened_at"]


def test_mark_opened_not_found(client: TestClient) -> None:
    resp = client.put("/api/notebooks/nonexistent-id/opened")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# generate-title endpoint tests
# ---------------------------------------------------------------------------


def test_generate_title_no_sources(client: TestClient) -> None:
    """When notebook has no sources, returns a default title."""
    create = client.post("/api/notebooks", json={"name": "Empty NB"})
    nb_id = create.json()["id"]

    resp = client.post(f"/api/notebooks/{nb_id}/generate-title")
    assert resp.status_code == 200
    data = resp.json()
    assert "title" in data
    assert len(data["title"]) > 0


def test_generate_title_with_sources_no_llm(client: TestClient) -> None:
    """When LLM is not configured, falls back to first document display title."""
    create = client.post("/api/notebooks", json={"name": "With Sources"})
    nb_id = create.json()["id"]

    import app.db.session
    conn = app.db.session.get_connection()
    conn.execute(
        "INSERT INTO source_documents (id, notebook_id, filename, display_title, status) "
        "VALUES (?, ?, ?, ?, ?)",
        ("doc-abc", nb_id, "biology101.pdf", "Introduction to Biology", "completed"),
    )
    conn.commit()

    with patch("app.routers.notebooks_router.LLMConfig") as mock_cfg_cls:
        mock_cfg = mock_cfg_cls.return_value
        mock_cfg.is_configured = False
        resp = client.post(f"/api/notebooks/{nb_id}/generate-title")

    assert resp.status_code == 200
    data = resp.json()
    assert "title" in data
    # Falls back to document title since LLM is not configured in test
    assert data.get("fallback") is True


def test_generate_title_notebook_not_found(client: TestClient) -> None:
    """Returns 404 for nonexistent notebook."""
    resp = client.post("/api/notebooks/nonexistent-id/generate-title")
    assert resp.status_code == 404


def test_generate_title_default_notebook(client: TestClient) -> None:
    """The 'default' alias resolves correctly."""
    resp = client.post("/api/notebooks/default/generate-title")
    assert resp.status_code == 200
    assert "title" in resp.json()
