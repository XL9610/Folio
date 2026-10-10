"""Integration tests for health check endpoints and system health."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_key(self):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    def test_health_has_components_list(self):
        resp = client.get("/health")
        data = resp.json()
        assert "components" in data
        assert isinstance(data["components"], list)
        assert len(data["components"]) >= 2  # at least sqlite + dense + keyword

    def test_health_sqlite_component(self):
        resp = client.get("/health")
        data = resp.json()
        sqlite = next(
            (c for c in data["components"] if c["component"] == "sqlite"), None
        )
        assert sqlite is not None
        assert sqlite["healthy"] is True

    def test_health_keyword_component(self):
        resp = client.get("/health")
        data = resp.json()
        kw = next(
            (c for c in data["components"] if c["component"] == "keyword_search"),
            None,
        )
        assert kw is not None
        assert kw["healthy"] is True

    def test_health_component_schema(self):
        resp = client.get("/health")
        data = resp.json()
        for c in data["components"]:
            assert "component" in c
            assert "healthy" in c
            assert "message" in c
            assert "latency_ms" in c
            assert isinstance(c["healthy"], bool)
            assert isinstance(c["message"], str)
            assert isinstance(c["latency_ms"], (int, float))


class TestSystemHealthService:
    """Tests for SystemHealthService static methods."""

    def test_check_sqlite_healthy(self):
        from app.services.system.health import SystemHealthService

        result = SystemHealthService.check_sqlite()
        assert result.healthy is True
        assert result.component == "sqlite"
        assert result.message == "connected"

    def test_check_keyword_backend_healthy(self):
        from app.services.system.health import SystemHealthService

        result = SystemHealthService.check_keyword_backend()
        assert result.healthy is True
        assert result.component == "keyword_search"

    def test_check_all_returns_list(self):
        from app.services.system.health import SystemHealthService

        results = SystemHealthService.check_all(dense_timeout=1.0)
        assert isinstance(results, list)
        assert len(results) >= 2
        component_names = {r.component for r in results}
        assert "sqlite" in component_names
        assert "dense" in component_names
        assert "keyword_search" in component_names
