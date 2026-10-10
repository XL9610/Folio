"""Aggregated health checks for all subsystems."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ComponentHealth:
    component: str
    healthy: bool
    message: str = ""
    latency_ms: float = 0.0


class SystemHealthService:

    @staticmethod
    def check_dense(timeout: float = 2.0) -> ComponentHealth:
        from ..index.sqlite_vec_store import SqliteVecStore

        try:
            store = SqliteVecStore()
            healthy, msg, ms = store.health_check(timeout)
            return ComponentHealth(
                component="dense",
                healthy=healthy,
                message=msg,
                latency_ms=ms,
            )
        except Exception as exc:
            return ComponentHealth(
                component="dense",
                healthy=False,
                message=str(exc),
            )

    @staticmethod
    def check_keyword_backend() -> ComponentHealth:
        try:
            from ..retrieval.keyword_backend import get_keyword_backend

            backend = get_keyword_backend()
            return ComponentHealth(
                component="keyword_search",
                healthy=True,
                message=backend.name(),
            )
        except Exception as exc:
            return ComponentHealth(
                component="keyword_search",
                healthy=False,
                message=str(exc),
            )

    @staticmethod
    def check_sqlite() -> ComponentHealth:
        from ...db.session import get_connection

        try:
            t0 = time.monotonic()
            conn = get_connection()
            conn.execute("SELECT 1")
            elapsed = (time.monotonic() - t0) * 1000
            return ComponentHealth(
                component="sqlite",
                healthy=True,
                message="connected",
                latency_ms=round(elapsed, 2),
            )
        except Exception as exc:
            return ComponentHealth(
                component="sqlite",
                healthy=False,
                message=str(exc),
            )

    @staticmethod
    def check_embedder() -> ComponentHealth:
        try:
            from ..embedding.sentence_transformer_embedder import SentenceTransformerEmbedder

            t0 = time.monotonic()
            embedder = SentenceTransformerEmbedder()
            # Lightweight: just verify the model loads
            embedder.embed_query("health check")
            elapsed = (time.monotonic() - t0) * 1000
            return ComponentHealth(
                component="embedder",
                healthy=True,
                message="model loaded",
                latency_ms=round(elapsed, 2),
            )
        except Exception as exc:
            return ComponentHealth(
                component="embedder",
                healthy=False,
                message=str(exc),
            )

    @staticmethod
    def check_all(
        dense_timeout: float = 2.0,
        include_embedder: bool = False,
    ) -> list[ComponentHealth]:
        results: list[ComponentHealth] = []

        # SQLite (always checked — fast)
        results.append(SystemHealthService.check_sqlite())

        # Dense (sqlite-vec, in-process)
        results.append(SystemHealthService.check_dense(timeout=dense_timeout))

        # Keyword backend
        results.append(SystemHealthService.check_keyword_backend())

        # Embedder (slow first-load — optional)
        if include_embedder:
            results.append(SystemHealthService.check_embedder())

        return results
