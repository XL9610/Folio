"""Database layer — SQLite for canonical metadata and FTS."""

from .session import get_connection, init_db

__all__ = ["get_connection", "init_db"]
