"""Database connection management.

Provides thread-local SQLite connections. Each thread gets its own
connection, avoiding cross-thread sharing issues while maintaining
WAL-mode concurrency for the local desktop use case.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import sqlite_vec

from ..core.config import config

# Thread-local storage — each thread gets its own SQLite connection
_local = threading.local()


def _db_path() -> Path:
    """Extract the file path from the DATABASE_URL."""
    url = config.database_url
    # Expected format: sqlite:///./rag_companion.db
    path_str = url[len("sqlite:///"):] if url.startswith("sqlite:///") else url
    db_path = Path(path_str)
    # Resolve relative paths against the local data directory
    if not db_path.is_absolute():
        db_path = (Path(config.local_data_dir) / db_path).resolve()
    return db_path


def get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it if needed.

    Each thread maintains its own connection via threading.local().
    This avoids the concurrency risks of a single shared connection
    with check_same_thread=False across multiple threads.
    """
    conn = getattr(_local, 'conn', None)
    if conn is None:
        path = _db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        # Concurrent writer threads now share one DB for chunks + vectors; wait
        # up to 5s for the write lock instead of failing immediately.
        conn.execute("PRAGMA busy_timeout=5000")
        # Load sqlite-vec so vec0 virtual tables are available on this connection.
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        _local.conn = conn
    return conn


def init_db() -> None:
    """Initialise database tables.

    Safe to call multiple times — uses IF NOT EXISTS.
    Called automatically on application startup.
    """
    from .schema import create_tables
    conn = get_connection()
    create_tables(conn)
