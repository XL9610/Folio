"""SQLite schema — canonical metadata store for the RAG companion.

Tables:
  notebooks          — top-level organisation unit
  source_documents   — one PDF = one source document
  chunks             — text chunks linked to a source document
  chunks_fts         — FTS5 virtual table for BM25 keyword search
  ingestion_jobs     — tracking async ingestion status
"""

from __future__ import annotations

import logging
import os
import sqlite3

from ..core.env_file import set_env

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def create_tables(conn: sqlite3.Connection) -> None:
    """Create all tables if they do not already exist."""
    _create_notebooks(conn)
    _create_source_documents(conn)
    _create_chunks(conn)
    _ensure_chunk_context_columns(conn)
    _create_chunks_fts(conn)
    _create_ingestion_jobs(conn)
    _create_saved_artifacts(conn)
    _create_review_items(conn)
    _create_document_extractions(conn)
    _create_saved_conversations(conn)
    _create_indexes(conn)
    _set_version(conn)
    run_migrations(conn)
    conn.commit()


# ---------------------------------------------------------------------------
# Individual table creation helpers
# ---------------------------------------------------------------------------

def _create_notebooks(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notebooks (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)


def _create_source_documents(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source_documents (
            id            TEXT PRIMARY KEY,
            notebook_id   TEXT NOT NULL,
            filename      TEXT NOT NULL,
            display_title TEXT NOT NULL DEFAULT '',
            content_hash  TEXT NOT NULL DEFAULT '',
            page_count    INTEGER NOT NULL DEFAULT 0,
            chunk_count   INTEGER NOT NULL DEFAULT 0,
            status        TEXT NOT NULL DEFAULT 'pending',
            chunking_mode  TEXT NOT NULL DEFAULT '',
            structure_status TEXT NOT NULL DEFAULT 'unknown_legacy',
            embedded_toc_present INTEGER DEFAULT NULL,
            synthetic_outline_mode TEXT DEFAULT NULL,
            structure_confidence TEXT DEFAULT NULL,
            structure_user_note TEXT DEFAULT NULL,
            structure_llm_instruction TEXT DEFAULT NULL,
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (notebook_id) REFERENCES notebooks(id)
        )
    """)


def _create_chunks(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id              TEXT PRIMARY KEY,
            document_id     TEXT NOT NULL,
            chunk_index     INTEGER NOT NULL,
            text            TEXT NOT NULL,
            title           TEXT NOT NULL DEFAULT '',
            section_path    TEXT NOT NULL DEFAULT '',
            page_start      INTEGER NOT NULL DEFAULT -1,
            page_end        INTEGER NOT NULL DEFAULT -1,
            chunk_strategy  TEXT NOT NULL DEFAULT 'unknown',
            metadata_json   TEXT NOT NULL DEFAULT '{}',
            content_hash    TEXT NOT NULL DEFAULT '',
            context_prefix  TEXT NOT NULL DEFAULT '',
            context_mode    TEXT NOT NULL DEFAULT 'metadata',
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (document_id) REFERENCES source_documents(id)
        )
    """)


def _create_chunks_fts(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
        USING fts5(
            text,
            title,
            section_path,
            content='chunks',
            content_rowid='rowid'
        )
    """)
    # Drop old triggers first so CREATE TRIGGER IF NOT EXISTS bodies
    # are never stale — old databases still carry the pre-enrichment
    # trigger SQL.  Then recreate with enriched text logic.
    conn.executescript("""
        DROP TRIGGER IF EXISTS chunks_ai;
        DROP TRIGGER IF EXISTS chunks_ad;
        DROP TRIGGER IF EXISTS chunks_au;

        CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text, title, section_path)
            VALUES (
                new.rowid,
                CASE
                    WHEN trim(new.context_prefix) <> ''
                    THEN new.context_prefix || char(10) || char(10) || new.text
                    ELSE new.text
                END,
                new.title,
                new.section_path
            );
        END;

        CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, title, section_path)
            VALUES (
                'delete',
                old.rowid,
                CASE
                    WHEN trim(old.context_prefix) <> ''
                    THEN old.context_prefix || char(10) || char(10) || old.text
                    ELSE old.text
                END,
                old.title,
                old.section_path
            );
        END;

        CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, title, section_path)
            VALUES (
                'delete',
                old.rowid,
                CASE
                    WHEN trim(old.context_prefix) <> ''
                    THEN old.context_prefix || char(10) || char(10) || old.text
                    ELSE old.text
                END,
                old.title,
                old.section_path
            );
            INSERT INTO chunks_fts(rowid, text, title, section_path)
            VALUES (
                new.rowid,
                CASE
                    WHEN trim(new.context_prefix) <> ''
                    THEN new.context_prefix || char(10) || char(10) || new.text
                    ELSE new.text
                END,
                new.title,
                new.section_path
            );
        END;
    """)


def _create_ingestion_jobs(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            id            TEXT PRIMARY KEY,
            notebook_id   TEXT NOT NULL DEFAULT '',
            document_id   TEXT NOT NULL DEFAULT '',
            status        TEXT NOT NULL DEFAULT 'queued',
            stage         TEXT NOT NULL DEFAULT 'queued',
            progress_pct  INTEGER NOT NULL DEFAULT 0,
            error_message TEXT NOT NULL DEFAULT '',
            warnings_json TEXT NOT NULL DEFAULT '',
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)


def _migrate_ingestion_jobs(conn: sqlite3.Connection) -> None:
    """Rebuild ingestion_jobs with async-job columns (was empty; safe to drop)."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(ingestion_jobs)").fetchall()]
    if "stage" in cols:
        return
    conn.execute("DROP TABLE IF EXISTS ingestion_jobs")
    _create_ingestion_jobs(conn)


def _create_saved_artifacts(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved_artifacts (
            id                  TEXT PRIMARY KEY,
            notebook_id         TEXT NOT NULL,
            kind                TEXT NOT NULL,
            title               TEXT NOT NULL,
            content_json        TEXT NOT NULL,
            source_document_ids TEXT NOT NULL DEFAULT '[]',
            params_json         TEXT NOT NULL DEFAULT '{}',
            created_at          TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (notebook_id) REFERENCES notebooks(id)
        )
    """)


def _create_review_items(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS review_items (
            id                 TEXT PRIMARY KEY,
            notebook_id        TEXT NOT NULL,
            artifact_id        TEXT NOT NULL,
            item_type          TEXT NOT NULL,
            item_index         INTEGER NOT NULL,
            prompt             TEXT NOT NULL,
            answer             TEXT NOT NULL DEFAULT '',
            choices_json       TEXT NOT NULL DEFAULT '[]',
            explanation        TEXT NOT NULL DEFAULT '',
            source_document_id TEXT NOT NULL DEFAULT '',
            source_chunk_id    TEXT NOT NULL DEFAULT '',
            section_path       TEXT NOT NULL DEFAULT '',
            page_start         INTEGER NOT NULL DEFAULT -1,
            page_end           INTEGER NOT NULL DEFAULT -1,
            is_starred         INTEGER NOT NULL DEFAULT 0,
            is_mistake         INTEGER NOT NULL DEFAULT 0,
            last_user_answer   TEXT NOT NULL DEFAULT '',
            created_at         TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at         TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (notebook_id) REFERENCES notebooks(id),
            FOREIGN KEY (artifact_id) REFERENCES saved_artifacts(id) ON DELETE CASCADE,
            UNIQUE(notebook_id, artifact_id, item_type, item_index)
        )
    """)


def _create_saved_conversations(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved_conversations (
            id                  TEXT PRIMARY KEY,
            notebook_id         TEXT NOT NULL,
            title               TEXT NOT NULL DEFAULT '',
            messages_json       TEXT NOT NULL DEFAULT '[]',
            source_document_ids TEXT NOT NULL DEFAULT '[]',
            created_at          TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (notebook_id) REFERENCES notebooks(id)
        )
    """)


def _create_document_extractions(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS document_extractions (
            document_id       TEXT PRIMARY KEY,
            extraction_engine TEXT NOT NULL,
            full_text_path    TEXT NOT NULL,
            markdown_path     TEXT DEFAULT NULL,
            pages_path        TEXT DEFAULT NULL,
            metadata_json     TEXT NOT NULL DEFAULT '{}',
            warnings_json     TEXT NOT NULL DEFAULT '[]',
            created_at        TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (document_id) REFERENCES source_documents(id)
        )
    """)


def _dedupe_notebook_names(conn: sqlite3.Connection) -> None:
    """Merge duplicate notebook names into the surviving row before the unique index is added.

    For each name with multiple rows, keep the row with the most documents (ties broken
    by oldest created_at). Reparent documents, chunks, artifacts, and other notebook-scoped
    rows from the discarded duplicates onto the survivor, then delete the duplicates.
    """
    dup_names = [
        r[0] for r in conn.execute(
            "SELECT name FROM notebooks GROUP BY name HAVING COUNT(*) > 1"
        ).fetchall()
    ]
    for name in dup_names:
        rows = conn.execute(
            """
            SELECT n.id, n.created_at,
                   (SELECT COUNT(*) FROM source_documents s WHERE s.notebook_id = n.id) AS doc_count
            FROM notebooks n WHERE n.name = ?
            ORDER BY doc_count DESC, n.created_at ASC
            """,
            (name,),
        ).fetchall()
        survivor_id = rows[0][0]
        loser_ids = [r[0] for r in rows[1:]]
        for loser_id in loser_ids:
            for tbl in (
                "source_documents", "saved_artifacts", "review_items",
                "saved_conversations", "ingestion_jobs",
            ):
                try:
                    conn.execute(
                        f"UPDATE {tbl} SET notebook_id = ? WHERE notebook_id = ?",
                        (survivor_id, loser_id),
                    )
                except sqlite3.OperationalError:
                    pass
            conn.execute("DELETE FROM notebooks WHERE id = ?", (loser_id,))


def _create_indexes(conn: sqlite3.Connection) -> None:
    """Create indexes and unique constraints for query performance and data integrity.

    Uses IF NOT EXISTS so these are safe to call on existing databases.
    """
    # Prevent duplicate ingest of the same file into the same notebook.
    # Partial unique index: only enforces uniqueness when content_hash is non-empty.
    # This allows legacy rows with empty hashes to coexist without migration errors.
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_source_notebook_hash
        ON source_documents(notebook_id, content_hash)
        WHERE content_hash != ''
    """)

    # Fast lookup by notebook (list documents, search scope)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_source_notebook
        ON source_documents(notebook_id)
    """)

    # Prevent duplicate notebook names (concurrent get_or_create_default safety).
    # First, deduplicate any pre-existing rows so the unique index can be built.
    _dedupe_notebook_names(conn)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_notebook_name
        ON notebooks(name)
    """)

    # Fast lookup of chunks by document (hydration, section outline)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_document
        ON chunks(document_id)
    """)

    # Fast lookup by content hash (dedup, force_reingest)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_hash
        ON chunks(content_hash)
    """)

    # Fast lookup of saved artifacts by notebook
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_artifacts_notebook
        ON saved_artifacts(notebook_id, created_at DESC)
    """)

    # Fast lookup of saved review items by notebook and source artifact.
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_review_items_notebook
        ON review_items(notebook_id, updated_at DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_review_items_artifact
        ON review_items(artifact_id)
    """)

    # Fast lookup of saved conversations by notebook
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_notebook
        ON saved_conversations(notebook_id, created_at DESC)
    """)


def _set_version(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        )
    """)
    conn.execute(
        "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
        (SCHEMA_VERSION,),
    )


# ---------------------------------------------------------------------------
# Migration helpers — safe ALTER TABLE additions
# ---------------------------------------------------------------------------


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    """Add a column if it doesn't already exist. Safe to call on existing DBs."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
    except sqlite3.OperationalError:
        # Column already exists — ignore
        pass


def _ensure_chunk_context_columns(conn: sqlite3.Connection) -> None:
    """Add context_prefix and context_mode columns to chunks if missing."""
    _ensure_column(conn, "chunks", "context_prefix", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "chunks", "context_mode", "TEXT NOT NULL DEFAULT 'metadata'")


def _drop_study_plans(conn: sqlite3.Connection) -> None:
    """Drop the removed study-plan feature's tables (idempotent).

    The active study plan feature was removed. Drop its tables from any DB that
    already created them (order matters — study_tasks references study_plans).
    """
    conn.execute("DROP TABLE IF EXISTS study_tasks")
    conn.execute("DROP TABLE IF EXISTS study_plans")


def _backfill_vec_notebook_ids(conn: sqlite3.Connection) -> None:
    """Repair chunks_vec rows indexed with an empty notebook_id (one-time).

    The 2026-09-21 migration from Qdrant to sqlite-vec left some rows with an
    empty notebook_id payload; dense search filters on notebook_id, so those
    rows were invisible to dense retrieval. One correlated UPDATE re-derives
    the value from the document (measured ~250 ms for 7.6k rows). Idempotent.
    Safe on connections without sqlite-vec loaded (OperationalError, ignored).
    """
    try:
        cur = conn.execute(
            "UPDATE chunks_vec SET notebook_id = COALESCE("
            "(SELECT notebook_id FROM source_documents "
            " WHERE source_documents.id = chunks_vec.document_id), '') "
            "WHERE notebook_id = ?",
            ("",),
        )
    except sqlite3.OperationalError:
        return  # vec table not present, or vec0 not loaded on this connection
    if cur.rowcount > 0:
        conn.commit()
        logger.info("Backfilled notebook_id for %d dense-index rows", cur.rowcount)


def run_migrations(conn: sqlite3.Connection) -> None:
    """Run any pending schema migrations."""
    _ensure_column(conn, "source_documents", "display_title", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "source_documents", "chunking_mode", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "source_documents", "structure_status", "TEXT NOT NULL DEFAULT 'unknown_legacy'")
    _ensure_column(conn, "source_documents", "embedded_toc_present", "INTEGER DEFAULT NULL")
    _ensure_column(conn, "source_documents", "synthetic_outline_mode", "TEXT DEFAULT NULL")
    _ensure_column(conn, "source_documents", "structure_confidence", "TEXT DEFAULT NULL")
    _ensure_column(conn, "source_documents", "structure_user_note", "TEXT DEFAULT NULL")
    _ensure_column(conn, "source_documents", "structure_llm_instruction", "TEXT DEFAULT NULL")
    _ensure_column(conn, "notebooks", "description", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "notebooks", "last_opened_at", "TEXT DEFAULT NULL")
    _create_document_extractions(conn)
    _ensure_chunk_context_columns(conn)
    _migrate_app_config_to_env(conn)
    _migrate_ingestion_jobs(conn)
    _drop_study_plans(conn)
    _backfill_vec_notebook_ids(conn)


# Legacy DB app_config keys → environment variable names. The app_config KV
# table was removed in favour of env-only config; this map drives the one-time
# migration of any still-stored values into .env before the table is dropped.
_APP_CONFIG_TO_ENV = {
    "llm_provider": "LLM_PROVIDER",
    "llm_api_format": "LLM_API_FORMAT",
    "llm_api_key": "LLM_API_KEY",
    "llm_base_url": "LLM_BASE_URL",
    "llm_model": "LLM_MODEL",
    "llm_remote_enabled": "LLM_REMOTE_ENABLED",
    "web_search_provider": "WEB_SEARCH_PROVIDER",
    "tavily_api_key": "TAVILY_API_KEY",
}


def _migrate_app_config_to_env(conn: sqlite3.Connection) -> None:
    """One-time migration: copy legacy app_config rows to .env, then drop the table.

    Config previously lived in the SQLite `app_config` KV table; it now lives in
    environment variables (.env). On first run against an existing database, copy
    any rows whose env var is not already set (env wins), then drop the table.
    Idempotent: fresh databases have no app_config table, so this is a no-op.
    """
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'app_config'"
    ).fetchone()
    if not exists:
        return

    try:
        rows = conn.execute("SELECT key, value FROM app_config").fetchall()
    except sqlite3.OperationalError:
        rows = []

    for row in rows:
        key, value = row[0], row[1]
        env_name = _APP_CONFIG_TO_ENV.get(key)
        if not env_name or os.getenv(env_name):
            continue  # unknown key, or env already set (env wins)
        try:
            set_env(env_name, value)
            logger.info("Migrated app_config '%s' → env %s", key, env_name)
        except Exception as exc:  # noqa: BLE001 — best-effort migration
            logger.warning("Failed to migrate app_config '%s': %s", key, exc)

    conn.execute("DROP TABLE IF EXISTS app_config")
    conn.commit()
