"""sqlite-vec dense vector store — the DenseIndexStore implementation.

Stores vectors in the same SQLite database as chunks/FTS via a vec0 virtual
table, so dense search needs no separate service (no Docker/Qdrant). vec0's
primary key is an implicit INTEGER rowid; the UUID ``chunk_id`` is stored as a
TEXT metadata column. Cosine distance is converted to similarity
(``score = 1 - distance``) to match the prior Qdrant cosine semantics.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from typing import Any

import sqlite_vec

from ...db.session import get_connection

logger = logging.getLogger(__name__)

# The vec0 virtual table holding one row per chunk's embedding. Named to mirror
# the `chunks_fts` keyword table that already lives in the same SQLite file.
VEC_TABLE = "chunks_vec"

# Metadata columns, mirroring the Qdrant payload exactly so search()'s return
# contract is unchanged. These are plain typed vec0 metadata columns (not `+`
# auxiliary columns) because search must filter on notebook_id/document_id.
_COLUMNS = (
    ("chunk_id", "text"),
    ("document_id", "text"),
    ("notebook_id", "text"),
    ("title", "text"),
    ("section_path", "text"),
    ("page_start", "integer"),
    ("page_end", "integer"),
    ("chunk_strategy", "text"),
    ("source_type", "text"),
)

_PAYLOAD_FIELDS = [name for name, _ in _COLUMNS]

_PAYLOAD_DEFAULTS = {
    "chunk_id": "",
    "document_id": "",
    "notebook_id": "",
    "title": "",
    "section_path": "",
    "page_start": -1,
    "page_end": -1,
    "chunk_strategy": "unknown",
    "source_type": "",
}

_DIM_RE = re.compile(r"float\[(\d+)\]")


def _ddl(vector_size: int) -> str:
    cols = ",\n    ".join(
        [f"embedding float[{int(vector_size)}] distance_metric=cosine"]
        + [f"{name} {typ}" for name, typ in _COLUMNS]
    )
    return f"CREATE VIRTUAL TABLE {VEC_TABLE} USING vec0(\n    {cols}\n)"


def _dimension_from_ddl(ddl: str) -> int | None:
    match = _DIM_RE.search(ddl)
    return int(match.group(1)) if match else None


class SqliteVecStore:
    """sqlite-vec dense index, in-process and Docker-free."""

    def ensure_collection(self, vector_size: int) -> None:
        """Create the vec0 table, or verify its dimension matches the embedder."""
        conn = get_connection()
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (VEC_TABLE,),
        ).fetchone()
        if row is None:
            conn.execute(_ddl(vector_size))
            conn.commit()
            logger.info("Created %s vec0 table (dim=%d)", VEC_TABLE, vector_size)
            return

        existing = _dimension_from_ddl(row["sql"])
        if existing is not None and existing != int(vector_size):
            raise ValueError(
                f"{VEC_TABLE} was built for {existing}-dim vectors but the "
                f"embedder produces {vector_size}-dim. Re-ingest (or drop the "
                "vector table) after switching embedding models."
            )

    def upsert_chunks(
        self, chunks: list[dict[str, Any]], vectors: list[list[float]]
    ) -> None:
        """Insert or replace chunks' vectors, idempotently by chunk_id.

        vec0 does not support INSERT OR REPLACE, so each upsert is a
        DELETE-by-chunk_id + INSERT pair within a single transaction.
        """
        if not chunks:
            return

        conn = get_connection()
        placeholders = ", ".join("?" * len(_PAYLOAD_FIELDS))
        try:
            for chunk, vec in zip(chunks, vectors):
                conn.execute(
                    f"DELETE FROM {VEC_TABLE} WHERE chunk_id = ?",
                    (chunk["chunk_id"],),
                )
                conn.execute(
                    f"INSERT INTO {VEC_TABLE} "
                    f"(embedding, {', '.join(_PAYLOAD_FIELDS)}) "
                    f"VALUES (?, {placeholders})",
                    [sqlite_vec.serialize_float32(vec)]
                    + [chunk.get(f, _PAYLOAD_DEFAULTS[f]) for f in _PAYLOAD_FIELDS],
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        logger.debug("Upserted %d chunks into %s", len(chunks), VEC_TABLE)

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all vector rows for a document (via the document_id metadata column).

        The vec0 table is created lazily on first embed; when it does not exist
        there are no vectors to delete, so treat that as idempotent success.
        """
        conn = get_connection()
        try:
            conn.execute(
                f"DELETE FROM {VEC_TABLE} WHERE document_id = ?", (document_id,)
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            conn.rollback()
            if "no such table" not in str(exc).lower():
                raise
            return
        logger.info("Deleted dense vectors for document %s", document_id)

    def search(
        self,
        query_vector: list[float],
        notebook_id: str | None,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """KNN search, filtered by notebook_id and optional document_ids."""
        conn = get_connection()

        # Keep `embedding MATCH ? AND k = ?` together, then metadata filters —
        # the vec0 constraint planner recognizes the MATCH + k pair.
        conds = ["embedding MATCH ?", "k = ?"]
        params: list[Any] = [sqlite_vec.serialize_float32(query_vector), int(top_k)]
        if notebook_id:
            conds.append("notebook_id = ?")
            params.append(notebook_id)
        if document_ids:
            conds.append(f"document_id IN ({', '.join('?' * len(document_ids))})")
            params.extend(document_ids)

        sql = (
            f"SELECT {', '.join(_PAYLOAD_FIELDS)}, distance "
            f"FROM {VEC_TABLE} WHERE " + " AND ".join(conds)
        )
        rows = conn.execute(sql, params).fetchall()
        return [
            {**{f: row[f] for f in _PAYLOAD_FIELDS}, "score": 1.0 - row["distance"]}
            for row in rows
        ]

    def health_check(self, timeout: float = 2.0) -> tuple[bool, str, float]:
        """Report dense health. In-process, so healthy whenever the DB is usable."""
        try:
            conn = get_connection()
            t0 = time.monotonic()
            conn.execute("SELECT 1").fetchone()
            elapsed = (time.monotonic() - t0) * 1000
            return True, "ok", round(elapsed, 2)
        except Exception as exc:
            msg = str(exc) or type(exc).__name__
            return False, msg, 0.0
