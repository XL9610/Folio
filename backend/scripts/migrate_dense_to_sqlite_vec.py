"""One-time migration: re-embed existing chunks into the sqlite-vec dense index.

Reads every chunk from the SQLite ``chunks`` table, reconstructs its index-time
text (context prefix + chunk text), re-embeds it with the current
sentence-transformer model, and upserts the vectors into the ``chunks_vec``
vec0 table via SqliteVecStore. Preserves every chunk_id UUID, so saved study
state (artifacts, review items) and the eval golden set stay valid.

Run once after switching from Qdrant to sqlite-vec:

    python backend/scripts/migrate_dense_to_sqlite_vec.py

Idempotent: refuses to run if chunks_vec already has vectors unless --force.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.db.session import get_connection  # noqa: E402
from app.services.embedding.sentence_transformer_embedder import (  # noqa: E402
    SentenceTransformerEmbedder,
)
from app.services.index.sqlite_vec_store import SqliteVecStore, VEC_TABLE  # noqa: E402
from app.services.ingestion.index_text import compose_indexed_text  # noqa: E402

_SOURCE_TYPE_BY_EXT = {
    ".pdf": "pdf",
    ".epub": "epub",
    ".pptx": "pptx",
    ".docx": "docx",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "txt",
}


def _source_type_for(filename: str) -> str:
    return _SOURCE_TYPE_BY_EXT.get(Path(filename).suffix.lower(), "")


def _count_existing(conn) -> int:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (VEC_TABLE,),
    ).fetchone()
    if row is None:
        return 0
    return conn.execute(f"SELECT COUNT(*) FROM {VEC_TABLE}").fetchone()[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--limit", type=int, default=0, help="Cap chunks (0 = all)")
    parser.add_argument("--force", action="store_true", help="Rebuild even if vectors exist")
    args = parser.parse_args()

    conn = get_connection()
    existing = _count_existing(conn)
    if existing and not args.force:
        print(
            f"{VEC_TABLE} already has {existing} vectors; nothing to do. "
            "Re-run with --force to rebuild."
        )
        return 0

    sql = """
        SELECT c.id AS chunk_id, c.document_id, d.notebook_id, c.title,
               c.section_path, c.page_start, c.page_end, c.chunk_strategy,
               c.text, c.context_prefix, d.filename
        FROM chunks c
        JOIN source_documents d ON c.document_id = d.id
        ORDER BY c.document_id, c.chunk_index
    """
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"
    rows = conn.execute(sql).fetchall()

    if not rows:
        print("No chunks found to migrate.")
        return 0

    store = SqliteVecStore()
    embedder = SentenceTransformerEmbedder()
    total = len(rows)
    print(f"Migrating {total} chunks into {VEC_TABLE} ...")

    ensured = False
    migrated = 0
    for start in range(0, total, args.batch):
        batch = rows[start : start + args.batch]
        texts = [compose_indexed_text(r["context_prefix"], r["text"]) for r in batch]
        vectors = embedder.embed_texts(texts)
        if not ensured:
            store.ensure_collection(len(vectors[0]))
            ensured = True
        payloads = [
            {
                "chunk_id": r["chunk_id"],
                "document_id": r["document_id"],
                "notebook_id": r["notebook_id"],
                "title": r["title"],
                "section_path": r["section_path"],
                "page_start": r["page_start"],
                "page_end": r["page_end"],
                "chunk_strategy": r["chunk_strategy"],
                "source_type": _source_type_for(r["filename"]),
            }
            for r in batch
        ]
        store.upsert_chunks(payloads, vectors)
        migrated += len(batch)
        print(f"  {migrated}/{total} chunks indexed")

    print(f"Done: {migrated} chunks migrated into {VEC_TABLE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
