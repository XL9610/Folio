# 0004 — sqlite-vec as the dense vector index

Date: 2026-09-21

## Status

Accepted.

## Context

Dense retrieval required a running Qdrant instance (started via
`docker compose up -d qdrant`), adding a Docker prerequisite and a second
stateful store alongside SQLite. The desktop-packaging direction needs a
single-process, no-Docker runtime.

sqlite-vec stores vectors in a `vec0` virtual table inside the same SQLite
file that already holds `chunks` + FTS5, so dense search needs no separate
service. It is pre-1.0 but ships a prebuilt Windows x86_64 wheel and has a
small, stable-enough surface for this project's scale: a few thousand
384-dim vectors, where exact brute-force cosine is sufficient and ANN is
unnecessary.

## Decision

Replace Qdrant with sqlite-vec as the single dense index backend:

- `SqliteVecStore` (`services/index/sqlite_vec_store.py`) implements the
  `DenseIndexStore` protocol against a `chunks_vec` vec0 table.
- vec0's primary key is an implicit INTEGER rowid; the UUID `chunk_id` is a
  TEXT metadata column. Metadata columns are filterable, so `search` filters
  `notebook_id` and `document_id` directly in SQL — no rowid bridge is needed.
- `embedding float[384] distance_metric=cosine`; results are read as cosine
  `distance` and converted to similarity (`score = 1 - distance`) to match the
  prior Qdrant cosine semantics.
- Idempotent upsert is DELETE-by-`chunk_id` + INSERT (vec0 does not support
  `INSERT OR REPLACE`).
- The extension is loaded per connection in `db.session.get_connection()`.
- Pin `sqlite-vec==0.1.9` (exact, not a range) — pre-1.0 breaking changes are
  expected.

Qdrant is removed outright rather than kept behind a toggle, matching ADR 0001
("pick one keyword backend"). Existing vectors are migrated by re-embedding
chunks from SQLite (`backend/scripts/migrate_dense_to_sqlite_vec.py`), which
preserves every chunk_id UUID and therefore all saved study state and the eval
golden set.

## Consequences

- **No Docker / no Qdrant**: dense search is in-process; the `/health` dense
  component reports healthy whenever the SQLite DB is usable.
- **One database file** now holds metadata, FTS5, and vectors (`chunks`,
  `chunks_fts`, `chunks_vec`).
- **Retrieval accuracy is unchanged**: the same embeddings and cosine metric
  are used, and sqlite-vec brute-force is exact (Qdrant's HNSW is approximate).
  RRF fusion uses ranks, not raw scores, so only ordering changes could matter.
- **Concurrency**: vector writes now share the SQLite writer lock, so
  `PRAGMA busy_timeout` was added; concurrent ingestion threads serialize on
  the single writer as before.
- **Risk**: sqlite-vec is pre-1.0 and its shadow-table format may change across
  releases; the exact version pin plus the re-embed migration script bound that
  risk.
