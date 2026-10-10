# Runtime Data Directory

This directory holds all runtime data — **entirely gitignored** (except this README).

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `books/` | Place your PDF/EPUB textbooks here |
| `extraction_artifacts/` | Extracted text/images from ingestion (future) |
| `uploads/` | Temporary upload buffer (future) |
| `bm25x_data/` | BM25x keyword search indices (auto-created) |

## Files

| File | Purpose |
|------|---------|
| `rag_companion.db` | SQLite store — metadata, chunks, FTS5, and dense vectors (sqlite-vec; auto-created on first run) |

## First-time Setup

Copy your textbooks into `books/`:
```bash
cp /path/to/your/textbooks/*.pdf local-data/books/
```

Then start the backend — the database and indices are created automatically.
