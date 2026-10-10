# 1. Keyword backend default — ImprovedFTS5

- **Status:** Accepted
- **Date:** 2026-09-20

## Context

The retrieval layer exposes three keyword-search implementations in
`backend/app/services/retrieval/keyword_backend.py`: raw FTS5 (a legacy query
builder), ImprovedFTS5 (stemming + stopword removal + FTS5 prefix terms), and
BM25x (a wrapper over the external SIRA `bm25x` Rust package using Lucene
scoring). A single factory, `get_keyword_backend()`, selects the active
backend: BM25x when `config.bm25x_enabled` is true *and* the `bm25x` package
imports, else ImprovedFTS5.

The M3 task intended to run a head-to-head A/B and record a winner, but the
TREC comparison was never completed — only a Jaccard-overlap compare script
exists, and no winner was ever landed. Separately, BM25x has become
un-reproducible: it is absent from `requirements.txt`, its vendored Rust
source (`sira-main/`) was deleted in the Phase-0 cleanup, and the only build
script (`backend/scripts/build_bm25x.ps1`) points at that deleted source. As a
result, the effective backend differs silently between machines depending on
whether a stale BM25x wheel happens to be installed.

## Decision

Make **ImprovedFTS5** the default keyword backend by flipping the
`BM25X_ENABLED` default from `true` to `false`. BM25x remains available as an
explicit opt-in (`BM25X_ENABLED=true` + a locally installed wheel).

Delete the orphaned `backend/scripts/build_bm25x.ps1`, which can never succeed
without the deleted Rust source.

Do **not** delete `BM25xKeywordBackend` itself: it is already guarded by
`is_available()` and costs nothing when disabled.

## Consequences

- The keyword backend is now deterministic and reproducible across
  environments: ImprovedFTS5 is pure-Python, dependency-free, and fully
  covered by tests.
- Deployments that want BM25x must supply the wheel themselves and set
  `BM25X_ENABLED=true`; this is now an explicit, documented choice rather than
  an implicit environment-dependent one.
- A real A/B (with a populated corpus and quality scoring) can still be run
  later; this decision does not preclude re-enabling BM25x if evidence
  warrants it.
