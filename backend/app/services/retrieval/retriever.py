"""Hybrid retriever — dense + keyword (BM25x or improved FTS5) with RRF fusion."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from ...repositories import chunk_repository
from ..embedding.base import Embedder
from ..index.base import DenseIndexStore
from .base import RetrievedChunk
from .keyword_backend import KeywordBackend, get_keyword_backend
from .retrieval_profile import RetrievalProfile, load_profile
from .search_options import SearchOptions

if TYPE_CHECKING:
    from .llm_reranker import LLMReranker

logger = logging.getLogger(__name__)

# RRF constant (standard from research papers)
RRF_K = 60


class HybridRetriever:
    """Combines dense and keyword retrieval using Reciprocal Rank Fusion.

    Usage:
        retriever = HybridRetriever(embedder, dense_store)
        results = retriever.search(notebook_id="...", query="...", top_k=10)
    """

    def __init__(
        self,
        embedder: Embedder,
        dense_store: DenseIndexStore | None = None,
        reranker: LLMReranker | None = None,
        rerank_top_n: int = 20,
        keyword_backend: KeywordBackend | None = None,
        profile: RetrievalProfile | None = None,
    ):
        self._embedder = embedder
        self._dense = dense_store
        self._reranker = reranker
        self._rerank_top_n = rerank_top_n
        self._keyword = keyword_backend or get_keyword_backend()
        self._profile = profile or load_profile()
        self._dense_healthy: bool | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def search(
        self,
        notebook_id: str,
        query: str,
        top_k: int = 10,
        document_ids: list[str] | None = None,
        options: SearchOptions | None = None,
    ) -> list[RetrievedChunk]:
        """Run hybrid search: dense + keyword, fused with RRF.

        All results are hydrated from SQLite after fusion so text
        is always present regardless of which backend provided the hit.
        """
        chunks, _meta = self.search_with_meta(notebook_id, query, top_k, document_ids, options=options)
        return chunks

    def search_with_meta(
        self,
        notebook_id: str,
        query: str,
        top_k: int = 10,
        document_ids: list[str] | None = None,
        options: SearchOptions | None = None,
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        """Run hybrid search and return results with diagnostic metadata.

        Returns:
            (chunks, meta) where meta contains:
              - dense_available: bool
              - keyword_backend: str
              - warnings: list[str]
              - dense_latency_ms: float
              - keyword_latency_ms: float
              - fusion_latency_ms: float
              - total_latency_ms: float
        """
        meta: dict[str, Any] = {
            "dense_available": True,
            "keyword_backend": self._keyword.name() if self._keyword else "none",
            "warnings": [],
            "dense_latency_ms": 0,
            "keyword_latency_ms": 0,
            "fusion_latency_ms": 0,
            "total_latency_ms": 0,
        }

        t_total_start = time.monotonic()

        dense_enabled = options is None or options.dense_enabled
        keyword_enabled = options is None or options.keyword_enabled

        # Check dense health
        self._dense_healthy = self._check_dense_health() if dense_enabled else None
        if dense_enabled and not self._dense_healthy:
            meta["dense_available"] = False
            meta["warnings"].append("Dense search unavailable — using keyword-only search")

        # Resolve per-query profile override
        profile = self._profile
        if options is not None and options.profile_name:
            profile = load_profile(options.profile_name)

        # Try embedding for dense (skipped entirely when dense is disabled)
        query_vector: list[float] | None = None
        embed_error: str | None = None
        if dense_enabled:
            query_vector, embed_error = self._embed_query_safe(query)
            if embed_error:
                meta["embedding_error"] = embed_error
                meta["dense_available"] = False
                meta["warnings"].append("Query embedding failed — dense search skipped")

        # Run both backends — fetch more when reranker needs a wider window
        fetch_k = top_k * max(profile.dense_fetch_k_mult, profile.keyword_fetch_k_mult)
        rfuse_k = top_k
        if self._reranker is not None:
            fetch_k = max(fetch_k, min(self._rerank_top_n * 2, 50))
            rfuse_k = max(top_k, min(self._rerank_top_n, fetch_k))

        if dense_enabled:
            t_dense_start = time.monotonic()
            dense_results, dense_error = self._dense_search(query_vector, notebook_id, fetch_k, document_ids)
            meta["dense_latency_ms"] = round((time.monotonic() - t_dense_start) * 1000, 2)
            if dense_error:
                meta["dense_error"] = dense_error
                meta["warnings"].append("Dense search failed")
        else:
            dense_results = []
            meta["dense_disabled"] = True

        if keyword_enabled:
            t_fts_start = time.monotonic()
            fts_results, fts_error = self._fts_search(notebook_id, query, fetch_k, document_ids)
            meta["keyword_latency_ms"] = round((time.monotonic() - t_fts_start) * 1000, 2)
            if fts_error:
                meta["keyword_error"] = fts_error
                meta["warnings"].append("Keyword search failed")
        else:
            fts_results = []
            meta["keyword_disabled"] = True

        # Post-filter results when scoped retrieval is active (Module E)
        if options is not None and options.filters is not None and options.filters.has_any_filter():
            from ..retrieval.search_filters import normalize_page_range

            filters = options.filters
            if filters.page_start is not None and filters.page_end is not None:
                ps, pe = normalize_page_range(filters.page_start, filters.page_end)
                filters_doc_ids = set(filters.document_ids) if filters.document_ids else None
                dense_results = [
                    c for c in dense_results
                    if (filters_doc_ids is None or c.get("document_id", "") in filters_doc_ids)
                    and ps is not None and c.get("page_start", -1) >= ps
                    and pe is not None and c.get("page_end", -1) <= pe
                ]
                fts_results = [
                    c for c in fts_results
                    if (filters_doc_ids is None or c.get("document_id", "") in filters_doc_ids)
                    and ps is not None and c.get("page_start", -1) >= ps
                    and pe is not None and c.get("page_end", -1) <= pe
                ]

        # RRF fusion
        t_fusion_start = time.monotonic()
        fused = self._rrf_fuse(dense_results, fts_results, rfuse_k, profile)
        meta["fusion_latency_ms"] = round((time.monotonic() - t_fusion_start) * 1000, 2)

        # Hydrate full text from SQLite for all results
        self._hydrate_from_db(fused)

        # Deduplicate by (title, page_start, page_end) — keep highest RRF score
        fused = self._dedupe_results(fused)

        # Optional LLM rerank pass — controlled by per-request SearchOptions
        rerank_enabled = (
            self._reranker is not None
            and options is not None
            and options.rerank_enabled
        )
        if rerank_enabled and len(fused) > 1:
            try:
                rerank_window = max(top_k, min(self._rerank_top_n, len(fused)))
                candidates = fused[:rerank_window]
                fused = self._reranker.rerank(query, candidates, top_n=rerank_window) + fused[rerank_window:]
                fused = fused[:top_k]
            except Exception as exc:
                logger.warning("Reranker failed, keeping RRF order: %s", exc)

        meta["total_latency_ms"] = round((time.monotonic() - t_total_start) * 1000, 2)
        return fused, meta

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _embed_query_safe(self, query: str) -> tuple[list[float] | None, str | None]:
        """Try to embed a query. Returns (vector, error) — error set on failure."""
        try:
            return self._embedder.embed_query(query), None
        except Exception as exc:
            logger.warning("Embedding failed, falling back to keyword-only: %s", exc)
            return None, str(exc)

    def _check_dense_health(self) -> bool:
        """Check if dense store is reachable. Returns True if healthy."""
        if self._dense is None:
            return False
        try:
            healthy, msg, _ms = self._dense.health_check(timeout=1.0)
            if not healthy:
                logger.warning("Dense store unhealthy: %s", msg)
            return healthy
        except Exception as exc:
            logger.warning("Dense health check failed: %s", exc)
            return False

    def _dense_search(
        self, query_vector: list[float] | None, notebook_id: str, top_k: int,
        document_ids: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if self._dense is None or query_vector is None:
            return [], None
        try:
            return self._dense.search(query_vector, notebook_id, top_k, document_ids), None
        except Exception as exc:
            logger.warning("Dense search failed: %s", exc)
            return [], str(exc)

    def _fts_search(
        self, notebook_id: str, query: str, top_k: int,
        document_ids: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Keyword search via the configured keyword backend (BM25x or improved FTS5)."""
        if not self._keyword:
            return [], None
        try:
            return self._keyword.search(query, top_k, notebook_id, document_ids), None
        except Exception as exc:
            logger.warning("Keyword search failed: %s", exc)
            return [], str(exc)

    def _rrf_fuse(
        self,
        dense: list[dict[str, Any]],
        fts: list[dict[str, Any]],
        top_k: int,
        profile: RetrievalProfile | None = None,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion — merge and re-rank two ranked lists.

        When a chunk appears in both lists, the FTS result (which has text)
        takes precedence for text fields, and RRF scores are summed.
        """
        profile = profile or self._profile
        merged: dict[str, dict[str, Any]] = {}  # chunk_id → merged info

        # First pass: dense results
        for rank, item in enumerate(dense):
            cid = item["chunk_id"]
            merged[cid] = {
                "chunk_id": cid,
                "document_id": item.get("document_id", ""),
                "title": item.get("title", ""),
                "section_path": item.get("section_path", ""),
                "page_start": item.get("page_start", -1),
                "page_end": item.get("page_end", -1),
                "text": item.get("text", ""),
                "dense_score": item.get("score", 0.0),
                "fts_score": 0.0,
                "dense_rank": rank + 1,
                "fts_rank": 1000,  # not present
            }

        # Second pass: FTS results — merge or add, preferring FTS text
        for rank, item in enumerate(fts):
            cid = item["chunk_id"]
            if cid in merged:
                # Same chunk: FTS always has real text — prefer it
                merged[cid]["text"] = item.get("text", merged[cid]["text"])
                merged[cid]["title"] = item.get("title") or merged[cid]["title"]
                merged[cid]["section_path"] = item.get("section_path") or merged[cid]["section_path"]
                merged[cid]["fts_score"] = item.get("score", 0.0)
                merged[cid]["fts_rank"] = rank + 1
            else:
                merged[cid] = {
                    "chunk_id": cid,
                    "document_id": item.get("document_id", ""),
                    "title": item.get("title", ""),
                    "section_path": item.get("section_path", ""),
                    "page_start": item.get("page_start", -1),
                    "page_end": item.get("page_end", -1),
                    "text": item.get("text", ""),
                    "dense_score": 0.0,
                    "fts_score": item.get("score", 0.0),
                    "dense_rank": 1000,
                    "fts_rank": rank + 1,
                }

        # Compute RRF scores
        rrf_results: list[RetrievedChunk] = []
        for cid, info in merged.items():
            dense_rrf = profile.dense_weight / (profile.rrf_k + info["dense_rank"])
            fts_rrf = profile.keyword_weight / (profile.rrf_k + info["fts_rank"])
            total = dense_rrf + fts_rrf

            rrf_results.append(RetrievedChunk(
                chunk_id=cid,
                document_id=info["document_id"],
                text=info.get("text", ""),
                title=info.get("title", ""),
                section_path=info.get("section_path", ""),
                page_start=info["page_start"],
                page_end=info["page_end"],
                dense_score=info["dense_score"],
                fts_score=info["fts_score"],
                rrf_score=total,
                metadata={
                    "dense_rank": info["dense_rank"] if info["dense_rank"] < 1000 else 0,
                    "keyword_rank": info["fts_rank"] if info["fts_rank"] < 1000 else 0,
                },
            ))

        # Apply section-depth weighting
        for chunk in rrf_results:
            depth_bonus = self._section_depth_bonus(chunk.section_path or "")
            chunk.rrf_score *= depth_bonus

        rrf_results.sort(key=lambda x: x.rrf_score, reverse=True)
        rrf_results = rrf_results[:top_k]

        # Write fusion rank into metadata
        for rank, chunk in enumerate(rrf_results, start=1):
            chunk.metadata["fusion_rank"] = rank

        return rrf_results

    def _hydrate_from_db(self, results: list[RetrievedChunk]) -> None:
        """Fill in missing text/title/page fields from SQLite for any chunk.

        Dense-only results come without text — this ensures all results
        have complete content from the canonical store.
        """
        for r in results:
            if r.text and r.title:
                continue  # already complete from FTS
            db_chunk = chunk_repository.get_chunk_by_id(r.chunk_id)
            if db_chunk:
                if not r.text:
                    r.text = db_chunk["text"]
                if not r.title:
                    r.title = db_chunk["title"]
                if not r.section_path:
                    r.section_path = db_chunk["section_path"]
                if r.page_start <= 0:
                    r.page_start = db_chunk["page_start"]
                if r.page_end <= 0:
                    r.page_end = db_chunk["page_end"]
                if not r.document_id:
                    r.document_id = db_chunk["document_id"]

    @staticmethod
    def _dedupe_results(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove duplicate chunks (same document_id + title + page_start + page_end).

        When the same chunk appears from multiple retrieval backends
        (dense + FTS), keep only the highest-scoring occurrence.

        Note: chunks from different documents are NOT deduplicated,
        even if they share the same title/page range — this is intentional
        to preserve cross-document comparison results.

        For page-less sources (EPUB, plain text) where page_start/page_end
        are both ≤ 0, all chunks from the same document would share the same
        dedup key — collapsing an entire book into a single chunk. We append
        the chunk's text prefix to disambiguate these cases.
        """
        seen: dict[tuple, RetrievedChunk] = {}
        for r in results:
            key_parts = [r.document_id, r.title, r.page_start, r.page_end]
            if r.page_start <= 0 and r.page_end <= 0:
                # Page-less sources: disambiguate chunks via text prefix
                text_sig = r.text[:64] if r.text else r.chunk_id
                key_parts.append(text_sig)
            key = tuple(key_parts)
            if key not in seen or r.rrf_score > seen[key].rrf_score:
                seen[key] = r
        deduped = list(seen.values())
        deduped.sort(key=lambda x: x.rrf_score, reverse=True)
        return deduped

    @staticmethod
    def _section_depth_bonus(section_path: str) -> float:
        """Compute section-depth weighting bonus for RRF fusion.

        Returns:
            1.15 for key sections (Key Concepts, Summary, Definition, Overview, Introduction)
            1.10 if section depth <= 2 (shallow, close to chapter root)
            1.00 otherwise
        """
        if not section_path:
            return 1.0

        key_sections = frozenset({"key concepts", "summary", "definition", "overview", "introduction"})

        section_lower = section_path.lower()
        if any(kw in section_lower for kw in key_sections):
            return 1.15

        depth = section_path.count(">") + 1
        if depth <= 2:
            return 1.10

        return 1.0

