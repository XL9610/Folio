"""retrieve_chunks: hybrid dense + FTS retrieval with conditional reranking."""

from __future__ import annotations

import logging

from ....core.config import config
from ...retrieval.profile_selector import ProfileSelector
from ...retrieval.reranker import rerank_results
from ...retrieval.search_options import SearchOptions
from ..agent_planner import ToolCall
from .types import ToolResults

logger = logging.getLogger(__name__)


_RERANK_TASKS: set[str] = {"explain_concept", "compare", "specific_search"}

_profile_selector = ProfileSelector()


def _should_rerank(task: str, retrieval_scope: str = "unknown") -> bool:
    """Return True if the planner task benefits from LLM reranking."""
    if config.rerank_force_all:  # eval-only override (RERANK_FORCE_ALL)
        return True
    if retrieval_scope == "whole_source":
        return False
    return task in _RERANK_TASKS


class RetrieveTool:
    """retrieve_chunks tool."""

    def _tool_retrieve(
        self,
        results: ToolResults,
        tc: ToolCall,
        notebook_id: str,
        doc_ids: list[str],
        top_k: int,
    ) -> None:
        query = tc.args.get("query", "") or tc.args.get("search_query", "")
        if not query:
            results.trace.append({
                "name": "retrieve_chunks",
                "status": "error",
                "args": tc.args,
                "error": "No search query provided",
            })
            return

        k = min(tc.args.get("top_k", top_k), top_k)
        requested_ids = tc.args.get("document_ids", doc_ids)
        bounded_ids = [did for did in requested_ids if did in doc_ids] if requested_ids else doc_ids

        search_k = max(k, min(max(top_k * 2, 20), 30))
        search_opts = SearchOptions(
            profile_name=tc.args.get("_profile_name", "baseline"),
            rerank_enabled=tc.args.get("_rerank_enabled", False),
            rerank_top_n=tc.args.get("_rerank_top_n", 20),
            filters=tc.args.get("_search_filters"),
            boost_config=tc.args.get("_boost_config"),
            dense_enabled=not tc.args.get("_raw", False),
        )
        chunks, meta = self._retriever.search_with_meta(
            notebook_id=notebook_id,
            query=query,
            top_k=search_k,
            document_ids=bounded_ids,
            options=search_opts,
        )

        # Mark retrieval unavailable only when every backend that could have
        # answered actually errored. Keyword-only degradation (dense failed,
        # keyword succeeded) still allows an answer.
        dense_available = meta.get("dense_available", True)
        keyword_available = meta.get("keyword_backend") not in (None, "none")
        dense_errored = bool(meta.get("dense_error") or meta.get("embedding_error"))
        keyword_errored = bool(meta.get("keyword_error"))
        tried = 0
        failed = 0
        if dense_available:
            tried += 1
            if dense_errored:
                failed += 1
        if keyword_available:
            tried += 1
            if keyword_errored:
                failed += 1
        if tried == 0 or failed == tried:
            results.tool_error = True

        # Rerank: downrank exercises/epilogue, diversify sections
        # Skip when LLM reranker already re-ranked (llm_rerank_score present)
        if not any(hasattr(c, "llm_rerank_score") for c in chunks):
            chunks = rerank_results(
                chunks,
                diversify_by_section=True,
                query=query,
                boost_config=tc.args.get("_boost_config"),
            )

        results.chunks.extend(chunks)
        # Build citations
        seen = set()
        for chunk in chunks[:8]:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            snippet = chunk.text[:120].replace("\n", " ").strip()
            if len(chunk.text) > 120:
                snippet += "\u2026"
            ref_label = self._next_ref_label(results, "book")
            results.chunk_citations.append({
                "source_type": "book",
                "ref_label": ref_label,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text_snippet": snippet,
                "section_path": chunk.section_path or "",
                "title": chunk.title or "",
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            })

        trace_entry: dict = {
            "name": "retrieve_chunks",
            "status": "ok",
            "args": {
                "query": query,
                "document_ids": bounded_ids,
                "top_k": k,
                "search_top_k": search_k,
                "results": len(chunks),
                "dense_available": meta.get("dense_available", True),
                "keyword_backend": meta.get("keyword_backend", "none"),
                "retrieval_latency_ms": meta.get("total_latency_ms", 0),
                "warnings": meta.get("warnings", []),
                "original_query": tc.args.get("_original_query", ""),
                "planner_query": tc.args.get("_planner_query", ""),
                "expanded_query": tc.args.get("_expanded_query", ""),
                "injected_terms": tc.args.get("_injected_terms", ""),
            },
        }
        if any(hasattr(c, "llm_rerank_score") for c in chunks):
            trace_entry["args"]["reranked"] = True
            trace_entry["args"]["rerank_scores"] = [
                round(getattr(c, "llm_rerank_score", 0.0), 1) for c in chunks[:5]
            ]
        results.trace.append(trace_entry)
