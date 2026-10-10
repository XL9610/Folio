"""QAToolExecutor — executes white-listed tools for the agentic QA pipeline.

Tools (registry):
- get_source_metadata: Returns document filename, page count, chunk count.
- get_source_outline: Returns section structure (section_path, page_start, page_end).
- retrieve_chunks: Hybrid dense + FTS retrieval with reranking.
- find_sections: Locate sections matching a natural-language hint.
- get_section_chunks: Retrieve chunks belonging to specific sections.

All tools enforce document_ids boundary and notebook scope.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ...repositories.chunk_repository import (
    get_chunks_by_document,
    get_chunks_by_section_prefix,
    get_chunks_for_section_paths,
    get_document_sections,
)
from ...repositories.document_repository import get_document
from ..pdf_ingestion.structure_metadata import (
    document_structure_fields,
    is_unreliable_structure_status,
)
from ..retrieval.base import RetrievedChunk
from ..retrieval.boost_config import BoostConfig
from ..retrieval.profile_selector import ProfileSelector
from ..retrieval.query_context import build_query_context
from ..retrieval.query_expander import QueryExpander
from ..retrieval.reranker import rerank_results
from ..retrieval.retriever import HybridRetriever
from ..retrieval.search_filters import SearchFilters
from ..retrieval.search_options import SearchOptions
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..web_search import DisabledWebSearchProvider, WebSearchProvider, fetch_page_content
from .agent_planner import PlanResult, ToolCall
from .citations import (
    MAX_BOOK_CITATIONS,
    MAX_WEB_CITATIONS,
    _dedupe_chunks as _citations_dedupe_chunks,
    _finalize_chunk_citations as _citations_finalize_chunk_citations,
    _next_ref_label as _citations_next_ref_label,
    finalize_merged_citations,
)
from .section_labels import (
    extract_numbered_section_label,
    extract_numbered_section_prefix,
    is_page_window_section_label,
    resolve_chapter_page_range,
    section_label_for_display,  # re-export (kept importable from tools)
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class SectionMatch:
    """A single section candidate matched by find_sections."""
    document_id: str
    section_path: str
    title: str
    page_start: int = -1
    page_end: int = -1
    chunk_count: int = 0
    rank: int = 0  # lower = better match
    match_type: str = "substring"  # "exact_chapter" | "substring" | "fuzzy"


_RERANK_TASKS: set[str] = {"explain_concept", "compare", "specific_search"}

_profile_selector = ProfileSelector()


def _should_rerank(task: str) -> bool:
    """Return True if the planner task benefits from LLM reranking."""
    return task in _RERANK_TASKS


@dataclass
class ToolResults:
    metadata: list[dict] = field(default_factory=list)
    outline: list[dict] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    chunk_citations: list[dict] = field(default_factory=list)
    web_results: list[dict] = field(default_factory=list)
    # Section-locating results
    section_matches: list[SectionMatch] = field(default_factory=list)
    # Information about what was targeted
    target_section: str = ""  # e.g. "Chapter 1"
    trace: list[dict] = field(default_factory=list)
    tool_error: bool = False  # True when a tool raised (retrieval unavailable, not "no hit")


# ---------------------------------------------------------------------------
# Tool registry type
# ---------------------------------------------------------------------------

ToolFn = Callable[..., None]  # mutates ToolResults + appends trace


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class QAToolExecutor:
    """Execute white-listed tools with boundary enforcement."""

    def __init__(
        self,
        retriever: HybridRetriever,
        web_search_provider: WebSearchProvider | None = None,
    ):
        self._retriever = retriever
        self._web_search = web_search_provider or DisabledWebSearchProvider()
        # Tool registry maps tool name → (handler, max_calls_per_round)
        self._registry: dict[str, ToolFn] = {
            "get_source_metadata": self._tool_metadata,
            "get_source_outline": self._tool_outline,
            "retrieve_chunks": self._tool_retrieve,
            "find_sections": self._tool_find_sections,
            "get_section_chunks": self._tool_get_section_chunks,
            "web_search": self._tool_web_search,
            "get_source_chunks": self._tool_get_source_chunks,
        }

    # ── public validate ────────────────────────────────────────────
    def is_whitelisted(self, tool_name: str) -> bool:
        return tool_name in self._registry

    def validate_tool_calls(
        self,
        tool_calls: list[ToolCall],
        document_ids: list[str],
        allow_web: bool = False,
    ) -> list[ToolCall]:
        """Filter and bound tool calls: whitelist + document boundary."""
        validated: list[ToolCall] = []
        for tc in tool_calls:
            if not self.is_whitelisted(tc.name):
                logger.warning("Tool '%s' not whitelisted, skipping", tc.name)
                continue
            args = dict(tc.args)
            if tc.name == "web_search":
                if not allow_web:
                    logger.info("Tool 'web_search' requested while disabled, skipping")
                    continue
                validated.append(ToolCall(name=tc.name, args=args))
                continue
            if "document_ids" in args and isinstance(args["document_ids"], list):
                args["document_ids"] = [
                    did for did in args["document_ids"] if did in document_ids
                ]
            if not args.get("document_ids"):
                args["document_ids"] = document_ids[:5]
            validated.append(ToolCall(name=tc.name, args=args))
        return validated

    # ── public execute ─────────────────────────────────────────────
    def execute(
        self,
        plan: PlanResult,
        notebook_id: str,
        document_ids: list[str],
        top_k: int = 10,
        allow_web: bool = False,
        user_query: str = "",
    ) -> ToolResults:
        """Execute all tool calls in the plan.

        Returns ToolResults with metadata, outline, chunks, citations, and trace.
        """
        results = ToolResults()
        doc_ids = document_ids[:10]  # cap for safety

        # Validate before execution
        validated = self.validate_tool_calls(plan.tool_calls, doc_ids, allow_web=allow_web)

        expander = QueryExpander()

        for tc in validated:
            # ── Query expansion for retrieve_chunks ──
            if tc.name == "retrieve_chunks":
                original_query = tc.args.get("query", "")
                planner_query = tc.args.get("original_query", "") or original_query
                sub_questions = [sq.question for sq in (plan.sub_questions or [])]

                # Gather document metadata (shared by profile selector + query context)
                doc_metadata = []
                for did in doc_ids[:5]:
                    doc = get_document(did)
                    if doc:
                        doc_metadata.append(document_structure_fields(doc))

                # Build QueryContext
                query_context = build_query_context(
                    user_query=user_query or original_query,
                    planner_task=plan.task,
                    document_metadata=doc_metadata,
                )
                section_hints = list(query_context.resolved_titles) if query_context.resolved_titles else []

                expanded = expander.expand(
                    user_query=user_query or original_query,
                    planner_query=planner_query,
                    plan_task=plan.task,
                    sub_questions=sub_questions,
                    section_hints=section_hints,
                )
                tc.args["query"] = expanded.expanded_query
                # Store expansion data for trace
                tc.args["_original_query"] = original_query
                tc.args["_planner_query"] = planner_query
                tc.args["_expanded_query"] = expanded.expanded_query
                tc.args["_injected_terms"] = expanded.injected_terms
                # Conditional reranker: per-request SearchOptions
                tc.args["_rerank_enabled"] = _should_rerank(plan.task)
                tc.args["_rerank_top_n"] = 20
                # Profile selection
                profile_name = _profile_selector.select(
                    task=plan.task,
                    query=user_query or original_query,
                    document_metadata=doc_metadata,
                )
                tc.args["_profile_name"] = profile_name
                # Store QueryContext fields for trace and BoostConfig (Modules C, F)
                tc.args["_has_front_matter"] = query_context.has_front_matter
                tc.args["_is_low_structure"] = query_context.is_low_structure
                tc.args["_has_chapter_ref"] = query_context.has_chapter_ref
                tc.args["_concept_phrase"] = query_context.concept_phrase
                # Build BoostConfig (Modules C, F)
                lead_threshold = 0
                if query_context.is_low_structure:
                    for doc_meta in doc_metadata:
                        page_count = doc_meta.get("page_count", 0)
                        if page_count > 0:
                            lead_threshold = max(lead_threshold, int(page_count * 0.15))
                tc.args["_boost_config"] = BoostConfig(
                    front_matter_boost=query_context.has_front_matter,
                    lead_paragraph_boost=query_context.is_low_structure,
                    lead_page_threshold=lead_threshold,
                )
                # Chapter resolution for scoped retrieval (Module D)
                tc.args["_search_filters"] = None
                if query_context.has_chapter_ref:
                    for did in doc_ids:
                        sections = get_document_sections(did)
                        if not sections:
                            continue
                        for ch_num in query_context.chapter_numbers:
                            page_range = resolve_chapter_page_range(ch_num, sections)
                            if page_range is not None:
                                tc.args["_search_filters"] = SearchFilters(
                                    document_ids=(did,),
                                    page_start=page_range[0],
                                    page_end=page_range[1],
                                )
                                section_hints = list(query_context.resolved_titles) if query_context.resolved_titles else section_hints
                                break
                        if tc.args["_search_filters"] is not None:
                            break

            try:
                handler = self._registry.get(tc.name)
                if handler is None:
                    results.trace.append({
                        "name": tc.name,
                        "status": "error",
                        "error": f"Unknown tool: {tc.name}",
                    })
                    continue
                handler(results, tc, notebook_id, doc_ids, top_k)
            except Exception as exc:
                logger.warning("Tool %s failed: %s", tc.name, exc)
                results.tool_error = True
                results.trace.append({
                    "name": tc.name,
                    "status": "error",
                    "args": tc.args,
                    "error": str(exc),
                })

        self._finalize_chunk_citations(results)
        return results

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _next_ref_label(results: ToolResults, source_type: str) -> str:
        return _citations_next_ref_label(results, source_type)

    @staticmethod
    def _dedupe_chunks(results: ToolResults) -> None:
        _citations_dedupe_chunks(results)

    @staticmethod
    def _finalize_chunk_citations(results: ToolResults) -> None:
        _citations_finalize_chunk_citations(results)

    def _tool_metadata(
        self,
        results: ToolResults,
        tc: ToolCall,
        notebook_id: str,
        doc_ids: list[str],
        top_k: int = 0,
    ) -> None:
        requested = tc.args.get("document_ids", doc_ids)[:5]
        for did in requested:
            if did not in doc_ids:
                continue
            doc = get_document(did)
            if doc and doc.get("notebook_id") == notebook_id:
                results.metadata.append({
                    "document_id": doc["id"],
                    "filename": doc["filename"],
                    "page_count": doc["page_count"],
                    "chunk_count": doc["chunk_count"],
                    "status": doc["status"],
                    **document_structure_fields(doc),
                })
        results.trace.append({
            "name": "get_source_metadata",
            "status": "ok",
            "args": {"document_ids": requested, "results": len(results.metadata)},
        })

    def _tool_outline(
        self,
        results: ToolResults,
        tc: ToolCall,
        notebook_id: str,
        doc_ids: list[str],
        top_k: int = 0,
    ) -> None:
        requested = tc.args.get("document_ids", doc_ids)[:5]
        for did in requested:
            if did not in doc_ids:
                continue
            doc = get_document(did)
            if not doc or doc.get("notebook_id") != notebook_id:
                continue
            sections = get_document_sections(did) or []
            structure = document_structure_fields(doc)
            results.outline.append({
                "document_id": did,
                "filename": doc["filename"],
                "page_count": doc["page_count"],
                **structure,
                "sections": [
                    {
                        "section_path": s.get("section_path", ""),
                        "title": s.get("title", ""),
                        "page_start": s.get("page_start", -1),
                        "page_end": s.get("page_end", -1),
                        "chunk_count": s.get("chunk_count", 0),
                    }
                    for s in sections
                ],
            })
        results.trace.append({
            "name": "get_source_outline",
            "status": "ok",
            "args": {"document_ids": requested, "results": len(results.outline)},
        })

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

    def _tool_get_source_chunks(
        self,
        results: ToolResults,
        tc: ToolCall,
        notebook_id: str,
        doc_ids: list[str],
        top_k: int = 0,
    ) -> None:
        """Retrieve representative body chunks across the document (overview/summary)."""
        from ..generation.sampling import sample_body_chunks

        requested = tc.args.get("document_ids", doc_ids)[:5]
        bounded = [did for did in requested if did in doc_ids]
        limit = max(1, min(int(tc.args.get("limit", 20) or 20), 50))

        collected: list[RetrievedChunk] = []
        for did in bounded:
            doc = get_document(did)
            if not doc or doc.get("notebook_id") != notebook_id:
                continue
            for rec in get_chunks_by_document(did):
                collected.append(RetrievedChunk(
                    chunk_id=rec["id"],
                    document_id=rec["document_id"],
                    text=rec["text"],
                    title=rec.get("title", ""),
                    section_path=rec.get("section_path", ""),
                    page_start=rec.get("page_start", -1),
                    page_end=rec.get("page_end", -1),
                ))

        sampled = sample_body_chunks(collected, max_chunks=min(limit, MAX_BOOK_CITATIONS))

        seen = set()
        for chunk in sampled:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            snippet = chunk.text[:120].replace("\n", " ").strip()
            if len(chunk.text) > 120:
                snippet += "…"
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
        # Overview samples are the primary evidence for whole-document queries;
        # the generator reads only the first 10 book chunks, and retrieve_chunks
        # (which ran earlier) could crowd them out. Prepend, not append.
        results.chunks = list(sampled) + results.chunks

        results.trace.append({
            "name": "get_source_chunks",
            "status": "ok" if sampled else "no_match",
            "args": {"document_ids": bounded, "limit": limit, "results": len(sampled)},
        })

    def _tool_web_search(
        self,
        results: ToolResults,
        tc: ToolCall,
        notebook_id: str,
        doc_ids: list[str],
        top_k: int = 0,
    ) -> None:
        query = (tc.args.get("query", "") or tc.args.get("search_query", "")).strip()
        max_results = max(1, min(int(tc.args.get("max_results", 5) or 5), 8))

        if not query:
            results.trace.append({
                "name": "web_search",
                "status": "error",
                "args": tc.args,
                "error": "No search query provided",
            })
            return

        if not getattr(self._web_search, "enabled", False):
            results.trace.append({
                "name": "web_search",
                "status": "disabled",
                "args": {
                    "query": query,
                    "max_results": max_results,
                    "provider": getattr(self._web_search, "name", "disabled"),
                },
                "error": "Internet research is enabled for this ask, but no web search provider is configured.",
            })
            return

        web_items = self._web_search.search(query=query, max_results=max_results)

        # Fetch full page content for top 2 results (in parallel).
        fetch_urls = [item.url for item in web_items[:2] if item.url]
        fetched: dict[str, str] = {}
        if fetch_urls:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(fetch_page_content, url, 10.0): url
                    for url in fetch_urls
                }
                for future in as_completed(futures, timeout=22.0):
                    url = futures[future]
                    try:
                        text = future.result()
                        if text:
                            fetched[url] = text
                    except Exception:
                        pass

        for item in web_items:
            ref_label = self._next_ref_label(results, "web")
            content = item.content[:1800].strip()
            snippet = content[:160].replace("\n", " ").strip()
            if len(content) > 160:
                snippet += "\u2026"
            result = {
                "source_type": "web",
                "ref_label": ref_label,
                "title": item.title,
                "url": item.url,
                "site_name": item.site_name,
                "published_date": item.published_date,
                "retrieved_at": item.retrieved_at,
                "score": item.score,
                "content": content,
                "fetched_content": fetched.get(item.url, ""),
                "text_snippet": snippet,
            }
            results.web_results.append(result)
            results.chunk_citations.append({
                "source_type": "web",
                "ref_label": ref_label,
                "chunk_id": f"web:{ref_label}",
                "document_id": "",
                "text_snippet": snippet,
                "section_path": item.site_name,
                "page_start": -1,
                "page_end": -1,
                "url": item.url,
                "title": item.title,
                "site_name": item.site_name,
                "published_date": item.published_date,
                "retrieved_at": item.retrieved_at,
            })

        results.trace.append({
            "name": "web_search",
            "status": "ok" if web_items else "no_match",
            "args": {
                "query": query,
                "max_results": max_results,
                "results": len(web_items),
                "provider": getattr(self._web_search, "name", "unknown"),
            },
        })

    def _tool_find_sections(
        self,
        results: ToolResults,
        tc: ToolCall,
        notebook_id: str,
        doc_ids: list[str],
        top_k: int = 0,
    ) -> None:
        """Find sections matching a natural-language hint like 'Chapter 1' or 'graphs'.

        Strategy (deterministic + conservative):
        1. Exact chapter-number match: "Chapter 1" "Chapter 10" "chapter 3"
        2. Substring match on section_path / title
        3. Fallback: return all top-level sections as candidates

        Returns ordered SectionMatch list in results.section_matches.
        """
        section_query = (
            tc.args.get("section_query", "")
            or tc.args.get("query", "")
            or tc.args.get("chapter", "")
        ).strip()
        if not section_query:
            results.trace.append({
                "name": "find_sections",
                "status": "error",
                "error": "No section_query or chapter hint provided",
            })
            return

        requested = tc.args.get("document_ids", doc_ids)[:5]
        bounded = [did for did in requested if did in doc_ids]

        # Store the target for later use (generator can reference it)
        results.target_section = section_query

        # Detect explicit chapter number
        chapter_match = re.search(
            r"(?:chapter|ch|section|sec)\s*#?\s*(\d+)", section_query, re.IGNORECASE
        )
        chapter_num = int(chapter_match.group(1)) if chapter_match else None

        all_matches: list[SectionMatch] = []
        for did in bounded:
            doc = get_document(did)
            if not doc or doc.get("notebook_id") != notebook_id:
                continue
            sections = get_document_sections(did) or []
            structure = document_structure_fields(doc)
            structure_status = structure.get("structure_status")

            for s in sections:
                sp = s.get("section_path", "") or ""
                title = s.get("title", "") or ""
                page_window_label = is_page_window_section_label(sp, title)
                match_type = ""
                rank = 999

                # Page-window fallback labels are not chapters.  Never infer
                # "Chapter 20" from "Pages 20-27".
                if (
                    chapter_num is not None
                    and page_window_label
                    and is_unreliable_structure_status(structure_status)
                ):
                    continue

                # Level 1: exact chapter number match
                if chapter_num is not None:
                    numbered, label = extract_numbered_section_label(sp, title)
                    # Use segment-aware parsing so "Chapter 1" does NOT match
                    # "Chapter 10", while "PART > 20. Title" does match chapter 20.
                    if numbered == chapter_num:
                        match_type = "exact_chapter"
                        rank = 1

                # Level 2: substring match (case-insensitive)
                # Guard: if we detected a chapter number, skip sections whose chapter number differs
                if not match_type:
                    q_lower = section_query.lower()
                    sp_lower = sp.lower()
                    title_lower = title.lower()
                    if q_lower in sp_lower or q_lower in title_lower:
                        # Avoid substring false-positives for chapter numbers
                        # e.g. "Chapter 1" appearing inside "Chapter 10"
                        if chapter_num is not None:
                            other_num, _ = extract_numbered_section_label(sp, title)
                            if other_num is not None and other_num != chapter_num:
                                # Different chapter — do NOT match via substring
                                pass
                            else:
                                match_type = "substring"
                                rank = 2
                        else:
                            match_type = "substring"
                            rank = 2

                # Level 3: individual word match
                if not match_type:
                    # If chapter number was detected, skip fuzzy matching on sections
                    # with a different chapter number to avoid false positives
                    if chapter_num is not None:
                        other_num, _ = extract_numbered_section_label(sp, title)
                        if other_num is not None and other_num != chapter_num:
                            # different chapter — skip entirely
                            match_type = ""
                        else:
                            words = [w.lower() for w in section_query.split() if len(w) > 2]
                            hit_count = sum(
                                1 for w in words if w in sp.lower() or w in title.lower()
                            )
                            if hit_count >= 1:
                                match_type = "fuzzy"
                                rank = 10 - hit_count
                    else:
                        words = [w.lower() for w in section_query.split() if len(w) > 2]
                        hit_count = sum(
                            1 for w in words if w in sp.lower() or w in title.lower()
                        )
                        if hit_count >= 1:
                            match_type = "fuzzy"
                            rank = 10 - hit_count

                if match_type:
                    all_matches.append(SectionMatch(
                        document_id=did,
                        section_path=sp,
                        title=title,
                        page_start=s.get("page_start", -1),
                        page_end=s.get("page_end", -1),
                        chunk_count=s.get("chunk_count", 0),
                        rank=rank,
                        match_type=match_type,
                    ))

        # Sort: lower rank first, then by page_start
        all_matches.sort(key=lambda m: (m.rank, m.section_path.count(">"), m.page_start))

        # Truncate to reasonable count
        results.section_matches = all_matches[:15]

        status = "ok" if results.section_matches else "no_match"
        results.trace.append({
            "name": "find_sections",
            "status": status,
            "args": {
                "section_query": section_query,
                "document_ids": bounded,
                "matches": len(results.section_matches),
            },
        })

    def _tool_get_section_chunks(
        self,
        results: ToolResults,
        tc: ToolCall,
        notebook_id: str,
        doc_ids: list[str],
        top_k: int = 0,
    ) -> None:
        """Retrieve chunks for matched sections from find_sections or explicit section_paths.

        Args:
            section_paths: list of section_path strings (takes precedence)
            section_prefix: string prefix to match (used if section_paths not given)
            document_ids: optional document filter
            limit: max chunks to return (default 20)
        """
        section_paths = tc.args.get("section_paths", [])
        section_prefix = tc.args.get("section_prefix", "")
        section_prefixes: list[str] = []
        requested = tc.args.get("document_ids", doc_ids)[:5]
        bounded = [did for did in requested if did in doc_ids]
        limit = min(tc.args.get("limit", 20), 50)

        # If the planner gave us the target from find_sections, use section_matches
        if not section_paths and not section_prefix and results.section_matches:
            for match in results.section_matches:
                if match.match_type != "exact_chapter":
                    continue
                _number, prefix, _label = extract_numbered_section_prefix(
                    match.section_path,
                    match.title,
                )
                if prefix and prefix not in section_prefixes:
                    section_prefixes.append(prefix)
            if len(section_prefixes) == 1:
                section_prefix = section_prefixes[0]
            else:
                section_paths = [m.section_path for m in results.section_matches[:3]]

        if not section_paths and not section_prefix and not section_prefixes:
            fallback_query = (results.target_section or "").strip()
            if fallback_query and not re.search(r"\b(?:chapter|ch|section|sec)\s*#?\s*\d+\b", fallback_query, re.I):
                chunks = self._retriever.search(
                    notebook_id=notebook_id,
                    query=fallback_query,
                    top_k=min(max(top_k, 1), limit),
                    document_ids=bounded,
                )
                results.chunks.extend(chunks)
                for chunk in chunks[:8]:
                    snippet = chunk.text[:120].replace("\n", " ").strip()
                    if len(chunk.text) > 120:
                        snippet += "\u2026"
                    results.chunk_citations.append({
                        "source_type": "book",
                        "ref_label": "",
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "text_snippet": snippet,
                        "section_path": chunk.section_path or "",
                        "title": chunk.title or "",
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                    })
                results.trace.append({
                    "name": "get_section_chunks",
                    "status": "fallback_retrieve" if chunks else "no_match",
                    "args": {
                        "query": fallback_query,
                        "document_ids": bounded,
                        "results": len(chunks),
                    },
                })
                return
            results.trace.append({
                "name": "get_section_chunks",
                "status": "error",
                "error": "No section_paths or section_prefix provided, and no prior section_matches",
            })
            return

        all_records: list[dict] = []

        for did in bounded:
            doc = get_document(did)
            if not doc or doc.get("notebook_id") != notebook_id:
                continue

            if section_paths:
                records = get_chunks_for_section_paths(did, section_paths, limit)
            elif section_prefixes:
                records = []
                for prefix in section_prefixes[:3]:
                    records.extend(get_chunks_by_section_prefix(did, prefix, limit))
            else:
                records = get_chunks_by_section_prefix(did, section_prefix, limit)

            all_records.extend(records)

        # Sort by page_start, chunk_index
        all_records.sort(key=lambda r: (r.get("page_start", 0), r.get("chunk_index", 0)))

        # Convert to RetrievedChunk
        for rec in all_records:
            snippet = rec["text"][:120].replace("\n", " ").strip()
            if len(rec["text"]) > 120:
                snippet += "\u2026"
            results.chunks.append(RetrievedChunk(
                chunk_id=rec["id"],
                document_id=rec["document_id"],
                text=rec["text"],
                title=rec.get("title", ""),
                section_path=rec.get("section_path", ""),
                page_start=rec.get("page_start", -1),
                page_end=rec.get("page_end", -1),
            ))
            ref_label = self._next_ref_label(results, "book")
            results.chunk_citations.append({
                "source_type": "book",
                "ref_label": ref_label,
                "chunk_id": rec["id"],
                "document_id": rec["document_id"],
                "text_snippet": snippet,
                "section_path": rec.get("section_path", ""),
                "title": rec.get("title", ""),
                "page_start": rec.get("page_start", -1),
                "page_end": rec.get("page_end", -1),
            })

        status = "ok" if all_records else "no_match"
        results.trace.append({
            "name": "get_section_chunks",
            "status": status,
            "args": {
                "section_paths": section_paths[:3] if section_paths else None,
                "section_prefix": section_prefix or None,
                "section_prefixes": section_prefixes[:3] if section_prefixes else None,
                "document_ids": bounded,
                "limit": limit,
                "results": len(all_records),
            },
        })

