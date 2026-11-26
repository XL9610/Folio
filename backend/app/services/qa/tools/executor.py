"""QAToolExecutor: validates and runs white-listed tools for the agentic QA pipeline."""

from __future__ import annotations

import logging

from ....core.config import config
from ....repositories.chunk_repository import (
    get_document_sections,
)
from ....repositories.document_repository import get_document
from ...pdf_ingestion.structure_metadata import (
    document_structure_fields,
)
from ...retrieval.base import RetrievedChunk
from ...retrieval.boost_config import BoostConfig
from ...retrieval.query_context import build_query_context
from ...retrieval.query_expander import QueryExpander
from ...retrieval.retriever import HybridRetriever
from ...retrieval.search_filters import SearchFilters
from ...web_search import DisabledWebSearchProvider, WebSearchProvider
from ..agent_planner import PlanResult, ToolCall
from ..citations import (
    _dedupe_chunks as _citations_dedupe_chunks,
)
from ..citations import (
    _finalize_chunk_citations as _citations_finalize_chunk_citations,
)
from ..citations import (
    _next_ref_label as _citations_next_ref_label,
)
from ..section_labels import (
    resolve_chapter_page_range,
)
from .retrieve import RetrieveTool, _profile_selector, _should_rerank
from .sections import SectionTools
from .sources import SourceTools
from .types import ToolFn, ToolResults
from .web import WebTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool registry type
# ---------------------------------------------------------------------------

# How many document IDs the planner prompt lists (agent_planner: document_ids[:5]).
_PLANNER_VISIBLE_DOCS = 5



class QAToolExecutor(SourceTools, RetrieveTool, SectionTools, WebTool):
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
            if tc.name == "retrieve_chunks" and len(document_ids) > _PLANNER_VISIBLE_DOCS:
                # The planner was only shown the first few IDs, so its subset
                # is blindness, not intent: search every selected source.
                args["document_ids"] = list(document_ids)
            if not args.get("document_ids"):
                args["document_ids"] = document_ids[:_PLANNER_VISIBLE_DOCS]
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
        validated = self.validate_tool_calls(plan.tool_calls, document_ids, allow_web=allow_web)

        expander = QueryExpander()

        for tc in validated:
            # ── Query expansion for retrieve_chunks ──
            # "_raw" (corrective retry) searches the literal query: no expansion.
            if tc.name == "retrieve_chunks" and not tc.args.get("_raw"):
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
                tc.args["_rerank_enabled"] = _should_rerank(
                    plan.task, plan.retrieval_scope
                )
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
                # Search is a cheap SQL filter: no source cap. Prompt-feeding
                # tools (outline, metadata, sections) keep the capped list.
                handler(
                    results, tc, notebook_id,
                    document_ids if tc.name == "retrieve_chunks" else doc_ids,
                    top_k,
                )
            except Exception as exc:
                logger.warning("Tool %s failed: %s", tc.name, exc)
                results.tool_error = True
                results.trace.append({
                    "name": tc.name,
                    "status": "error",
                    "args": tc.args,
                    "error": str(exc),
                })

        self._apply_hierarchical_overview(results, plan, notebook_id, doc_ids)
        self._finalize_chunk_citations(results)
        return results

    # ------------------------------------------------------------------
    # Hierarchical overview route (roadmap Phase 3)
    # ------------------------------------------------------------------

    def _apply_hierarchical_overview(
        self,
        results: ToolResults,
        plan: PlanResult,
        notebook_id: str,
        doc_ids: list[str],
    ) -> None:
        """Deterministic hierarchical route for whole-source asks.

        Only ``retrieval_scope == "whole_source"`` may use this path. When
        the flag is on and every requested document has a current complete
        hierarchy, replace the gathered tool evidence with hydrated
        round-robin raw evidence and attach the summary map. Otherwise emit
        a fallback trace item and keep the existing 6/4 results untouched.
        """
        if not config.hierarchical_overview_enabled:
            return
        if plan.retrieval_scope != "whole_source":
            return
        from ..hierarchical_context import (
            HierarchicalContextAdapter,
            hydrate_chunks,
            start_background_builds,
        )

        ctx = HierarchicalContextAdapter(notebook_id, doc_ids).prepare()
        if ctx.status != "ready":
            building = (
                config.overview_summary_reader
                and ctx.reason == "missing_or_stale_index"
                and start_background_builds(notebook_id, doc_ids)
            )
            results.trace.append({
                "name": "hierarchical_overview",
                "status": "building" if building else "fallback",
                "args": {"reason": ctx.reason or "missing_or_stale_index"},
            })
            return
        try:
            rows = hydrate_chunks(doc_ids, ctx.evidence_chunk_ids)
        except Exception as exc:  # noqa: BLE001 — any route failure falls back
            logger.warning("Hierarchical overview hydration failed: %s", exc)
            results.trace.append({
                "name": "hierarchical_overview",
                "status": "fallback",
                "args": {"reason": f"hierarchy_validation_failed: {exc}"},
            })
            return
        results.chunks = [RetrievedChunk(
            chunk_id=rec["id"],
            document_id=rec["document_id"],
            text=rec["text"],
            title=rec.get("title", ""),
            section_path=rec.get("section_path", ""),
            page_start=rec.get("page_start", -1),
            page_end=rec.get("page_end", -1),
        ) for rec in rows]
        results.summary_context = ctx.summary_context
        results.trace.append({
            "name": "hierarchical_overview",
            "status": "ok",
            "args": ctx.trace_args,
        })

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
