"""FastAPI router for the Ask endpoint — LLM-required agentic RAG.

POST /api/ask
  Requires LLM_REMOTE_ENABLED=true and valid LLM_API_KEY.
  Returns 503 if LLM is not configured.
  No deterministic fallback answers — either LLM generates, or we fail clearly.

Architecture (bounded loop):
  plan → [execute tools → evaluate evidence]×N → generate answer
  - Max N rounds (default 3)
  - Evidence evaluator gates whether to answer, clarify, or re-search
  - Clarification is a first-class outcome, not a fallback
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from ..core.config import config
from ..repositories.document_repository import list_documents
from ..repositories.notebook_repository import get_notebook, get_or_create_default
from ..services.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from ..services.index.sqlite_vec_store import SqliteVecStore
from ..services.llm import LLMConfig
from ..services.llm.client import create_llm_client
from ..services.qa.agent_generator import AgentAnswerGenerator
from ..services.qa.agent_planner import AgentPlanner
from ..services.qa.ask_debug import _debug_retrieved_chunks, _debug_source_metadata
from ..services.qa.evidence_evaluator import EvidenceEval, EvidenceEvaluator
from ..services.qa.follow_up import (
    _build_follow_up_query,
    _fallback_follow_up_plan,
    _should_continue_prior_topic,
)
from ..services.qa.source_framing import _with_source_framing_questions
from ..services.qa.structural_answers import (
    _try_answer_structural_query,
    _with_unresolved_section_disclaimer,
)
from ..services.qa.tools import QAToolExecutor, ToolResults, finalize_merged_citations
from ..services.qa.types import AskResult, CitationItem, CitationResponse, ToolTraceItem
from ..services.retrieval.llm_reranker import LLMReranker
from ..services.retrieval.retrieval_profile import load_profile
from ..services.retrieval.retriever import HybridRetriever
from ..services.web_search import get_web_search_provider

from .ask_schemas import (
    AskHistoryItem,  # re-export (kept importable from the router)
    AnswerPreferences,  # re-export
    AskRequest,
    AskResponse,
    RetrievedChunkDebugResponse,
    ToolTraceResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ask"])

# ── Agent loop limits ──
MAX_AGENT_ROUNDS = 3
MAX_WALL_CLOCK_SECONDS = 60

# ── Re-exports ─────────────────────────────────────────────────────────────
# These symbols moved to focused modules; re-exporting them here preserves the
# router's historical import surface (tests and config_router reach into
# `app.routers.ask_router`). Re-export only — definitions live in the modules
# named on the right.
from ..services.qa.ask_debug import _safe_json_object  # noqa: F401
from ..services.qa.follow_up import _latest_history_intent, _latest_history_text  # noqa: F401
from ..services.qa.source_framing import (  # noqa: F401
    _chunk_contains_concept,
    _citation_ref_by_chunk,
    _concept_phrase_from_query,
    _extract_questions_after_concept,
    _normalize_for_phrase,
)
from ..services.qa.structural_answers import (  # noqa: F401
    _common_top_section_label,
    _has_unreliable_structure,
    _page_span,
    _unique_section_matches,
)

# ---------------------------------------------------------------------------
# Agentic pipeline — LLM-required
# ---------------------------------------------------------------------------

_agent: AgenticAskPipeline | None = None


class AgenticAskPipeline:
    """LLM-first agentic QA pipeline with bounded loop.

    plan → [execute tools → evaluate evidence]×N → generate answer
    """

    def __init__(self):
        llm_config = LLMConfig()
        if not llm_config.is_configured:
            raise RuntimeError("LLM not configured")
        logger.info(
            "LLM config: provider=%s api_format=%s base_url=%s model=%s",
            llm_config.provider or "(default)",
            llm_config.api_format,
            llm_config.base_url or "(empty)",
            llm_config.model or "(empty)",
        )
        self._llm = create_llm_client(llm_config)
        self._planner = AgentPlanner(self._llm)

        # Optional LLM reranker (feature flag)
        reranker = None
        if config.rerank_enabled:
            reranker = LLMReranker(
                llm_client=self._llm,
                max_concurrency=config.rerank_concurrency,
                max_doc_chars=config.rerank_max_doc_chars,
            )

        self._tools = QAToolExecutor(
            retriever=HybridRetriever(
                embedder=SentenceTransformerEmbedder(),
                dense_store=SqliteVecStore(),
                reranker=reranker,
                rerank_top_n=config.rerank_top_n,
                profile=load_profile(config.retrieval_profile),
            ),
            web_search_provider=get_web_search_provider(),
        )
        self._evaluator = EvidenceEvaluator(self._llm)
        self._generator = AgentAnswerGenerator(self._llm)

    def answer(
        self,
        notebook_id: str,
        query: str,
        document_ids: list[str],
        top_k: int,
        history: list[dict],
        allow_web: bool = False,
        preferences: dict | None = None,
        follow_up: bool = False,
        debug_retrieval: bool = False,
    ) -> AskResult:
        """Run the agentic ask pipeline with bounded loop.

        Returns AskResult. agent_status may be:
          - "completed": answer generated from evidence
          - "clarification_needed": ambiguous query, needs user input
          - "insufficient_context": evidence too weak to answer
          - "timeout": wall-clock budget exhausted
        """
        t0 = time.monotonic()
        tool_trace: list[ToolTraceItem] = []
        evals: list[EvidenceEval] = []
        effective_query = _build_follow_up_query(query, history, follow_up)

        # ── 1. Plan ──
        plan = self._planner.plan(
            query=effective_query,
            history=history,
            document_ids=document_ids,
            allow_web=allow_web,
        )
        tool_trace.append(ToolTraceItem(
            name="planner",
            status="ok",
            args={"task": plan.task, "retrieval_scope": plan.retrieval_scope},
        ))

        # If planner itself says clarify
        if plan.clarify_needed and plan.clarifying_question:
            if _should_continue_prior_topic(query, history, follow_up):
                plan = _fallback_follow_up_plan(query, history, document_ids, plan, allow_web=allow_web)
                tool_trace.append(ToolTraceItem(
                    name="follow_up_guard",
                    status="ok",
                    args={"task": plan.task},
                ))
            else:
                return AskResult(
                    notebook_id=notebook_id,
                    query=query,
                    answer=plan.clarifying_question,
                    agent_status="clarification_needed",
                    planner_task=plan.task,
                    used_llm=True,
                    citations=[],
                    tool_trace=tool_trace,
                )

        # ── 2. Bounded tool-execution loop ──
        all_tool_results = ToolResults()
        final_eval: EvidenceEval | None = None

        for round_idx in range(1, MAX_AGENT_ROUNDS + 1):
            # Timeout check
            if time.monotonic() - t0 > MAX_WALL_CLOCK_SECONDS:
                logger.warning("Agent loop timed out at round %d", round_idx)
                tool_trace.append(ToolTraceItem(
                    name="agent_loop",
                    status="timeout",
                    args={"round": round_idx},
                    error=f"Exceeded {MAX_WALL_CLOCK_SECONDS}s budget",
                ))
                final_eval = EvidenceEval(
                    status="insufficient",
                    reason="Agent loop timed out before gathering sufficient evidence.",
                )
                break

            # ── 2a. Execute tools ──
            round_results = self._tools.execute(
                plan=plan,
                notebook_id=notebook_id,
                document_ids=document_ids,
                top_k=top_k,
                allow_web=allow_web,
                user_query=effective_query,
            )
            # Merge results
            _merge_tool_results(all_tool_results, round_results)
            for tr in round_results.trace:
                tool_trace.append(ToolTraceItem(
                    name=tr["name"],
                    status=tr.get("status", "ok"),
                    args=tr.get("args", {}),
                    error=tr.get("error", ""),
                ))

            # ── 2b. Structural guard (deterministic-only) ──
            # Use the original query (not the follow-up wrapper) so that the
            # evaluator checks the user's actual question against evidence. The
            # exact-term off-topic gate is skipped for follow-ups (grounded in
            # conversation history instead of fresh retrieval).
            eval_result = self._evaluator.evaluate(
                query=query,
                plan=plan,
                tool_results=all_tool_results,
                notebook_id=notebook_id,
                document_ids=document_ids,
                follow_up=follow_up,
            )
            evals.append(eval_result)
            # Only surface guard status to trace when it's NOT "proceed"
            if eval_result.status != "proceed":
                tool_trace.append(ToolTraceItem(
                    name="guard",
                    status=eval_result.status,
                    args={
                        "round": round_idx,
                        "chunks": len(all_tool_results.chunks),
                        "reason": eval_result.reason[:200],
                    },
                ))

            # ── 2c. Decide next action ──
            if eval_result.status == "ambiguous":
                # Return clarification question directly — do NOT generate answer
                return AskResult(
                    notebook_id=notebook_id,
                    query=query,
                    answer=eval_result.clarifying_question or "I need more information to answer that. Could you clarify?",
                    agent_status="clarification_needed",
                    planner_task=plan.task,
                    used_llm=True,
                    citations=[],
                    tool_trace=tool_trace,
                    retrieved_chunks=_debug_retrieved_chunks(all_tool_results) if debug_retrieval else [],
                    source_metadata=_debug_source_metadata(all_tool_results) if debug_retrieval else [],
                )

            # proceed / sufficient / insufficient → always break and generate
            # Generator owns semantic answer quality from full chunk context
            if eval_result.status == "insufficient":
                return AskResult(
                    notebook_id=notebook_id,
                    query=query,
                    answer="The selected source does not contain enough information to answer this.",
                    agent_status="insufficient_context",
                    planner_task=plan.task,
                    used_llm=True,
                    citations=[],
                    tool_trace=tool_trace,
                    retrieved_chunks=_debug_retrieved_chunks(all_tool_results) if debug_retrieval else [],
                    source_metadata=_debug_source_metadata(all_tool_results) if debug_retrieval else [],
                )

            if eval_result.status == "error":
                return AskResult(
                    notebook_id=notebook_id,
                    query=query,
                    answer="Search is currently unavailable. Please try again, or re-index your sources.",
                    agent_status="retrieval_unavailable",
                    planner_task=plan.task,
                    used_llm=True,
                    citations=[],
                    tool_trace=tool_trace,
                    retrieved_chunks=_debug_retrieved_chunks(all_tool_results) if debug_retrieval else [],
                    source_metadata=_debug_source_metadata(all_tool_results) if debug_retrieval else [],
                )

            final_eval = eval_result
            break

        structural_answer = _try_answer_structural_query(effective_query, all_tool_results)
        if structural_answer:
            tool_trace.append(ToolTraceItem(
                name="structural_answer",
                status="ok",
                args={"type": structural_answer["type"]},
            ))
            return AskResult(
                notebook_id=notebook_id,
                query=query,
                answer=structural_answer["answer"],
                agent_status="completed",
                planner_task=plan.task,
                used_llm=True,
                citations=[],
                tool_trace=tool_trace,
                retrieved_chunks=_debug_retrieved_chunks(all_tool_results) if debug_retrieval else [],
                source_metadata=_debug_source_metadata(all_tool_results) if debug_retrieval else [],
            )

        # ── 3. Generate answer ──
        # Renumber citations to match the passages the LLM actually sees
        # (≤10 book chunks / ≤8 web results), so returned refs never collide
        # across rounds or dangle past what was shown.
        finalize_merged_citations(all_tool_results)
        answer_text, citations = self._generator.generate(
            query=effective_query,
            history=history,
            plan=plan,
            tool_results=all_tool_results,
            source_info={
                "notebook_id": notebook_id,
                "document_ids": document_ids,
                "allow_web": allow_web,
                "evidence_status": final_eval.status if final_eval else "unknown",
                "evidence_reason": final_eval.reason if final_eval else "",
            },
            preferences=preferences,
        )
        answer_text = _with_source_framing_questions(effective_query, all_tool_results, answer_text)
        answer_text = _with_unresolved_section_disclaimer(effective_query, all_tool_results, answer_text)

        # Convert citations
        citems: list[CitationItem] = []
        seen_cids = set()
        for c in citations:
            cid = c.get("chunk_id", "")
            cite_key = cid or c.get("url", "")
            if cite_key in seen_cids:
                continue
            seen_cids.add(cite_key)
            citems.append(CitationItem(
                chunk_id=str(cid),
                document_id=str(c.get("document_id", "")),
                text_snippet=str(c.get("text_snippet", "")),
                section_path=str(c.get("section_path", "")),
                page_start=int(c.get("page_start", -1)),
                page_end=int(c.get("page_end", -1)),
                source_type=str(c.get("source_type", "book") or "book"),
                ref_label=str(c.get("ref_label", "")),
                url=str(c.get("url", "")),
                title=str(c.get("title", "")),
                site_name=str(c.get("site_name", "")),
                published_date=str(c.get("published_date", "")),
                retrieved_at=str(c.get("retrieved_at", "")),
            ))

        evidence_status = final_eval.status if final_eval else "unknown"
        agent_status = "completed"
        if evidence_status == "insufficient":
            agent_status = "insufficient_context"

        return AskResult(
            notebook_id=notebook_id,
            query=query,
            answer=answer_text,
            agent_status=agent_status,
            planner_task=plan.task,
            used_llm=True,
            citations=citems,
            tool_trace=tool_trace,
            retrieved_chunks=_debug_retrieved_chunks(all_tool_results) if debug_retrieval else [],
            source_metadata=_debug_source_metadata(all_tool_results) if debug_retrieval else [],
        )


# ---------------------------------------------------------------------------
# Helper: merge tool results across rounds
# ---------------------------------------------------------------------------

def _merge_tool_results(into: ToolResults, from_round: ToolResults) -> None:
    """Merge round-level ToolResults into accumulated results (dedup by chunk_id)."""
    # Metadata / outline: simple extend
    into.metadata.extend(from_round.metadata)
    into.outline.extend(from_round.outline)
    into.section_matches.extend(from_round.section_matches)

    # Chunks: dedup by chunk_id
    existing_ids = {c.chunk_id for c in into.chunks}
    for c in from_round.chunks:
        if c.chunk_id not in existing_ids:
            into.chunks.append(c)
            existing_ids.add(c.chunk_id)

    # Web results: dedup by URL
    existing_urls = {c.get("url", "") for c in into.web_results if c.get("url")}
    for c in from_round.web_results:
        url = c.get("url", "")
        if url and url not in existing_urls:
            into.web_results.append(c)
            existing_urls.add(url)

    # Citations: dedup by chunk_id for books and URL for web
    existing_cite_ids = {
        c.get("url") or c.get("chunk_id", "")
        for c in into.chunk_citations
    }
    for c in from_round.chunk_citations:
        cite_key = c.get("url") or c.get("chunk_id", "")
        if cite_key not in existing_cite_ids:
            into.chunk_citations.append(c)
            existing_cite_ids.add(cite_key)

    # Preserve target_section from the first round that sets it
    if from_round.target_section and not into.target_section:
        into.target_section = from_round.target_section

    # Retrieval failure is sticky across rounds: once a backend errored, the
    # merged result must still report it so the evaluator returns
    # retrieval_unavailable instead of treating it as a zero-hit.
    into.tool_error = into.tool_error or from_round.tool_error


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def _get_agent() -> AgenticAskPipeline:
    global _agent
    if _agent is None:
        _agent = AgenticAskPipeline()
    return _agent


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """LLM-required agentic QA. Returns 503 if LLM not configured.

    POST /api/ask
    Body:
      {
        "notebook_id": "default",
        "query": "What is gradient descent?",
        "top_k": 10,
        "document_ids": ["doc-id-1"],
        "history": [],
        "preferences": { "length": "short", "style": "plain" }
      }

    Requires LLM_REMOTE_ENABLED=true and valid LLM_API_KEY.
    """
    # --- Check LLM availability ---
    llm_config = LLMConfig()
    if not llm_config.is_configured:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "llm_not_configured",
                "message": "Agentic QA requires LLM_REMOTE_ENABLED=true and a valid LLM_API_KEY. Set these in backend/.env"
            }
        )

    # --- Resolve notebook ---
    if request.notebook_id == "default":
        nb = get_or_create_default()
        notebook_id = nb["id"]
    else:
        nb = get_notebook(request.notebook_id)
        if nb is None:
            raise HTTPException(status_code=404, detail=f"Notebook '{request.notebook_id}' not found.")
        notebook_id = nb["id"]

    # --- Resolve document_ids ---
    if request.document_ids is None:
        docs = list_documents(notebook_id)
        document_ids = [d["id"] for d in docs]
    else:
        document_ids = request.document_ids

    # --- Run agentic pipeline ---
    history = [{"role": h.role, "text": h.text, "intent": h.intent} for h in request.history]
    preferences = (
        request.preferences.model_dump(exclude_none=True)
        if request.preferences is not None
        else None
    )

    try:
        agent = _get_agent()
        result = agent.answer(
            notebook_id=notebook_id,
            query=request.query,
            document_ids=document_ids,
            top_k=request.top_k,
            history=history,
            allow_web=request.allow_web,
            preferences=preferences,
            follow_up=request.follow_up,
            debug_retrieval=request.debug_retrieval,
        )
    except RuntimeError as exc:
        # LLM provider failure
        raise HTTPException(
            status_code=503,
            detail={
                "code": "llm_provider_error",
                "message": str(exc)
            }
        ) from exc

    return AskResponse(
        notebook_id=result.notebook_id,
        query=result.query,
        answer=result.answer,
        agent_status=result.agent_status,
        planner_task=result.planner_task,
        used_llm=result.used_llm,
        citations=[
            CitationResponse(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text_snippet=c.text_snippet,
                section_path=c.section_path,
                page_start=c.page_start,
                page_end=c.page_end,
                source_type=c.source_type,
                ref_label=c.ref_label,
                url=c.url,
                title=c.title,
                site_name=c.site_name,
                published_date=c.published_date,
                retrieved_at=c.retrieved_at,
            )
            for c in result.citations
        ],
        tool_trace=[
            ToolTraceResponse(
                name=t.name,
                status=t.status,
                args=t.args,
                error=t.error,
            )
            for t in result.tool_trace
        ],
        retrieved_chunks=[
            RetrievedChunkDebugResponse(**item)
            for item in result.retrieved_chunks
        ] if request.debug_retrieval else [],
        source_metadata=result.source_metadata if request.debug_retrieval else [],
    )
