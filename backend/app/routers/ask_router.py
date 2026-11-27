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

import json
import logging
import queue
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..repositories.document_repository import list_documents
from ..repositories.notebook_repository import get_notebook, get_or_create_default
from ..services.llm import LLMConfig

# noqa F401 below: re-exports kept so tests can import from ask_router.
from ..services.qa.ask_debug import _debug_retrieved_chunks, _debug_source_metadata  # noqa: F401
from ..services.qa.ask_pipeline import (  # noqa: F401
    MAX_AGENT_ROUNDS,
    MAX_WALL_CLOCK_SECONDS,
    AgenticAskPipeline,
    _apply_overview_presentation,
    _merge_tool_results,
)
from ..services.qa.source_framing import _with_source_framing_questions  # noqa: F401
from ..services.qa.structural_answers import (  # noqa: F401
    _try_answer_structural_query,
    _with_unresolved_section_disclaimer,
)
from ..services.qa.types import CitationResponse
from .ask_schemas import (  # noqa: F401 — AskHistoryItem/AnswerPreferences are re-exports
    AnswerPreferences,
    AskHistoryItem,
    AskRequest,
    AskResponse,
    RetrievedChunkDebugResponse,
    ToolTraceResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ask"])

_agent: AgenticAskPipeline | None = None


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
    kwargs = _prepare(request)
    try:
        result = _get_agent().answer(**kwargs)
    except RuntimeError as exc:
        # LLM provider failure
        raise HTTPException(
            status_code=503,
            detail={
                "code": "llm_provider_error",
                "message": str(exc)
            }
        ) from exc
    return _to_response(result, request)


@router.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Same as /ask, streamed as server-sent events.

    Events (one JSON object per ``data:`` line): {"type": "step", "reasoning",
    "label"} while gathering evidence, {"type": "token", "text"} answer deltas,
    then {"type": "done", "result": <AskResponse>} or {"type": "error",
    "message"}. The final answer in "done" is authoritative: post-processing
    can differ from the streamed tokens.
    """
    kwargs = _prepare(request)
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            result = _get_agent().answer(**kwargs, on_event=events.put)
            events.put({"type": "done", "result": _to_response(result, request).model_dump()})
        except Exception as exc:  # noqa: BLE001 — reported to the client as an event
            logger.warning("Streaming ask failed: %s", exc)
            events.put({"type": "error", "message": str(exc)})
        events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def sse():
        while (event := events.get()) is not None:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


def _prepare(request: AskRequest) -> dict:
    """Validate the request and resolve notebook/documents into answer() kwargs."""
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

    history = [{"role": h.role, "text": h.text, "intent": h.intent} for h in request.history]
    preferences = (
        request.preferences.model_dump(exclude_none=True)
        if request.preferences is not None
        else None
    )
    return {
        "notebook_id": notebook_id,
        "query": request.query,
        "document_ids": document_ids,
        "top_k": request.top_k,
        "history": history,
        "allow_web": request.allow_web,
        "preferences": preferences,
        "follow_up": request.follow_up,
        "debug_retrieval": request.debug_retrieval,
    }


def _to_response(result, request: AskRequest) -> AskResponse:
    return AskResponse(
        notebook_id=result.notebook_id,
        query=result.query,
        answer=result.answer,
        agent_status=result.agent_status,
        planner_task=result.planner_task,
        used_llm=result.used_llm,
        internal_answer=result.internal_answer,
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
