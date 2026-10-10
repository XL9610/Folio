"""Pydantic request/response models for the Ask endpoint.

Kept separate from the router so the HTTP contract is readable without
scrolling past the agentic pipeline and structural-Q&A logic.
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from ..services.qa.types import CitationResponse


class AskHistoryItem(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    text: str = Field(..., description="Message content")
    intent: str = Field("", description="Intent of the assistant message, if known")


class AnswerPreferences(BaseModel):
    """Optional answer style preferences.

    These are presentation hints for final answer generation only. Extra keys are
    allowed so older and newer frontend settings payloads stay compatible.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    style: str | None = Field(None, max_length=40, description="Answer style preset")
    answer_style: str | None = Field(
        None,
        validation_alias=AliasChoices("answer_style", "answerStyle"),
        max_length=40,
        description="Answer style preset, camelCase-compatible",
    )
    length: str | None = Field(None, max_length=40, description="Target answer length")
    answer_length: str | None = Field(
        None,
        validation_alias=AliasChoices("answer_length", "answerLength"),
        max_length=40,
        description="Target answer length, camelCase-compatible",
    )
    detail_level: str | None = Field(
        None,
        validation_alias=AliasChoices("detail_level", "detailLevel"),
        max_length=40,
        description="Desired explanation depth",
    )
    tone: str | None = Field(None, max_length=40, description="Desired tone")
    format: str | None = Field(None, max_length=40, description="Preferred format")
    citations: str | None = Field(
        None,
        validation_alias=AliasChoices("citations", "citation_density", "citationDensity"),
        max_length=40,
        description="Preferred citation detail level",
    )
    auto_summary_table: bool | None = Field(
        None,
        validation_alias=AliasChoices("auto_summary_table", "autoSummaryTable"),
        description="Whether to include a compact summary table when useful",
    )
    audience: str | None = Field(None, max_length=80, description="Intended audience")
    language: str | None = Field(None, max_length=80, description="Preferred language")
    custom_instructions: str | None = Field(
        None,
        validation_alias=AliasChoices(
            "custom_instructions",
            "customInstructions",
            "instructions",
        ),
        max_length=1000,
        description="Additional style-only instructions",
    )


class AskRequest(BaseModel):
    notebook_id: str = Field("default", description="Notebook ID or 'default'")
    query: str = Field(..., min_length=1, description="User's question")
    top_k: int = Field(10, ge=1, le=30, description="Max chunks to retrieve")
    document_ids: list[str] | None = Field(
        None, description="Optional filter: only consider these documents"
    )
    history: list[AskHistoryItem] = Field(
        default_factory=list,
        description="Recent conversation history for context-aware routing"
    )
    allow_web: bool = Field(
        False,
        description="Opt-in internet research. When false, the agent only uses uploaded sources."
    )
    preferences: AnswerPreferences | None = Field(
        None,
        description="Optional answer style preferences. Does not affect planning or retrieval."
    )
    follow_up: bool = Field(
        False,
        description="Treat the query as a follow-up to the supplied history."
    )
    debug_retrieval: bool = Field(
        False,
        description="Eval/debug only: include retrieved passages and source structure metadata."
    )


class ToolTraceResponse(BaseModel):
    name: str
    status: str
    args: dict = {}
    error: str = ""


class RetrievedChunkDebugResponse(BaseModel):
    passage_id: str
    chunk_id: str
    document_id: str = ""
    text: str = ""
    title: str = ""
    section_path: str = ""
    section_label: str = ""
    page_start: int = -1
    page_end: int = -1
    outline_source: str = ""
    chunking_strategy: str = ""
    source_type: str = "book"
    dense_score: float = 0.0
    fts_score: float = 0.0
    rrf_score: float = 0.0
    score: float = 0.0
    structure_status: str = "unknown_legacy"
    structure_confidence: str = ""
    dense_rank: int = 0
    keyword_rank: int = 0
    fusion_rank: int = 0
    retrieval_latency_ms: int = 0
    planner_query: str = ""
    expanded_query: str = ""
    warnings: str = ""


class AskResponse(BaseModel):
    notebook_id: str
    query: str
    answer: str
    agent_status: str = "completed"
    planner_task: str = ""
    used_llm: bool = True
    citations: list[CitationResponse] = []
    tool_trace: list[ToolTraceResponse] = []
    agent_rounds: int = 1
    evidence_status: str = ""
    retrieved_chunks: list[RetrievedChunkDebugResponse] = []
    source_metadata: list[dict] = []
