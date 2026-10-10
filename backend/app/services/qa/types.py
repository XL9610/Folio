"""Shared response types used by both the agentic and legacy QA pipelines.

These dataclasses define the ask response shape and are the canonical
definitions for ToolTraceItem, CitationItem, and AskResult.

The production /api/ask path (agentic) and the legacy deterministic
orchestrator both depend on these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class ToolTraceItem:
    name: str
    status: str = "ok"
    args: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class CitationItem:
    chunk_id: str
    document_id: str
    text_snippet: str  # first ~120 chars of chunk text
    section_path: str = ""
    page_start: int = -1
    page_end: int = -1
    source_type: str = "book"  # "book" | "web"
    ref_label: str = ""        # e.g. "S1" or "W1"
    url: str = ""
    title: str = ""
    site_name: str = ""
    published_date: str = ""
    retrieved_at: str = ""


class CitationResponse(BaseModel):
    """Wire shape for a single citation — shared by the ask and generate paths."""

    chunk_id: str
    document_id: str
    text_snippet: str
    section_path: str = ""
    page_start: int = -1
    page_end: int = -1
    source_type: str = "book"
    ref_label: str = ""
    url: str = ""
    title: str = ""
    site_name: str = ""
    published_date: str = ""
    retrieved_at: str = ""


def make_book_citation(
    *,
    chunk_id: str,
    document_id: str,
    section_path: str = "",
    page_start: int = -1,
    page_end: int = -1,
    title: str = "",
    ref_label: str = "",
    text_snippet: str = "",
) -> CitationItem:
    """Build the canonical book CitationItem — the single citation shape for ask + generate."""
    return CitationItem(
        chunk_id=chunk_id,
        document_id=document_id,
        text_snippet=text_snippet[:120],
        section_path=section_path,
        page_start=page_start,
        page_end=page_end,
        source_type="book",
        ref_label=ref_label,
        title=title,
    )


def collect_book_citations(
    items: list[dict],
    document_id: str,
    title: str = "",
) -> list[CitationItem]:
    """Build unique book citations from enriched items carrying source_chunk_id + section/page."""
    seen: set[str] = set()
    citations: list[CitationItem] = []
    for item in items:
        cid = item.get("source_chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        citations.append(make_book_citation(
            chunk_id=cid,
            document_id=document_id,
            section_path=item.get("section_path", ""),
            page_start=item.get("page_start", -1),
            page_end=item.get("page_end", -1),
            title=title,
        ))
    return citations


@dataclass
class AskResult:
    notebook_id: str
    query: str
    answer: str
    intent: str = ""
    confidence: float = 1.0
    needs_clarification: bool = False
    agent_status: str = "completed"
    planner_task: str = ""
    used_llm: bool = False
    citations: list[CitationItem] = field(default_factory=list)
    tool_trace: list[ToolTraceItem] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    source_metadata: list[dict] = field(default_factory=list)
