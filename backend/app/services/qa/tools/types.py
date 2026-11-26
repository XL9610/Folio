"""Result types shared by the QA tools."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from ...retrieval.base import RetrievedChunk

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
    # Hierarchical overview (roadmap Phase 3): generated navigation/synthesis
    # cards. Deliberately NOT RetrievedChunk instances and never receive
    # citation labels — only hydrated raw chunks are citable evidence.
    summary_context: list[dict] = field(default_factory=list)


ToolFn = Callable[..., None]  # mutates ToolResults + appends trace
