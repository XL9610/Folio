from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DocumentPage:
    """Represents a single page extracted from a PDF."""
    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TocEntry:
    """A Table-of-Contents entry with hierarchical support."""
    title: str
    level: int
    page_start: int
    page_end: int | None = None
    children: list["TocEntry"] = field(default_factory=list)

    @property
    def page_range(self) -> tuple[int, int | None]:
        """Return (start, end) page range for this entry."""
        return (self.page_start, self.page_end)


@dataclass
class Chunk:
    """A text chunk produced by the chunking pipeline.

    Includes all metadata needed by the RAG retriever to provide
    proper citations back to the source document.
    """
    text: str
    document_id: str
    chunk_index: int

    # TOC / section metadata
    title: str = ""
    section_path: list[str] = field(default_factory=list)

    # Page metadata
    page_start: int = -1
    page_end: int = -1

    # Strategy metadata
    chunk_strategy: str = "unknown"

    # Extra storage for embedding pipeline
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk to a dictionary suitable for vector DB storage."""
        return {
            "text": self.text,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "section_path": " > ".join(self.section_path) if self.section_path else self.title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_strategy": self.chunk_strategy,
            **self.metadata,
        }

    def citation(self) -> str:
        """Produce a human-readable citation string."""
        pages = (
            f"pp. {self.page_start}" if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        section = " > ".join(self.section_path) if self.section_path else self.title
        if section:
            return f"{self.document_id}, {pages}, {section}"
        return f"{self.document_id}, {pages}"


@dataclass
class TOCUnit:
    """A TOC entry with its full hierarchical context for chunking.

    Unlike a raw TocEntry, this carries the section_path, level,
    has_children flag, and a preorder index — everything the chunker
    and LLM classifier need in one place.
    """
    entry: TocEntry
    section_path: list[str]
    level: int
    has_children: bool
    index: int  # preorder index in the full TOC tree


@dataclass
class ChunkingResult:
    """Aggregated result of the chunking pipeline for one document."""
    document_id: str
    pages: list[DocumentPage] = field(default_factory=list)
    toc: list[TocEntry] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    strategy_used: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    # TOC filtering metadata
    toc_entries_total: int = 0
    toc_entries_used: int = 0
    front_matter_skipped: bool = False
    skipped_titles: list[str] = field(default_factory=list)
    first_used_title: str = ""

    # LLM TOC filter metadata
    toc_filter_strategy: str = "rules"       # "rules" or "rules+llm"
    llm_toc_filter_used: bool = False
    llm_first_content_title: str = ""
    llm_confidence: float = 0.0
    llm_reasoning_summary: str = ""

    # Hierarchical chunking metadata
    first_chunk_title: str = ""
    chunking_mode: str = ""   # e.g. "hierarchical_toc" or "token_fallback"
    embedded_toc_present: bool | None = None
    synthetic_outline_mode: str = ""

    # Content boundary metadata (body window)
    body_start_title: str = ""
    body_end_title: str = ""
    front_matter_skipped_count: int = 0
    back_matter_skipped_count: int = 0
    wrapper_skipped_count: int = 0
    back_matter_skipped_titles: list[str] = field(default_factory=list)
    wrapper_skipped_titles: list[str] = field(default_factory=list)
