"""Retrieval result data model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """A single retrieved chunk from any search backend."""

    chunk_id: str
    document_id: str
    text: str
    title: str = ""
    section_path: str = ""
    page_start: int = -1
    page_end: int = -1

    # Source scores (backend-specific)
    dense_score: float = 0.0
    fts_score: float = 0.0
    rrf_score: float = 0.0

    # Extra
    metadata: dict = field(default_factory=dict)

    def citation(self) -> str:
        """Produce a human-readable citation string."""
        scope = (
            f"p. {self.page_start}" if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        if self.section_path:
            return f"{self.document_id}, {scope}, {self.section_path}"
        return f"{self.document_id}, {scope}"
