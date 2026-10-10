"""Data models for LLM-based TOC classification."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TOCEntryForLLM:
    """Minimal TOC entry sent to the LLM for classification.

    Now includes tree metadata (level, has_children) so the LLM
    can select the right parent entry rather than only leaf entries.
    """
    index: int
    title: str
    path: list[str]
    page_start: int
    page_end: int
    level: int = 1
    has_children: bool = False

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "path": self.path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "level": self.level,
            "has_children": self.has_children,
        }


@dataclass
class TOCClassificationResult:
    """Result from LLM-based TOC classification."""
    first_content_index: int         # index of first real study content entry
    skip_indices: list[int] = field(default_factory=list)
    keep_indices: list[int] = field(default_factory=list)
    confidence: float = 0.0          # 0.0 - 1.0
    reasoning_summary: str = ""

    @property
    def is_usable(self) -> bool:
        """Check if the result is confident enough to use."""
        return self.confidence >= 0.65
