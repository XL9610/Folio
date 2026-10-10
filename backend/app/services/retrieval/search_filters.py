from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchFilters:
    """Immutable filter constraints passed through the retrieval pipeline.

    All fields optional — None means 'no constraint on this dimension'.
    Filters are applied at the backend level when natively supported,
    or as post-filtering after fusion when not.
    """

    document_ids: tuple[str, ...] | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_paths: tuple[str, ...] | None = None

    def has_any_filter(self) -> bool:
        return (
            self.document_ids is not None
            or self.page_start is not None
            or self.page_end is not None
            or self.section_paths is not None
        )

    def matches(self, chunk: Any) -> bool:
        if self.document_ids is not None and chunk.document_id not in self.document_ids:
            return False
        if self.page_start is not None and chunk.page_start < self.page_start:
            return False
        if self.page_end is not None and chunk.page_end > self.page_end:
            return False
        return not (self.section_paths is not None and chunk.section_path not in self.section_paths)


def normalize_page_range(page_start: int | None, page_end: int | None) -> tuple[int | None, int | None]:
    """Ensure page_start <= page_end. Swap if inverted."""
    if page_start is not None and page_end is not None and page_start > page_end:
        return page_end, page_start
    return page_start, page_end
