"""StructureStatus enum — single source of truth for document outline reliability.

All other modules import from here.  No other file defines its own
structure-status strings or enumerated values.
"""

from __future__ import annotations

from enum import Enum


class StructureStatus(str, Enum):
    EMBEDDED_TOC = "embedded_toc"
    AUTO_DETECTED_OUTLINE = "auto_detected_outline"
    PAGE_WINDOW_OUTLINE = "page_window_outline"
    NO_RELIABLE_OUTLINE = "no_reliable_outline"
    UNKNOWN_LEGACY = "unknown_legacy"

    @property
    def is_reliable(self) -> bool:
        """Only an embedded TOC extracted directly from PDF bookmarks is
        considered fully reliable for inferring chapter structure."""
        return self in (StructureStatus.EMBEDDED_TOC,)

    @property
    def has_any_outline(self) -> bool:
        """True when the document has *some* kind of outline (even a
        synthetic or page-window one).  False for unknown / no-outline."""
        return self not in (
            StructureStatus.NO_RELIABLE_OUTLINE,
            StructureStatus.UNKNOWN_LEGACY,
        )
