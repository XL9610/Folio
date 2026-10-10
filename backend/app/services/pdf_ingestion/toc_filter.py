"""TOC cleaning and filtering policies.

Body-window detection via content_boundary.find_body_window() removes
front matter (title page, TOC, preface, ...) and back matter (bibliography,
index, glossary, ...) generically across book formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import TOCUnit


@dataclass
class FilterDebugInfo:
    """Debug metadata returned by filter_units so callers can inspect decisions."""
    skipped_titles: list[str] = field(default_factory=list)
    first_used_title: str = ""
    total_entries: int = 0
    used_entries: int = 0

    # Body-window fields
    back_matter_skipped_titles: list[str] = field(default_factory=list)
    wrapper_skipped_titles: list[str] = field(default_factory=list)
    last_used_title: str = ""
    back_matter_start_title: str = ""
    wrapper_detected: bool = False


@dataclass
class TOCFilter:
    """Body-window TOC filter that works across many book formats.

    Removes front and back matter from a TOC unit list, keeping the
    study-content body window in between.

    Usage::

        toc_filter = TOCFilter()
        filtered, skipped, debug = toc_filter.filter_units(units)
    """

    include_front_matter: bool = False
    page_count: int = 0  # set by chunker so we know document size

    def filter_units(
        self,
        units: list[TOCUnit],
    ) -> tuple[list[TOCUnit], int, FilterDebugInfo]:
        """Filter TOC-units using body-window detection (front + back matter removal).

        Uses content_boundary.find_body_window() for generic body-window
        detection, replacing the old prefix-only trim to also remove
        back matter (bibliography, index, glossary, etc.).

        Returns:
            (filtered_units, skipped_count, debug_info)
        """
        debug = FilterDebugInfo(total_entries=len(units))

        if self.include_front_matter:
            debug.used_entries = len(units)
            return units, 0, debug

        if not units:
            return units, 0, debug

        from .content_boundary import find_body_window

        decision = find_body_window(units, self.page_count)

        # Populate debug info from body-window decision
        debug.skipped_titles = decision.skipped_front_titles
        debug.back_matter_skipped_titles = decision.skipped_back_titles
        debug.wrapper_skipped_titles = decision.skipped_wrapper_titles
        debug.first_used_title = decision.body_start_title
        debug.last_used_title = decision.body_end_title
        debug.back_matter_start_title = decision.back_matter_start_title
        debug.wrapper_detected = decision.wrapper_detected
        debug.used_entries = len(decision.filtered_units)

        skipped_count = decision.body_start_index

        filtered = decision.filtered_units

        # Safety: if everything was filtered, return all
        if not filtered and units:
            debug.first_used_title = units[0].entry.title
            debug.used_entries = len(units)
            return units, 0, debug

        return filtered, skipped_count, debug
