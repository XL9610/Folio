"""Immutable search options for conditional reranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..retrieval.boost_config import BoostConfig
    from ..retrieval.search_filters import SearchFilters


@dataclass(frozen=True)
class SearchOptions:
    """Immutable per-request options passed through the retrieval pipeline.

    Each request gets its own instance — concurrent requests with different
    rerank settings cannot interfere with one another.
    """

    profile_name: str = "baseline"
    rerank_enabled: bool = False
    rerank_top_n: int = 20
    filters: SearchFilters | None = None
    boost_config: BoostConfig | None = None
