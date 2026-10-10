from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalProfile:
    """Immutable retrieval parameter bundle.

    Switched via RETRIEVAL_PROFILE env var.
    """

    name: str  # "baseline" | "keyword_heavy" | "dense_heavy" | "balanced"
    dense_weight: float = 1.0
    keyword_weight: float = 1.0
    rrf_k: int = 60
    dense_fetch_k_mult: int = 2  # dense_fetch_k = top_k * N
    keyword_fetch_k_mult: int = 2
    final_top_k: int = 10
    rerank_enabled: bool = False
    rerank_top_n: int = 20


# Presets
PROFILES: dict[str, RetrievalProfile] = {
    "baseline": RetrievalProfile(
        name="baseline",
        dense_fetch_k_mult=3,
        keyword_fetch_k_mult=3,
    ),
    "dense_heavy": RetrievalProfile(
        name="dense_heavy",
        dense_weight=1.5,
        keyword_weight=0.5,
        dense_fetch_k_mult=3,
    ),
    "keyword_heavy": RetrievalProfile(
        name="keyword_heavy",
        dense_weight=0.5,
        keyword_weight=1.5,
        keyword_fetch_k_mult=3,
    ),
    "balanced": RetrievalProfile(
        name="balanced",
        dense_weight=1.2,
        keyword_weight=0.8,
        dense_fetch_k_mult=3,
        keyword_fetch_k_mult=3,
    ),
}


def load_profile(name: str | None = None) -> RetrievalProfile:
    """Load profile by name. Falls back to 'baseline' if name is invalid."""
    if name and name in PROFILES:
        return PROFILES[name]
    return PROFILES["baseline"]
