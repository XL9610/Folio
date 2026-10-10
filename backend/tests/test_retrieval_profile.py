"""Tests for RetrievalProfile dataclass and preset loading."""
from __future__ import annotations

import pytest

from app.services.retrieval.retrieval_profile import (
    PROFILES,
    RetrievalProfile,
    load_profile,
)


class TestRetrievalProfile:
    """Test suite for RetrievalProfile dataclass and load_profile factory."""

    def test_profile_defaults(self) -> None:
        """Baseline profile fields should have correct default values."""
        profile = RetrievalProfile(name="baseline")

        assert profile.name == "baseline"
        assert profile.dense_weight == 1.0
        assert profile.keyword_weight == 1.0
        assert profile.rrf_k == 60
        assert profile.dense_fetch_k_mult == 2
        assert profile.keyword_fetch_k_mult == 2
        assert profile.final_top_k == 10
        assert profile.rerank_enabled is False
        assert profile.rerank_top_n == 20

    def test_load_dense_heavy(self) -> None:
        """load_profile('dense_heavy') should return (1.5, 0.5) weights."""
        profile = load_profile("dense_heavy")

        assert profile.name == "dense_heavy"
        assert profile.dense_weight == 1.5
        assert profile.keyword_weight == 0.5

    def test_invalid_name_fallback(self) -> None:
        """load_profile('nonexistent') should fall back to baseline."""
        profile = load_profile("nonexistent")

        assert profile.name == "baseline"
        assert profile.dense_weight == 1.0
        assert profile.keyword_weight == 1.0

    def test_different_profile_differs(self) -> None:
        """Different profile names should yield meaningfully different weights."""
        dense = load_profile("dense_heavy")
        keyword = load_profile("keyword_heavy")

        # The two profiles should differ in at least one weight
        assert dense.dense_weight != keyword.dense_weight
        assert dense.keyword_weight != keyword.keyword_weight
        # dense_heavy should emphasize dense over keyword
        assert dense.dense_weight > dense.keyword_weight
        # keyword_heavy should emphasize keyword over dense
        assert keyword.keyword_weight > keyword.dense_weight

    def test_frozen_dataclass(self) -> None:
        """RetrievalProfile should be immutable (frozen=True)."""
        profile = RetrievalProfile(name="test")

        with pytest.raises(Exception):
            profile.dense_weight = 999.0  # type: ignore[misc]

    def test_all_preset_names_map_correctly(self) -> None:
        """Every key in PROFILES should have a matching name field."""
        for name, profile in PROFILES.items():
            assert profile.name == name, (
                f"Profile key '{name}' does not match profile.name='{profile.name}'"
            )
