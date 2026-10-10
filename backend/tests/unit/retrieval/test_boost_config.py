"""Unit tests for BoostConfig — Module C front matter boost signals."""

from __future__ import annotations

import pytest

from app.services.retrieval.boost_config import BoostConfig


class TestBoostConfig:
    def test_default_boost_config(self):
        bc = BoostConfig()
        assert bc.front_matter_boost is False
        assert bc.lead_paragraph_boost is False
        assert bc.lead_page_threshold == 0

    def test_immutability(self):
        bc = BoostConfig()
        with pytest.raises(AttributeError):
            bc.front_matter_boost = True  # type: ignore[misc]

    def test_equality(self):
        a = BoostConfig(front_matter_boost=True)
        b = BoostConfig(front_matter_boost=True)
        c = BoostConfig(front_matter_boost=False)
        assert a == b
        assert hash(a) == hash(b)
        assert a != c
