"""Tests for M6 Conditional Reranker."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.services.retrieval.search_options import SearchOptions


class TestSearchOptions:
    """Immutable SearchOptions dataclass."""

    def test_default_rerank_disabled(self):
        opts = SearchOptions()
        assert opts.rerank_enabled is False
        assert opts.rerank_top_n == 20

    def test_rerank_enabled(self):
        opts = SearchOptions(rerank_enabled=True, rerank_top_n=15)
        assert opts.rerank_enabled is True
        assert opts.rerank_top_n == 15

    def test_immutable(self):
        opts = SearchOptions(rerank_enabled=True)
        with pytest.raises(FrozenInstanceError):
            opts.rerank_enabled = False


class TestShouldRerank:
    """Task-based rerank gating."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("locate", False),
            ("overview", False),
            ("explain_concept", True),
            ("compare", True),
            ("specific_search", True),
            ("answer_directly", False),
            ("summarize", False),
            ("clarify", False),
            ("nonexistent_task", False),
        ],
        ids=['locate_no_rerank', 'overview_no_rerank', 'explain_concept_rerank', 'compare_rerank', 'specific_search_rerank', 'answer_directly_no_rerank', 'summarize_no_rerank', 'clarify_no_rerank', 'unknown_task_no_rerank'],
    )
    def test_should_rerank(self, value, expected):
        from app.services.qa.tools import _should_rerank
        assert _should_rerank(value) is expected

    def test_whole_source_scope_suppresses_pointwise_rerank(self):
        from app.services.qa.tools import _should_rerank
        assert _should_rerank("explain_concept", "whole_source") is False


class TestConcurrentSafety:
    """SearchOptions must be immutable — concurrent requests safe."""

    def test_different_requests_independent(self):
        req1 = SearchOptions(rerank_enabled=True, rerank_top_n=10)
        req2 = SearchOptions(rerank_enabled=False, rerank_top_n=20)
        assert req1.rerank_enabled != req2.rerank_enabled
        assert req1.rerank_top_n != req2.rerank_top_n
