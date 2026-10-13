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

    def test_locate_no_rerank(self):
        from app.services.qa.tools import _should_rerank
        assert _should_rerank("locate") is False

    def test_overview_no_rerank(self):
        from app.services.qa.tools import _should_rerank
        assert _should_rerank("overview") is False

    def test_explain_concept_rerank(self):
        from app.services.qa.tools import _should_rerank
        assert _should_rerank("explain_concept") is True

    def test_whole_source_scope_suppresses_pointwise_rerank(self):
        from app.services.qa.tools import _should_rerank
        assert _should_rerank("explain_concept", "whole_source") is False

    def test_compare_rerank(self):
        from app.services.qa.tools import _should_rerank
        assert _should_rerank("compare") is True

    def test_specific_search_rerank(self):
        from app.services.qa.tools import _should_rerank
        assert _should_rerank("specific_search") is True

    def test_answer_directly_no_rerank(self):
        from app.services.qa.tools import _should_rerank
        assert _should_rerank("answer_directly") is False

    def test_summarize_no_rerank(self):
        from app.services.qa.tools import _should_rerank
        assert _should_rerank("summarize") is False

    def test_clarify_no_rerank(self):
        from app.services.qa.tools import _should_rerank
        assert _should_rerank("clarify") is False

    def test_unknown_task_no_rerank(self):
        from app.services.qa.tools import _should_rerank
        assert _should_rerank("nonexistent_task") is False


class TestConcurrentSafety:
    """SearchOptions must be immutable — concurrent requests safe."""

    def test_different_requests_independent(self):
        req1 = SearchOptions(rerank_enabled=True, rerank_top_n=10)
        req2 = SearchOptions(rerank_enabled=False, rerank_top_n=20)
        assert req1.rerank_enabled != req2.rerank_enabled
        assert req1.rerank_top_n != req2.rerank_top_n
