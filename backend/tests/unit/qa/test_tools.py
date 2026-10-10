"""Unit tests for QAToolExecutor helpers — Module G conditional reranker."""

from __future__ import annotations

from app.services.qa.tools import _should_rerank


class TestShouldRerank:
    def test_explain_concept(self):
        assert _should_rerank("explain_concept") is True

    def test_compare(self):
        assert _should_rerank("compare") is True

    def test_specific_search(self):
        assert _should_rerank("specific_search") is True

    def test_locate(self):
        assert _should_rerank("locate") is False

    def test_source_overview(self):
        assert _should_rerank("source_overview") is False

    def test_unknown_task(self):
        assert _should_rerank("unknown") is False

    def test_empty_task(self):
        assert _should_rerank("") is False
