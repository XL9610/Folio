import pytest
from app.services.retrieval.profile_selector import ProfileSelector


class TestProfileSelector:
    def test_explain_concept_with_specific_terms_returns_keyword_heavy(self):
        selector = ProfileSelector()
        result = selector.select(
            task="explain_concept",
            query="What is the concept of marginal utility?",
            document_metadata=[],
        )
        assert result == "keyword_heavy"

    def test_explain_concept_with_generic_query_returns_balanced(self):
        selector = ProfileSelector()
        result = selector.select(
            task="explain_concept",
            query="explain this",
            document_metadata=[],
        )
        assert result == "balanced"

    def test_compare_with_multi_doc_returns_dense_heavy(self):
        selector = ProfileSelector()
        result = selector.select(
            task="compare",
            query="Compare X and Y",
            document_metadata=[{"id": "1"}, {"id": "2"}, {"id": "3"}],
        )
        assert result == "dense_heavy"

    def test_compare_with_single_doc_returns_balanced(self):
        selector = ProfileSelector()
        result = selector.select(
            task="compare",
            query="Compare X and Y",
            document_metadata=[{"id": "1"}],
        )
        assert result == "balanced"

    def test_locate_returns_keyword_heavy(self):
        selector = ProfileSelector()
        result = selector.select(
            task="locate",
            query="Where is Chapter 5?",
            document_metadata=[],
        )
        assert result == "keyword_heavy"

    def test_source_overview_returns_keyword_heavy(self):
        selector = ProfileSelector()
        result = selector.select(
            task="source_overview",
            query="What does this book cover?",
            document_metadata=[],
        )
        assert result == "keyword_heavy"

    def test_unknown_task_returns_balanced(self):
        selector = ProfileSelector()
        result = selector.select(
            task="unknown_label",
            query="some query",
            document_metadata=[],
        )
        assert result == "balanced"

    def test_empty_task_returns_balanced(self):
        selector = ProfileSelector()
        result = selector.select(
            task="",
            query="some query",
            document_metadata=[],
        )
        assert result == "balanced"

    def test_empty_metadata_returns_balanced(self):
        selector = ProfileSelector()
        result = selector.select(
            task="specific_search",
            query="some query",
            document_metadata=[],
        )
        assert result == "balanced"

    def test_single_word_query_returns_balanced(self):
        selector = ProfileSelector()
        result = selector.select(
            task="explain_concept",
            query="Why",
            document_metadata=[],
        )
        assert result == "balanced"

    def test_has_specific_terms_two_title_case(self):
        selector = ProfileSelector()
        result = selector._has_specific_terms("Opportunity Cost explained")
        assert result is True

    def test_has_specific_terms_three_non_stopwords(self):
        selector = ProfileSelector()
        result = selector._has_specific_terms("explain the concept of marginal utility")
        assert result is True

    def test_has_specific_terms_generic(self):
        selector = ProfileSelector()
        result = selector._has_specific_terms("what is this")
        assert result is False
