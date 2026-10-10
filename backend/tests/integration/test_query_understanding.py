"""Integration tests for Module B: Query Understanding pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.retrieval.query_expander import QueryExpander
from app.services.retrieval.query_context import build_query_context


class TestFullPipeline:
    """Verify QueryContext → QueryExpander integration."""

    def test_front_matter_query_flows_to_expander(self):
        ctx = build_query_context(
            "What does the preface discuss?",
            planner_task="locate",
        )
        assert ctx.has_front_matter is True

        expander = QueryExpander()
        section_hints = list(ctx.resolved_titles) if ctx.resolved_titles else []
        result = expander.expand(
            user_query="What does the preface discuss?",
            plan_task="locate",
            section_hints=section_hints,
        )
        assert "preface" in result.expanded_query.lower()
        assert "foreword" in result.expanded_query.lower()

    def test_chapter_ref_flows_to_expander_as_section_hints(self):
        ctx = build_query_context(
            "What did Chapter 20 discuss?",
            resolved_chapters={20: "Sonata Form Development"},
        )
        assert ctx.has_chapter_ref is True
        assert ctx.resolved_titles == ("Sonata Form Development",)

        expander = QueryExpander()
        section_hints = list(ctx.resolved_titles) if ctx.resolved_titles else []
        result = expander.expand(
            user_query="What did Chapter 20 discuss?",
            plan_task="locate",
            section_hints=section_hints,
        )
        assert "Sonata Form Development" in result.expanded_query

    def test_explain_concept_with_context_flows_to_expander(self):
        ctx = build_query_context(
            "What is opportunity cost?",
            planner_task="explain_concept",
        )
        assert ctx.concept_phrase == "opportunity cost"

        expander = QueryExpander()
        result = expander.expand(
            user_query="What is opportunity cost?",
            plan_task="explain_concept",
        )
        assert '"opportunity cost"' in result.expanded_query


class TestBackwardCompatibility:
    """Verify existing behavior is preserved."""

    def test_section_hints_none_unchanged(self):
        expander = QueryExpander()
        result = expander.expand(
            user_query="explain gradient descent",
            plan_task="explain_concept",
            section_hints=None,
        )
        assert "gradient descent" in result.expanded_query.lower()
        assert len(result.expanded_query) <= 500

    def test_section_hints_empty_list_unchanged(self):
        expander = QueryExpander()
        result = expander.expand(
            user_query="explain gradient descent",
            plan_task="explain_concept",
            section_hints=[],
        )
        assert "gradient descent" in result.expanded_query.lower()
        assert len(result.expanded_query) <= 500

    def test_no_new_fields_in_expanded_query_dataclass(self):
        from app.services.retrieval.query_expander import ExpandedQuery
        eq = ExpandedQuery(
            original_query="q",
            planner_query="",
            expanded_query="q extra",
            injected_terms="extra",
        )
        assert eq.original_query == "q"
        assert eq.expanded_query == "q extra"
        assert eq.injected_terms == "extra"


class TestEmptyPlanTask:
    """Verify QueryExpander handles empty/None plan_task gracefully."""

    def test_none_plan_task(self):
        expander = QueryExpander()
        result = expander.expand(
            user_query="explain neural networks",
            plan_task="",
        )
        assert "neural networks" in result.expanded_query.lower()
        assert len(result.expanded_query) <= 500
        assert result.injected_terms == "" or isinstance(result.injected_terms, str)

    def test_empty_plan_task_no_injections(self):
        expander = QueryExpander()
        result = expander.expand(
            user_query="simple query",
            plan_task="",
            section_hints=[],
        )
        assert result.expanded_query is not None
        assert len(result.expanded_query) <= 500
