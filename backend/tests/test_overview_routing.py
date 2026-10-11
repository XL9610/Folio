"""Branch C (roadmap Phase 2): overview queries route through coverage, not
similarity. Deterministic regex + planner augmentation checks.
"""

from __future__ import annotations

from app.services.qa.agent_planner import _augment_overview_tool_calls
from app.services.qa.query_intent import is_overview_query
from app.services.qa.tools import ToolCall


def _tc(name: str, **args) -> ToolCall:
    return ToolCall(name=name, args=args)


class TestOverviewIntent:
    def test_ground_truth_overview_questions_match(self):
        assert is_overview_query(
            "What are the main stories and topics covered in this issue of "
            "U.S. News & World Report?"
        )
        assert is_overview_query("What does this document cover about opera?")
        assert is_overview_query("What does Module 6 cover?")
        assert is_overview_query("Summarize this book.")

    def test_pinpoint_questions_do_not_match(self):
        assert not is_overview_query("What is binary search and why is it faster?")
        assert not is_overview_query("Who narrates Wuthering Heights?")
        assert not is_overview_query("What is the load factor of a hash table?")
        assert not is_overview_query("Who won the 2026 FIFA World Cup?")


class TestOverviewRouting:
    DOCS = ["doc-a", "doc-b"]

    def test_unscoped_overview_replaces_similarity_with_coverage(self):
        routed = _augment_overview_tool_calls(
            "What does Module 6 cover?",
            [_tc("retrieve_chunks", query="module 6 topics", document_ids=self.DOCS)],
            self.DOCS,
        )
        names = [t.name for t in routed]
        assert "retrieve_chunks" not in names
        sweep = next(t for t in routed if t.name == "get_source_chunks")
        assert sweep.args["limit"] == 20
        assert sweep.args["_coverage"] is True

    def test_overview_without_retrieval_gets_coverage_sweep(self):
        routed = _augment_overview_tool_calls(
            "What are the main stories in this issue?",
            [_tc("get_source_outline", document_ids=self.DOCS)],
            self.DOCS,
        )
        assert any(t.name == "get_source_chunks" for t in routed)
        assert any(t.name == "get_source_outline" for t in routed)

    def test_scoped_overview_keeps_section_plan(self):
        plan = [_tc("find_sections", query="Chapter 1", document_ids=self.DOCS)]
        routed = _augment_overview_tool_calls("Summarize Chapter 1.", plan, self.DOCS)
        assert [t.name for t in routed] == ["find_sections"]

    def test_numbered_chapter_question_not_rerouted(self):
        # "What does Chapter 3 cover?" is scoped, not whole-document
        plan = [_tc("retrieve_chunks", query="supply and demand", document_ids=self.DOCS)]
        routed = _augment_overview_tool_calls(
            "What does Chapter 3, 'The Market Forces of Supply and Demand', cover?",
            plan,
            self.DOCS,
        )
        assert [t.name for t in routed] == ["retrieve_chunks"]

    def test_pinpoint_queries_untouched(self):
        plan = [_tc("retrieve_chunks", query="binary search", document_ids=self.DOCS)]
        routed = _augment_overview_tool_calls("What is binary search?", plan, self.DOCS)
        assert routed == plan
