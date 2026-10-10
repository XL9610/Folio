"""Unit tests for QueryContext — Module B query understanding."""

from __future__ import annotations

import pytest

from app.services.retrieval.query_context import (
    QueryContext,
    build_query_context,
)


class TestQueryContextBuild:
    def test_detect_chapter_20(self):
        ctx = build_query_context("What did Chapter 20 discuss?")
        assert ctx.has_chapter_ref is True
        assert ctx.chapter_numbers == (20,)

    def test_detect_ch_abbrev(self):
        ctx = build_query_context("Show me Ch. 5")
        assert ctx.has_chapter_ref is True
        assert ctx.chapter_numbers == (5,)

    def test_detect_multiple_chapters(self):
        ctx = build_query_context("Compare Chapter 3 and Chapter 7")
        assert ctx.chapter_numbers == (3, 7)

    def test_no_chapter_ref(self):
        ctx = build_query_context("What is opportunity cost?")
        assert ctx.has_chapter_ref is False
        assert ctx.chapter_numbers == ()

    def test_detect_preface(self):
        ctx = build_query_context("What is in the preface?")
        assert ctx.has_front_matter is True

    def test_detect_intended_audience(self):
        ctx = build_query_context("Who is the intended audience?")
        assert ctx.has_front_matter is True

    def test_no_front_matter_keywords(self):
        ctx = build_query_context("Explain sonata form")
        assert ctx.has_front_matter is False

    def test_low_structure_doc(self):
        ctx = build_query_context(
            "some query",
            document_metadata=[{"structure_status": "no-TOC"}],
        )
        assert ctx.is_low_structure is True

    def test_mixed_structure_docs(self):
        ctx = build_query_context(
            "some query",
            document_metadata=[
                {"structure_status": "good"},
                {"structure_status": "no-TOC"},
            ],
        )
        assert ctx.is_low_structure is True

    def test_empty_metadata(self):
        ctx = build_query_context("some query", document_metadata=[])
        assert ctx.is_low_structure is False
        assert ctx.has_chapter_ref is False

    def test_explain_concept_phrase(self):
        ctx = build_query_context(
            "What is opportunity cost?",
            planner_task="explain_concept",
        )
        assert ctx.concept_phrase == "opportunity cost"

    def test_explain_concept_no_phrase(self):
        ctx = build_query_context("explain", planner_task="explain_concept")
        assert ctx.concept_phrase == ""

    def test_resolved_chapters_mapping(self):
        ctx = build_query_context(
            "What did Chapter 20 discuss?",
            resolved_chapters={20: "Sonata Form"},
        )
        assert ctx.resolved_titles == ("Sonata Form",)

    def test_immutability(self):
        ctx = QueryContext(has_chapter_ref=True, chapter_numbers=(1, 2))
        with pytest.raises(AttributeError):
            ctx.has_chapter_ref = False  # type: ignore[misc]

    def test_defaults(self):
        ctx = QueryContext()
        assert ctx.has_chapter_ref is False
        assert ctx.chapter_numbers == ()
        assert ctx.resolved_titles == ()
        assert ctx.has_front_matter is False
        assert ctx.concept_phrase == ""
        assert ctx.is_low_structure is False
