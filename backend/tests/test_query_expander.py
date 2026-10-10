"""Unit tests for QueryExpander — deterministic query expansion (zero LLM calls)."""

from __future__ import annotations

from app.services.retrieval.query_expander import ExpandedQuery, QueryExpander


class TestQueryExpander:
    """TDD tests for the deterministic QueryExpander."""

    # ------------------------------------------------------------------
    # 1. High-signal extraction
    # ------------------------------------------------------------------

    def test_high_signal_preserved(self):
        """Quoted phrases like 'opportunity cost' must be preserved in the expansion."""
        expander = QueryExpander()
        result = expander.expand(
            user_query='what is "opportunity cost" in economics',
        )
        assert "opportunity cost" in result.expanded_query.lower()
        # The original query text should be at the front
        assert result.original_query == 'what is "opportunity cost" in economics'

    def test_chapter_not_lost(self):
        """Chapter references like 'Chapter 20' must survive expansion."""
        expander = QueryExpander()
        result = expander.expand(
            user_query="what topics are covered in Chapter 20",
        )
        assert "chapter 20" in result.expanded_query.lower()

    # ------------------------------------------------------------------
    # 2. Task keyword injection
    # ------------------------------------------------------------------

    def test_explain_task_injects_definition(self):
        """When plan_task='explain_concept', inject definition/overview keywords."""
        expander = QueryExpander()
        result = expander.expand(
            user_query="what is entropy",
            plan_task="explain_concept",
        )
        # At least one of the explain_concept bonus keywords should appear
        combined = result.expanded_query.lower()
        explain_words = {"definition", "overview", "introduction", "characteristics"}
        assert any(w in combined for w in explain_words), (
            f"Expected one of {explain_words} in: {result.expanded_query}"
        )

    def test_locate_task_injects_section_terms(self):
        """When plan_task='locate', inject section/chapter heading keywords."""
        expander = QueryExpander()
        result = expander.expand(
            user_query="find the section on mitochondria",
            plan_task="locate",
        )
        combined = result.expanded_query.lower()
        locate_words = {"section", "chapter", "heading", "page"}
        assert any(w in combined for w in locate_words), (
            f"Expected one of {locate_words} in: {result.expanded_query}"
        )

    # ------------------------------------------------------------------
    # 3. Concept validation from sub_questions
    # ------------------------------------------------------------------

    def test_concepts_bonus_only(self):
        """Concepts extracted from sub_questions are injected into the expansion."""
        expander = QueryExpander()
        result = expander.expand(
            user_query="what is a neural network",
            sub_questions=[
                "What is the definition of a neural network?",
                "What are the characteristics of neural networks?",
            ],
        )
        # Keywords from sub_questions should appear
        combined = result.expanded_query.lower()
        assert "definition" in combined or "characteristics" in combined, (
            f"Expected concepts in: {result.expanded_query}"
        )

    def test_concepts_invalid_skipped(self):
        """Concepts that do NOT appear in any sub_question text should be skipped."""
        expander = QueryExpander()
        result = expander.expand(
            user_query="what is a cat",
            sub_questions=[
                "How does a cat behave?",
            ],
        )
        # "dog" should NOT be injected because it doesn't appear in any sub_question
        assert "dog" not in result.expanded_query.lower().split()

    # ------------------------------------------------------------------
    # 4. Length limit
    # ------------------------------------------------------------------

    def test_length_limit(self):
        """The expanded_query must not exceed 500 characters."""
        expander = QueryExpander()
        # Create a very verbose setup
        result = expander.expand(
            user_query="explain the theory of " + "very " * 50 + "long query text",
            planner_query="detailed analysis and " + "comprehensive " * 30 + "review",
            plan_task="explain_concept",
            sub_questions=[
                "What is the overview of this topic?",
                "What are important characteristics and definitions?",
            ],
        )
        assert len(result.expanded_query) <= 500, (
            f"Expanded query is {len(result.expanded_query)} chars, "
            f"exceeds 500 limit: {result.expanded_query[:100]}..."
        )

    # ------------------------------------------------------------------
    # 5. Determinism and no I/O
    # ------------------------------------------------------------------

    def test_no_llm_call(self):
        """Expansion must be pure in-memory — no network, no LLM, no file I/O."""
        expander = QueryExpander()
        # Multiple calls with same input must produce identical output
        result1 = expander.expand(
            user_query="explain gradient descent",
            planner_query="optimization algorithms",
            plan_task="explain_concept",
            sub_questions=[
                "What is the definition of gradient descent?",
            ],
        )
        result2 = expander.expand(
            user_query="explain gradient descent",
            planner_query="optimization algorithms",
            plan_task="explain_concept",
            sub_questions=[
                "What is the definition of gradient descent?",
            ],
        )
        # Deterministic: same input → same output
        assert result1.expanded_query == result2.expanded_query
        assert result1.injected_terms == result2.injected_terms

    # ------------------------------------------------------------------
    # 6. Edge cases
    # ------------------------------------------------------------------

    def test_empty_query_handled(self):
        """Empty user_query should not crash."""
        expander = QueryExpander()
        result = expander.expand(user_query="")
        assert result.original_query == ""
        assert result.expanded_query == ""
        assert result.injected_terms == ""

    def test_single_word_query(self):
        """Single-word query should still produce a valid expansion."""
        expander = QueryExpander()
        result = expander.expand(
            user_query="photosynthesis",
            plan_task="explain_concept",
        )
        assert result.original_query == "photosynthesis"
        assert len(result.expanded_query) > 0
        assert len(result.expanded_query) <= 500

    def test_none_sub_questions(self):
        """None sub_questions should be handled gracefully."""
        expander = QueryExpander()
        result = expander.expand(
            user_query="test query",
            sub_questions=None,
        )
        assert result.original_query == "test query"
        assert "test query" in result.expanded_query

    def test_compare_task_injects_contrast_words(self):
        """plan_task='compare' should inject comparison-related bonus words."""
        expander = QueryExpander()
        result = expander.expand(
            user_query="compare python and javascript",
            plan_task="compare",
        )
        combined = result.expanded_query.lower()
        compare_words = {"difference", "similarity", "comparison", "contrast", "versus"}
        assert any(w in combined for w in compare_words), (
            f"Expected one of {compare_words} in: {result.expanded_query}"
        )

    def test_expanded_query_dataclass(self):
        """ExpandedQuery dataclass fields should be populated correctly."""
        result = ExpandedQuery(
            original_query="test",
            planner_query="plan",
            expanded_query="test plan extra",
            injected_terms="plan,extra",
        )
        assert result.original_query == "test"
        assert result.planner_query == "plan"
        assert result.expanded_query == "test plan extra"
        assert result.injected_terms == "plan,extra"

    def test_injected_terms_is_comma_separated(self):
        """injected_terms should be a comma-separated string of injected keywords."""
        expander = QueryExpander()
        result = expander.expand(
            user_query="explain neural networks",
            plan_task="explain_concept",
        )
        # injected_terms should contain task keywords
        assert isinstance(result.injected_terms, str)
        terms = result.injected_terms.split(",") if result.injected_terms else []
        assert any("definition" in t or "overview" in t or
                   "introduction" in t or "characteristics" in t
                   for t in terms)

    # ------------------------------------------------------------------
    # 7. Module B: section_hints, front matter, concept phrase (P1.2)
    # ------------------------------------------------------------------

    def test_section_hints_active(self):
        expander = QueryExpander()
        result = expander.expand(
            user_query="summarize chapter 5",
            section_hints=["Sonata Form Development"],
        )
        assert "Sonata Form Development" in result.expanded_query

    def test_front_matter_keywords(self):
        expander = QueryExpander()
        result = expander.expand(
            user_query="preface intended audience",
        )
        combined = result.expanded_query.lower()
        assert "preface" in combined
        assert "foreword" in combined
        assert "front_matter" in combined

    def test_explain_concept_phrase_quoted(self):
        expander = QueryExpander()
        result = expander.expand(
            user_query="What is opportunity cost?",
            plan_task="explain_concept",
        )
        assert '"opportunity cost"' in result.expanded_query

    def test_all_three_injections_together(self):
        expander = QueryExpander()
        result = expander.expand(
            user_query="What does the preface say about Chapter 3?",
            plan_task="explain_concept",
            section_hints=["Chapter 3 Overview"],
        )
        combined = result.expanded_query.lower()
        assert "preface" in combined
        assert "foreword" in combined
        assert "chapter 3 overview" in combined
        assert '"' in result.expanded_query

    def test_output_capped_at_word_boundary(self):
        expander = QueryExpander()
        long_query = "explain " + "very " * 200 + "long query text"
        result = expander.expand(
            user_query=long_query,
            plan_task="explain_concept",
            section_hints=["hint a", "hint b", "hint c"],
        )
        assert len(result.expanded_query) <= 500

    def test_existing_behavior_preserved(self):
        expander = QueryExpander()
        result = expander.expand(
            user_query="explain gradient descent",
            planner_query="optimization algorithms",
            plan_task="explain_concept",
        )
        assert result.original_query == "explain gradient descent"
        assert "definition" in result.expanded_query.lower() or \
               "overview" in result.expanded_query.lower() or \
               "introduction" in result.expanded_query.lower()
        assert len(result.expanded_query) <= 500
