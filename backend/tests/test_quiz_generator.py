"""Tests for QuizGenerator count handling and prompt requirements."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.generation.quiz_service import QuizGenerator


def make_fake_resolver(chunks=None, layer="toc"):
    from app.services.generation.scope_resolver import ScopeResult

    resolver = MagicMock()
    resolver.resolve.return_value = ScopeResult(chunks=chunks or [], metadata={"layer": layer})
    return resolver


def make_chunk(cid="c1", text="Gradient descent minimizes loss by following gradients."):
    from app.services.retrieval.base import RetrievedChunk

    return RetrievedChunk(
        chunk_id=cid,
        document_id="d1",
        text=text,
        title="",
        section_path="Ch 1",
        page_start=1,
        page_end=1,
    )


def q_json(*questions):
    import json

    return json.dumps({"questions": list(questions)})


def question(n, source=1):
    return {
        "question": f"What is concept {n}?",
        "choices": ["A. Alpha", "B. Beta", "C. Gamma", "D. Delta"],
        "answer": "A",
        "explanation": f"Because concept {n} is supported.",
        "source": source,
    }


def test_quiz_returns_exact_requested_count_without_repair():
    llm = MagicMock()
    llm.complete_chat.return_value = q_json(question(1), question(2))
    gen = QuizGenerator(llm, make_fake_resolver([make_chunk()]))

    result = gen.generate("nb1", "d1", "", count=2, doc_filename="test.pdf")

    assert len(result["questions"]) == 2
    assert result["stats"]["question_count"] == 2
    assert result["stats"]["repair_attempted"] is False
    assert llm.complete_chat.call_count == 1


def test_quiz_trims_over_generated_questions_to_requested_count():
    llm = MagicMock()
    llm.complete_chat.return_value = q_json(question(1), question(2), question(3))
    gen = QuizGenerator(llm, make_fake_resolver([make_chunk()]))

    result = gen.generate("nb1", "d1", "", count=2, doc_filename="test.pdf")

    assert [q["question"] for q in result["questions"]] == ["What is concept 1?", "What is concept 2?"]
    assert result["stats"]["question_count"] == 2


def test_quiz_repairs_under_generated_questions_once():
    llm = MagicMock()
    llm.complete_chat.side_effect = [
        q_json(question(1)),
        q_json(question(2), question(3)),
    ]
    gen = QuizGenerator(llm, make_fake_resolver([make_chunk()]))

    result = gen.generate("nb1", "d1", "", count=3, doc_filename="test.pdf")

    assert len(result["questions"]) == 3
    assert result["stats"]["repair_attempted"] is True
    assert llm.complete_chat.call_count == 2
    repair_prompt = llm.complete_chat.call_args_list[1].kwargs["messages"][1]["content"]
    assert "Generate exactly 2 additional questions" in repair_prompt
    assert "What is concept 1?" in repair_prompt


def test_quiz_prompt_includes_special_requirements():
    llm = MagicMock()
    llm.complete_chat.return_value = q_json(question(1))
    gen = QuizGenerator(llm, make_fake_resolver([make_chunk()]))

    gen.generate(
        "nb1",
        "d1",
        "Chapter 1",
        count=1,
        difficulty="hard",
        requirements="Focus on formulas and common misconceptions.",
    )

    prompt = llm.complete_chat.call_args.kwargs["messages"][1]["content"]
    assert "Special instructions: Focus on formulas and common misconceptions." in prompt
    assert "Generate exactly 1 questions" in prompt
