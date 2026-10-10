"""Unit tests for LLMReranker."""

from unittest.mock import MagicMock
import pytest
from app.services.retrieval.llm_reranker import LLMReranker, RERANK_PROMPT
from app.services.retrieval.base import RetrievedChunk


def _make_chunk(cid: str, text: str, rrf: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id="doc1",
        text=text,
        title="",
        section_path="",
        page_start=1,
        page_end=1,
        dense_score=0.0,
        fts_score=0.0,
        rrf_score=rrf,
    )


def test_rerank_reorders_by_llm_score():
    """Higher LLM score should rank first, even if RRF score was lower."""
    fake_llm = MagicMock()
    fake_llm.complete_chat.side_effect = [
        '{"score": 30}',
        '{"score": 80}',
    ]
    reranker = LLMReranker(fake_llm, max_concurrency=1)
    chunks = [
        _make_chunk("a", "low relevance text", rrf=0.9),
        _make_chunk("b", "high relevance text", rrf=0.5),
    ]
    out = reranker.rerank("test query", chunks, top_n=2)
    assert out[0].chunk_id == "b"
    assert out[1].chunk_id == "a"


def test_rerank_falls_back_on_total_failure():
    """If all LLM calls fail, return original order."""
    fake_llm = MagicMock()
    fake_llm.complete_chat.side_effect = Exception("boom")
    reranker = LLMReranker(fake_llm, max_concurrency=1)
    chunks = [_make_chunk("a", "x", rrf=0.9), _make_chunk("b", "y", rrf=0.5)]
    out = reranker.rerank("q", chunks, top_n=2)
    assert [c.chunk_id for c in out] == ["a", "b"]


def test_rerank_handles_partial_failure():
    """Some chunks fail to score, others succeed — succeeded ones rank by score, failed get score 0."""
    fake_llm = MagicMock()
    fake_llm.complete_chat.side_effect = [
        '{"score": 75}',
        "garbage no number",
    ]
    reranker = LLMReranker(fake_llm, max_concurrency=1)
    chunks = [_make_chunk("a", "x", rrf=0.5), _make_chunk("b", "y", rrf=0.9)]
    out = reranker.rerank("q", chunks, top_n=2)
    assert out[0].chunk_id == "a"  # score 75 wins
    assert out[1].chunk_id == "b"  # parse failed -> score 0


def test_rerank_preserves_tail_beyond_top_n():
    """Chunks beyond top_n are appended in original order, untouched."""
    fake_llm = MagicMock()
    fake_llm.complete_chat.side_effect = ['{"score": 10}', '{"score": 90}']
    reranker = LLMReranker(fake_llm, max_concurrency=1)
    chunks = [
        _make_chunk("a", "x", rrf=0.9),
        _make_chunk("b", "y", rrf=0.7),
        _make_chunk("c", "z", rrf=0.5),  # beyond top_n=2
    ]
    out = reranker.rerank("q", chunks, top_n=2)
    assert out[2].chunk_id == "c"  # tail preserved


def test_parse_score_robust():
    """LLM may include thinking/extra text — parser should still find score."""
    parse = LLMReranker._parse_score
    assert parse('{"score": 85}') == 85.0
    assert parse('Reasoning: blah blah\n{"score": 42}') == 42.0
    assert parse("Score is 67") == 67.0  # fallback regex
    assert parse("total nonsense") is None
    assert parse('{"score": 150}') == 100.0  # clamped


def test_prompt_contains_critical_rules():
    """Prompt must include the synthesize-friendly scoring guidance."""
    assert "score at least 41" in RERANK_PROMPT
    assert "instances ARE valid evidence" in RERANK_PROMPT

def test_reranker_clamps_concurrency_to_1():
    """Invalid concurrency values should be clamped to at least 1."""
    from unittest.mock import MagicMock
    from app.services.retrieval.llm_reranker import LLMReranker
    fake_llm = MagicMock()
    r = LLMReranker(fake_llm, max_concurrency=0)
    assert r._concurrency == 1
    r2 = LLMReranker(fake_llm, max_concurrency=-5)
    assert r2._concurrency == 1
    r3 = LLMReranker(fake_llm, max_concurrency=10)
    assert r3._concurrency == 10
