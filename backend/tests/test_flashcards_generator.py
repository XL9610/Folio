"""Tests for FlashcardsGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from app.services.generation.flashcards_service import FlashcardsGenerator


def make_fake_llm(json_text: str):
    llm = MagicMock()
    llm.complete_chat.return_value = json_text
    return llm


def make_fake_resolver(chunks=None, layer="toc"):
    r = MagicMock()
    from app.services.generation.scope_resolver import ScopeResult
    r.resolve.return_value = ScopeResult(
        chunks=chunks or [],
        metadata={"layer": layer},
    )
    return r


def make_chunk(cid="c1", text="test", section_path="Ch 1", page_start=1, page_end=1):
    from app.services.retrieval.base import RetrievedChunk
    return RetrievedChunk(
        chunk_id=cid, document_id="d1", text=text, title="",
        section_path=section_path, page_start=page_start, page_end=page_end,
    )


def test_generate_valid_cards():
    """LLM returns valid JSON — cards should be parsed and enriched."""
    llm = make_fake_llm('{"cards": [{"term": "Antiphon", "definition": "A chant.", "source": 1}]}')
    chunks = [make_chunk("c1", "Antiphon is a chant sung before and after a psalm.", "Ch 1", 84, 84)]
    resolver = make_fake_resolver(chunks, "toc")
    gen = FlashcardsGenerator(llm, resolver)
    result = gen.generate("nb1", "d1", "Chapter 1", count=10, doc_filename="test.pdf")
    assert len(result["cards"]) == 1
    assert result["cards"][0]["term"] == "Antiphon"
    assert result["cards"][0]["source_chunk_id"] == "c1"
    assert result["cards"][0]["page_start"] == 84
    assert result["stats"]["card_count"] == 1
    assert result["stats"]["requested_count"] == 10


def test_parse_cards_with_code_fence():
    """LLM wraps JSON in ```json code fence — parser should strip it."""
    gen = FlashcardsGenerator(MagicMock(), make_fake_resolver())
    raw = '```json\n{"cards": [{"term": "Mode", "definition": "A scale pattern.", "source": 1}]}\n```'
    cards = gen._parse_cards(raw)
    assert len(cards) == 1
    assert cards[0]["term"] == "Mode"


def test_parse_cards_garbage_returns_empty():
    gen = FlashcardsGenerator(MagicMock(), make_fake_resolver())
    cards = gen._parse_cards("total nonsense no json at all")
    assert cards == []


def test_parse_cards_partial_json():
    """Only valid cards with term+definition should survive."""
    gen = FlashcardsGenerator(MagicMock(), make_fake_resolver())
    raw = '{"cards": [{"term": "", "definition": "no term"}, {"term": "Good", "definition": "valid", "source": 1}]}'
    cards = gen._parse_cards(raw)
    assert len(cards) == 1
    assert cards[0]["term"] == "Good"


def test_source_mapping_uses_1_based_index():
    """source:2 should map to chunks[1]."""
    chunks = [
        make_chunk("ca", "chunk A", "A", 1, 1),
        make_chunk("cb", "chunk B", "B", 2, 2),
    ]
    gen = FlashcardsGenerator(MagicMock(), make_fake_resolver(chunks))
    cards_raw = [{"term": "B-term", "definition": "from B", "source": 2}]
    enriched = gen._enrich_cards(cards_raw, chunks)
    assert enriched[0]["source_chunk_id"] == "cb"


def test_empty_chunks_returns_empty_cards():
    gen = FlashcardsGenerator(MagicMock(), make_fake_resolver([]))
    result = gen.generate("nb1", "d1", "", count=5, doc_filename="test.pdf")
    assert result["cards"] == []
    assert result["stats"]["card_count"] == 0


def test_source_out_of_range_graceful():
    """source index beyond chunks should still produce a card (without chunk metadata)."""
    chunks = [make_chunk("c1", "only one chunk", "A", 1, 1)]
    gen = FlashcardsGenerator(MagicMock(), make_fake_resolver(chunks))
    cards_raw = [{"term": "X", "definition": "y", "source": 99}]
    enriched = gen._enrich_cards(cards_raw, chunks)
    assert enriched[0]["term"] == "X"
    assert enriched[0]["source_chunk_id"] == ""


def test_parser_recovers_from_truncated_json():
    """LLM truncated mid-third-card — recover the first two complete cards."""
    gen = FlashcardsGenerator(MagicMock(), make_fake_resolver())
    truncated = ('{"cards": ['
                 '{"term": "A", "definition": "first card", "source": 1},'
                 '{"term": "B", "definition": "second card", "source": 2},'
                 '{"term": "C", "definition": "third')  # truncated mid-string
    cards = gen._parse_cards(truncated)
    assert len(cards) == 2
    assert cards[0]["term"] == "A"
    assert cards[1]["term"] == "B"


def test_parser_handles_code_fence_and_truncation():
    """Truncation inside a code-fenced response."""
    gen = FlashcardsGenerator(MagicMock(), make_fake_resolver())
    raw = ('```json\n'
           '{"cards": [\n'
           '  {"term": "X", "definition": "ok", "source": 1},\n'
           '  {"term": "Y", "definition": "also ok"')  # truncated mid-object
    cards = gen._parse_cards(raw)
    assert len(cards) == 1
    assert cards[0]["term"] == "X"


def test_generate_raises_on_zero_cards():
    """Empty parse result should raise RuntimeError, not return empty deck."""
    from app.services.retrieval.base import RetrievedChunk

    mock_llm = MagicMock()
    mock_llm.complete_chat.return_value = "garbage non-json output"

    mock_resolver = MagicMock()
    mock_resolver.resolve.return_value = make_fake_resolver(
        chunks=[RetrievedChunk(chunk_id="c1", document_id="d1", text="some text",
                               title="", section_path="Ch 1", page_start=1, page_end=1)]
    ).resolve.return_value

    gen = FlashcardsGenerator(mock_llm, mock_resolver)
    with pytest.raises(RuntimeError, match="no valid flashcards"):
        gen.generate(notebook_id="nb", document_id="d1", scope="", count=10, doc_filename="x.pdf")