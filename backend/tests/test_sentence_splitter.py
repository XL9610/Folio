"""Tests for sentence_splitter.py — pure regex sentence splitting."""

from __future__ import annotations

import pytest

from app.services.pdf_ingestion.sentence_splitter import (
    split_long_sentence,
    split_sentences,
)


# ---------------------------------------------------------------------------
# Test 1: normal 2-sentence split
# ---------------------------------------------------------------------------
def test_split_two_sentences():
    sentences = split_sentences("Hello world. This is a test.")
    assert len(sentences) == 2
    assert sentences[0] == "Hello world."
    assert sentences[1] == "This is a test."


# ---------------------------------------------------------------------------
# Test 2: Dr. abbreviation not split
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "arg0,arg1,arg2,arg3",
    [
        ("Dr. Smith arrived. The patient waited.", "Dr. Smith arrived.", "The patient waited.", 1),
        ("The U.S. and U.K. agreed. The treaty was signed.", "U.S.", "U.K.", 0),
    ],
    ids=['dr_abbreviation_not_split', 'us_uk_abbreviations'],
)
def test_dr_abbreviation_not_split(arg0, arg1, arg2, arg3):
    text = arg0
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert arg1 in sentences[0]
    assert arg2 in sentences[arg3]


# ---------------------------------------------------------------------------
# Test 3: e.g. / Fig. complex abbreviations
# ---------------------------------------------------------------------------
def test_complex_abbreviations():
    text = "Something, e.g. an apple, was found. Fig. 3 shows the result. The study concluded."
    sentences = split_sentences(text)
    assert len(sentences) >= 3
    combined = " ".join(sentences)
    assert "e.g." in combined
    assert "Fig. 3" in combined


# ---------------------------------------------------------------------------
# Test 5: J. R. R. Tolkien initials
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        ("J. R. R. Tolkien wrote the book. It became famous.", "J. R. R. Tolkien"),
        ("The price is $3.14 per item. That is cheap.", "$3.14"),
    ],
    ids=['jrr_tolkien_initials', 'decimal_protection'],
)
def test_jrr_tolkien_initials(value, expected):
    text = value
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert expected in sentences[0]


# ---------------------------------------------------------------------------
# Test 7: semicolon/colon separation
# ---------------------------------------------------------------------------
def test_semicolon_colon_separation():
    text = "The experiment succeeded: the hypothesis was confirmed; however, more data is needed."
    sentences = split_sentences(text)
    assert len(sentences) >= 2


# ---------------------------------------------------------------------------
# Test 8: very long sentence (>800 chars) -> split_long_sentence
# ---------------------------------------------------------------------------
def test_split_long_sentence_fallback():
    # Build a 900-character sentence by repeating a clause
    clause = "The quick brown fox jumps over the lazy dog while the sun sets. "

    long_sentence = (clause * 40).strip()  # ~850+ chars
    assert len(long_sentence) > 600, f"Expected >600 chars, got {len(long_sentence)}"

    pieces = split_long_sentence(long_sentence, max_chars=600)
    assert len(pieces) >= 2, f"Expected 2+ pieces, got {len(pieces)}"
    for piece in pieces:
        assert len(piece) <= 600, f"Piece length {len(piece)} exceeds 600"
        assert len(piece) > 0, "Got empty piece"


# ---------------------------------------------------------------------------
# Test 9: empty input -> empty list
# ---------------------------------------------------------------------------
def test_empty_input_returns_empty_list():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    assert split_sentences("\n\n") == []
