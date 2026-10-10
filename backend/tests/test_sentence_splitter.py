"""Tests for sentence_splitter.py — pure regex sentence splitting."""

from __future__ import annotations

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
def test_dr_abbreviation_not_split():
    text = "Dr. Smith arrived. The patient waited."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "Dr. Smith arrived." in sentences[0]
    assert "The patient waited." in sentences[1]


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
# Test 4: U.S. / U.K. abbreviations
# ---------------------------------------------------------------------------
def test_us_uk_abbreviations():
    text = "The U.S. and U.K. agreed. The treaty was signed."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "U.S." in sentences[0]
    assert "U.K." in sentences[0]


# ---------------------------------------------------------------------------
# Test 5: J. R. R. Tolkien initials
# ---------------------------------------------------------------------------
def test_jrr_tolkien_initials():
    text = "J. R. R. Tolkien wrote the book. It became famous."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "J. R. R. Tolkien" in sentences[0]


# ---------------------------------------------------------------------------
# Test 6: decimal number protection
# ---------------------------------------------------------------------------
def test_decimal_protection():
    text = "The price is $3.14 per item. That is cheap."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "$3.14" in sentences[0]


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
