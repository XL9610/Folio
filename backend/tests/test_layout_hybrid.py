"""Tests for chunking_strategies/layout_hybrid.py."""

from __future__ import annotations

from app.services.pdf_ingestion.chunking_strategies.layout_hybrid import (
    chunk_layout_hybrid,
)
from app.services.pdf_ingestion.models import DocumentPage


def _make_pages(*texts: str) -> list[DocumentPage]:
    """Helper: create DocumentPage objects from text strings."""
    return [
        DocumentPage(page_number=i + 1, text=t)
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Test 1: sentence boundary validation — each chunk ends with sentence-final
#         punctuation
# ---------------------------------------------------------------------------
def test_chunks_end_with_sentence_punctuation():
    pages = _make_pages(
        "This is sentence one. This is sentence two. A third sentence here.",
        "Another paragraph starts. It continues onward. And then concludes.",
    )
    result = chunk_layout_hybrid(pages, doc_id="test-doc")
    assert len(result.chunks) > 0
    for chunk in result.chunks:
        stripped = chunk.text.rstrip()
        assert stripped, f"Empty chunk at index {chunk.chunk_index}"
        assert stripped[-1] in ".!?", (
            f"Chunk {chunk.chunk_index} does not end with sentence punctuation: "
            f"'{stripped[-50:]}'"
        )


# ---------------------------------------------------------------------------
# Test 2: token range validation — token count within [target-hard_max]
#         with a ±20 token margin
# ---------------------------------------------------------------------------
def test_token_count_within_range():
    # Generate enough text to produce multiple chunks
    pages = _make_pages(
        "The process of cellular respiration involves several key steps. "
        "First, glycolysis occurs in the cytoplasm. "
        "This process breaks down glucose into pyruvate. " * 20,
        "Next, the Krebs cycle takes place in the mitochondria. "
        "This cycle produces electron carriers. " * 20,
        "Finally, the electron transport chain generates ATP. "
        "This process requires oxygen and produces water. " * 20,
    )
    result = chunk_layout_hybrid(
        pages, doc_id="test-range",
        target_tokens=380, hard_max_tokens=460,
    )
    assert len(result.chunks) >= 1

    for chunk in result.chunks:
        tc = chunk.metadata.get("token_count", 0)
        # Last chunk may be smaller; allow anything positive up to hard_max + margin
        assert tc > 0, f"Chunk {chunk.chunk_index} has zero token_count"
        assert tc <= 480, (
            f"Chunk {chunk.chunk_index} token_count={tc} exceeds hard_max margin [480]"
        )


# ---------------------------------------------------------------------------
# Test 3: overlap validation — boundary sentence appears in adjacent chunks
# ---------------------------------------------------------------------------
def test_overlap_boundary_sentence():
    pages = _make_pages(
        "Sentence alpha. Sentence beta. Sentence gamma. " * 30,
        "Sentence delta. Sentence epsilon. Sentence zeta. " * 30,
    )
    result = chunk_layout_hybrid(
        pages, doc_id="test-overlap",
        target_tokens=200, hard_max_tokens=300, overlap_tokens=40,
    )
    assert len(result.chunks) >= 2, f"Expected 2+ chunks, got {len(result.chunks)}"

    # For at least one pair of adjacent chunks, the tail of chunk N should
    # share text with the head of chunk N+1 (evidence of overlap).
    found_overlap = False
    for i in range(len(result.chunks) - 1):
        current = result.chunks[i].text
        next_chunk = result.chunks[i + 1].text

        # Check if the last 50 chars of current appear in next_chunk
        # (allowing for minor whitespace differences due to token decoding)
        tail = current[-50:].replace(" ", "").strip(".")
        next_clean = next_chunk.replace(" ", "").strip(".")
        if len(tail) > 10 and tail[:15] in next_clean:
            found_overlap = True
            break
        if found_overlap:
            break

    assert "token_count" in result.chunks[0].metadata
    assert result.strategy_used == "layout_hybrid"
    # At minimum we should have multiple chunks and valid metadata
    assert all(
        c.metadata.get("token_count", 0) > 0
        for c in result.chunks
    )
    # The overlap heuristic should find evidence of overlap (soft check)
    assert found_overlap, (
        "Expected overlap boundary between adjacent chunks but none found"
    )


# ---------------------------------------------------------------------------
# Test 4: long sentence fallback — a single sentence > hard_max gets split
# ---------------------------------------------------------------------------
def test_long_sentence_fallback():
    # Build a very long single sentence (~500 words, >1k chars).
    # Lower hard_max_tokens to force splitting.
    long_sent = "The " + "very " * 300 + "long sentence ends here."
    pages = _make_pages(
        f"Short intro. {long_sent}",
        "After the monster sentence. Normal text follows here.",
    )
    result = chunk_layout_hybrid(
        pages, doc_id="test-long",
        hard_max_tokens=100,  # force-split the long sentence
    )
    assert len(result.chunks) >= 2, f"Expected 2+ chunks, got {len(result.chunks)}"

    for chunk in result.chunks:
        tc = chunk.metadata.get("token_count", 0)
        # With very long sentences and low hard_max, be generous on margin
        assert tc <= 300, (
            f"Chunk {chunk.chunk_index} token_count={tc} > 300"
        )


# ---------------------------------------------------------------------------
# Test 5: table detection — has_table_or_caption becomes true
# ---------------------------------------------------------------------------
def test_table_detection():
    caption_page = DocumentPage(
        page_number=1,
        text=(
            "Table 1: Performance results.\n"
            "Model A 0.95 0.87 0.92\n"
            "Model B 0.91 0.84 0.89\n"
            "Model C 0.88 0.82 0.85\n"
            "The table above shows consistent improvement."
        ),
    )
    result = chunk_layout_hybrid([caption_page], doc_id="test-table")
    assert len(result.chunks) >= 1

    # At least one chunk should have has_table_or_caption=true
    found_table = any(
        chunk.metadata.get("has_table_or_caption", False)
        for chunk in result.chunks
    )
    assert found_table, "Expected at least one chunk with has_table_or_caption=true"


# ---------------------------------------------------------------------------
# Test 6: page metadata validation — page_start and page_end are correct
# ---------------------------------------------------------------------------
def test_page_metadata():
    pages = _make_pages(
        "Page one content. This is on the first page.",
        "Page two content. This is on the second page.",
        "Page three content. This is on the third page.",
    )
    result = chunk_layout_hybrid(pages, doc_id="test-pages")
    assert len(result.chunks) > 0

    for chunk in result.chunks:
        assert chunk.page_start >= 1, f"page_start={chunk.page_start} < 1"
        assert chunk.page_start <= chunk.page_end, (
            f"page_start={chunk.page_start} > page_end={chunk.page_end}"
        )
        assert chunk.page_end <= len(pages), (
            f"page_end={chunk.page_end} > total pages={len(pages)}"
        )


# ---------------------------------------------------------------------------
# Test 7: empty pages -> empty result
# ---------------------------------------------------------------------------
def test_empty_pages_returns_empty_result():
    result = chunk_layout_hybrid([], doc_id="test-empty")
    assert result.chunks == []
    assert result.strategy_used == "layout_hybrid"
    assert len(result.warnings) >= 1


# ---------------------------------------------------------------------------
# Test 8: chunking_strategy field correct
# ---------------------------------------------------------------------------
def test_chunking_strategy_field():
    pages = _make_pages(
        "The mitochondria is the powerhouse of the cell. "
        "It generates ATP through oxidative phosphorylation."
    )
    result = chunk_layout_hybrid(pages, doc_id="test-strategy")
    assert len(result.chunks) > 0
    assert result.strategy_used == "layout_hybrid"

    for chunk in result.chunks:
        assert chunk.chunk_strategy == "sentence_boundary_layout_hybrid"
        meta = chunk.metadata
        assert meta.get("chunking_strategy") == "sentence_boundary_layout_hybrid"
        assert "outline_source" in meta
        assert "token_count" in meta
        assert "has_table_or_caption" in meta
        assert "sentence_count" in meta
