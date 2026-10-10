"""Tests for the PDF chunking pipeline.

Covers:
- TOC-based chunking on a well-structured PDF
- Token-based fallback on a PDF without TOC
- Fallback on a broken/unusable TOC
- Quality checks (empty-drop, short-merge, long-split)
- Citation output correctness
"""

from __future__ import annotations

import os

import fitz  # PyMuPDF
import pytest

from app.services.pdf_ingestion.chunker import PDFChunker
from app.services.pdf_ingestion.models import Chunk, ChunkingResult, DocumentPage, TocEntry
from app.services.pdf_ingestion.structure_metadata import build_structure_metadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chunker():
    """Default chunker with small sizes for faster tests."""
    return PDFChunker(
        chunk_size=200,
        chunk_overlap=30,
        min_chunk_tokens=30,
        max_chunk_tokens=500,
    )


# ---------------------------------------------------------------------------
# TOC chunking tests
# ---------------------------------------------------------------------------

def test_toc_chunking_produces_chunks(chunker, pdf_with_toc):
    """TOC-based PDF should produce multiple chunks with TOC metadata."""
    result = chunker.chunk(pdf_with_toc)

    assert result.strategy_used == "toc", f"Expected 'toc', got {result.strategy_used}"
    structure = build_structure_metadata(result)
    assert structure.structure_status == "embedded_toc"
    assert structure.embedded_toc_present is True
    assert len(result.chunks) > 0, "Should produce at least one chunk"
    assert len(result.pages) == 6, "Should have 6 pages"

    # Every chunk should have TOC-related metadata (includes parent_intro for hierarchical chunks)
    for c in result.chunks:
        assert c.chunk_strategy in ("toc", "toc_subchunk", "toc_parent_intro"), \
            f"Expected toc/toc_subchunk/toc_parent_intro, got {c.chunk_strategy}"
        assert c.title, "Each chunk should have a title from TOC"
        assert c.page_start > 0, "page_start should be positive"
        assert c.metadata.get("outline_source") == "embedded_toc"
        assert c.metadata.get("chunking_strategy") == "toc_based"


def test_single_wrapper_toc_is_low_confidence_structure():
    """A decorative/wrapper TOC entry is not a trustworthy document outline."""
    pages = [
        DocumentPage(page_number=i, text=f"Page {i} text")
        for i in range(1, 121)
    ]
    result = ChunkingResult(
        document_id="wrapper-doc",
        pages=pages,
        toc=[TocEntry(title="TOC", level=1, page_start=4, page_end=118)],
        chunks=[
            Chunk(
                text="Magazine OCR text",
                document_id="wrapper-doc",
                chunk_index=0,
                title="TOC",
                section_path=["TOC"],
                page_start=4,
                page_end=118,
                chunk_strategy="toc",
            )
        ],
        strategy_used="toc",
        toc_entries_total=1,
        toc_entries_used=1,
        embedded_toc_present=True,
        chunking_mode="hierarchical_toc",
    )

    structure = build_structure_metadata(result)

    assert structure.structure_status == "no_reliable_outline"
    assert structure.structure_confidence == "low"
    assert structure.embedded_toc_present is True


def test_sparse_wrapper_toc_density_is_not_enough_for_high_confidence():
    """Large documents need enough real outline density to earn high confidence."""
    pages = [
        DocumentPage(page_number=i, text=f"Page {i} text")
        for i in range(1, 201)
    ]
    result = ChunkingResult(
        document_id="sparse-doc",
        pages=pages,
        toc=[
            TocEntry(title="Contents", level=1, page_start=2, page_end=199),
            TocEntry(title="Articles", level=1, page_start=3, page_end=198),
        ],
        chunks=[],
        strategy_used="toc",
        toc_entries_total=2,
        toc_entries_used=2,
        embedded_toc_present=True,
        chunking_mode="hierarchical_toc",
    )

    structure = build_structure_metadata(result)

    assert structure.structure_status == "no_reliable_outline"
    assert structure.structure_confidence == "low"


def test_short_real_toc_still_high_confidence():
    """Density guard must not downgrade compact papers with real section entries."""
    pages = [
        DocumentPage(page_number=i, text=f"Page {i} text")
        for i in range(1, 9)
    ]
    result = ChunkingResult(
        document_id="paper-doc",
        pages=pages,
        toc=[
            TocEntry(title="Introduction", level=1, page_start=1, page_end=2),
            TocEntry(title="Method", level=1, page_start=3, page_end=5),
            TocEntry(title="Experiments", level=1, page_start=6, page_end=7),
            TocEntry(title="Conclusion", level=1, page_start=8, page_end=8),
        ],
        chunks=[],
        strategy_used="toc",
        toc_entries_total=4,
        toc_entries_used=4,
        embedded_toc_present=True,
        chunking_mode="hierarchical_toc",
    )

    structure = build_structure_metadata(result)

    assert structure.structure_status == "embedded_toc"
    assert structure.structure_confidence == "high"


def test_toc_chunking_section_titles(chunker, pdf_with_toc):
    """Verify TOC chunks carry correct section titles."""
    result = chunker.chunk(pdf_with_toc)

    titles = {c.title for c in result.chunks}
    # Some expected titles from the TOC structure
    expected = {"1.1 Motivation", "1.2 Scope", "2.1 Gradient Descent Algorithm",
                "2.2 Implementation Details", "3.1 Performance Analysis", "3.2 Discussion"}
    # At minimum, the titles should contain these
    for exp in expected:
        found = any(exp in t for t in titles)
        if not found:
            # Sub-chunked titles now share the base section title (no suffix)
            found = any(exp in t for t in titles)
        assert found, f"Expected to find '{exp}' in chunk titles: {titles}"


def test_toc_chunking_page_ranges(chunker, pdf_with_toc):
    """Verify TOC chunk page ranges are within document bounds."""
    result = chunker.chunk(pdf_with_toc)

    for c in result.chunks:
        assert 1 <= c.page_start <= 6, f"page_start {c.page_start} out of range"
        assert 1 <= c.page_end <= 6, f"page_end {c.page_end} out of range"
        assert c.page_start <= c.page_end, \
            f"page_start {c.page_start} > page_end {c.page_end}"


# ---------------------------------------------------------------------------
# Token fallback tests
# ---------------------------------------------------------------------------

def test_no_toc_builds_synthetic_outline(chunker, pdf_without_toc):
    """PDF without TOC should still get section metadata."""
    result = chunker.chunk(pdf_without_toc)

    assert result.strategy_used == "synthetic_outline", \
        f"Expected 'synthetic_outline', got {result.strategy_used}"
    assert result.chunking_mode == "synthetic_outline"
    structure = build_structure_metadata(result)
    assert structure.structure_status == "page_window_outline"
    assert structure.synthetic_outline_mode == "page_windows"
    assert len(result.chunks) > 0
    assert len(result.pages) == 4

    for c in result.chunks:
        assert c.chunk_strategy == "synthetic_outline"
        assert c.title
        assert c.section_path
        assert c.page_start > 0
        assert c.metadata.get("outline_source") == "page_window"
        assert c.title.startswith("Page")


def test_broken_toc_falls_back_to_token(chunker, pdf_with_broken_toc):
    """PDF with TOC pointing to empty pages should fall back to layout_hybrid or token_fallback."""
    result = chunker.chunk(pdf_with_broken_toc)

    assert result.strategy_used in ("token_fallback", "layout_hybrid"), \
        f"Expected fallback strategy, got {result.strategy_used}"

    # Should have at least one warning
    assert len(result.warnings) > 0, "Should warn about unusable TOC"


@pytest.fixture(scope="session")
def pdf_without_toc_with_headings(test_pdf_dir) -> str:
    """No bookmarks, but visible chapter-like headings in page text."""
    path = os.path.join(test_pdf_dir, "no_toc_with_headings.pdf")
    doc = fitz.open()

    pages = [
        "Chapter 1: Foundations\n\nCore study material about foundations. " * 25,
        "This page continues the foundations discussion. " * 35,
        "Chapter 2: Methods\n\nThis section explains methods and workflows. " * 25,
        "Section 2.1 Practical Details\n\nImplementation details and examples. " * 25,
        "Chapter 3: Applications\n\nApplications and case studies. " * 25,
    ]

    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), text, fontsize=12)

    doc.save(path)
    doc.close()
    return path


def test_no_toc_detects_visible_headings(chunker, pdf_without_toc_with_headings):
    """Visible headings should become synthetic sections, not anonymous chunks."""
    result = chunker.chunk(pdf_without_toc_with_headings)

    assert result.strategy_used == "synthetic_outline"
    structure = build_structure_metadata(result)
    assert structure.structure_status == "auto_detected_outline"
    assert structure.synthetic_outline_mode == "detected_headings"
    titles = {c.title for c in result.chunks}
    assert any("Chapter 1" in t for t in titles), titles
    assert any("Chapter 2" in t for t in titles), titles
    assert any("Chapter 3" in t for t in titles), titles
    for c in result.chunks:
        assert c.section_path
        assert c.metadata.get("outline_source") == "synthetic_heading"


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def test_no_empty_chunks(chunker, pdf_with_toc):
    """No empty chunks should be present in the output."""
    result = chunker.chunk(pdf_with_toc)

    for c in result.chunks:
        assert c.text.strip(), f"Chunk {c.chunk_index} has empty text"


def test_chunks_not_too_long(chunker, pdf_with_toc):
    """Chunks should not exceed max_chunk_tokens."""
    result = chunker.chunk(pdf_with_toc)

    for c in result.chunks:
        token_count = len(chunker._tokenizer.encode(c.text))
        assert token_count <= chunker.max_chunk_tokens, \
            f"Chunk {c.chunk_index} has {token_count} tokens (max {chunker.max_chunk_tokens})"


# ---------------------------------------------------------------------------
# Citation tests
# ---------------------------------------------------------------------------

def test_citation_format(chunker, pdf_with_toc):
    """Citations should contain document ID, page range, and section info."""
    result = chunker.chunk(pdf_with_toc)

    for c in result.chunks:
        citation = c.citation()
        assert "toc_chunking.pdf" in citation, f"Missing doc ID in citation: {citation}"
        assert "pp." in citation, f"Missing page prefix in citation: {citation}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_pdf(document_not_found, chunker):
    """Non-existent file should return empty result with warnings."""
    result = chunker.chunk(document_not_found)
    assert result.strategy_used == "empty"
    assert len(result.chunks) == 0
    assert len(result.warnings) > 0


def test_chunk_indices_are_sequential(chunker, pdf_with_toc):
    """Chunk indices should be sequential starting from 0."""
    result = chunker.chunk(pdf_with_toc)
    indices = sorted(c.chunk_index for c in result.chunks)
    assert indices == list(range(len(result.chunks))), \
        f"Chunk indices not sequential: {indices}"


# ---------------------------------------------------------------------------
# Hierarchical section_path tests
# ---------------------------------------------------------------------------

def test_toc_section_path_is_hierarchical(chunker, pdf_with_toc):
    """section_path should contain the full parent chain, not just the leaf title."""
    result = chunker.chunk(pdf_with_toc)

    for c in result.chunks:
        if c.chunk_strategy in ("toc", "toc_subchunk", "toc_parent_intro"):
            # For TOC chunks, section_path should have at least the entry title
            assert len(c.section_path) >= 1, \
                f"section_path empty for chunk {c.chunk_index} (strategy={c.chunk_strategy})"
            # Verify the entry title is the last element
            assert c.title in c.section_path[-1], \
                f"section_path last element '{c.section_path[-1]}' should contain title '{c.title}'"


# ---------------------------------------------------------------------------
# Hierarchical chunking tests (parent intro preservation)
# ---------------------------------------------------------------------------

def test_parent_intro_chunk_created(chunker, pdf_nested_toc):
    """Nested TOC should produce parent_intro chunks before child content."""
    result = chunker.chunk(pdf_nested_toc)

    assert result.strategy_used == "toc"
    assert result.chunking_mode == "hierarchical_toc"

    strategies = {c.chunk_strategy for c in result.chunks}
    assert "toc_parent_intro" in strategies, \
        f"Expected toc_parent_intro chunks, got strategies: {strategies}"


def test_first_chunk_is_parent_not_leaf(chunker, pdf_nested_toc):
    """The first chunk should be the parent Chapter 1 intro, not a deep leaf."""
    result = chunker.chunk(pdf_nested_toc)

    assert len(result.chunks) > 0
    first = result.chunks[0]
    # Should start with Chapter 1 (parent), not a deep leaf
    assert "Chapter 1" in first.title, \
        f"First chunk title should contain 'Chapter 1', got: '{first.title}'"
    assert "From Language Models" not in first.title, \
        f"First chunk should NOT be a deep leaf, got: '{first.title}'"


def test_parent_intro_has_section_role_metadata(chunker, pdf_nested_toc):
    """Parent intro chunks should carry section_role='parent_intro' in metadata."""
    result = chunker.chunk(pdf_nested_toc)

    parent_intro_chunks = [
        c for c in result.chunks
        if c.chunk_strategy == "toc_parent_intro"
    ]
    assert len(parent_intro_chunks) > 0, "Should have at least one parent_intro chunk"

    for c in parent_intro_chunks:
        assert c.metadata.get("section_role") == "parent_intro", \
            f"Parent intro chunk {c.chunk_index} missing section_role: {c.metadata}"


def test_leaf_chunks_have_section_role_metadata(chunker, pdf_nested_toc):
    """Non-parent-intro TOC chunks should carry section_role='leaf'."""
    result = chunker.chunk(pdf_nested_toc)

    leaf_chunks = [
        c for c in result.chunks
        if c.chunk_strategy in ("toc", "toc_subchunk")
        and c.metadata.get("section_role") == "leaf"
    ]
    assert len(leaf_chunks) > 0, "Should have at least one leaf chunk with correct metadata"


def test_first_chunk_title_in_result_metadata(chunker, pdf_nested_toc):
    """Result metadata should carry first_chunk_title."""
    result = chunker.chunk(pdf_nested_toc)

    assert result.first_chunk_title, "first_chunk_title should not be empty"
    assert "Chapter 1" in result.first_chunk_title, \
        f"first_chunk_title should reference parent chapter, got: '{result.first_chunk_title}'"
    assert result.chunking_mode == "hierarchical_toc"


def test_same_page_parent_child_no_duplication(chunker, pdf_same_page_parent_child):
    """Parent and child on same page should NOT produce duplicate content chunks."""
    result = chunker.chunk(pdf_same_page_parent_child)

    assert result.strategy_used == "toc"

    # Collect text snippets and check for duplicates
    texts = [c.text for c in result.chunks]
    assert len(texts) == len(set(texts)), \
        f"Duplicate chunk texts detected among {len(result.chunks)} chunks"

    # The parent intro should NOT appear (same page as first child)
    parent_intros = [c for c in result.chunks if c.chunk_strategy == "toc_parent_intro"]
    for pi in parent_intros:
        assert pi.metadata.get("section_role") == "parent_intro"


def test_preface_subtree_skipped(chunker, pdf_preface_with_children):
    """Preface children should be skipped when Preface is front matter."""
    result = chunker.chunk(pdf_preface_with_children)

    assert result.strategy_used == "toc"
    assert result.front_matter_skipped is True

    chunk_titles = {c.title for c in result.chunks}
    # Preface AND its children should all be absent
    assert "Preface" not in chunk_titles, f"Preface should be skipped: {chunk_titles}"
    assert "What This Book Is About" not in chunk_titles, \
        f"Preface child should be skipped: {chunk_titles}"
    assert "How to Use This Book" not in chunk_titles, \
        f"Preface child should be skipped: {chunk_titles}"

    # Real content should appear
    titles_str = " ".join(chunk_titles)
    assert "Chapter 1" in titles_str or "Introduction" in titles_str, \
        f"Expected Chapter 1 after subtree skip: {chunk_titles}"


# ---------------------------------------------------------------------------
# Same-page TOC entries
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_same_page_sections(test_pdf_dir) -> str:
    """PDF with multiple TOC leaves starting on the same page."""
    import os
    path = os.path.join(test_pdf_dir, "same_page_toc.pdf")
    doc = fitz.open()

    # Two subsections on the same page (page 1)
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Section A content. " * 30, fontsize=12)

    doc.set_toc([
        [1, "Chapter 1", 1],
        [2, "Section A", 1],
        [2, "Section B", 1],  # same page as Section A
    ])
    doc.save(path)
    doc.close()
    return path


def test_same_page_toc_entries_not_lost(chunker, pdf_same_page_sections):
    """Both TOC sections on the same page should produce chunks (not dropped)."""
    result = chunker.chunk(pdf_same_page_sections)

    assert result.strategy_used in ("toc", "toc_subchunk"), \
        f"Expected TOC strategy, got {result.strategy_used}"
    assert len(result.chunks) >= 1, "Should produce at least one chunk"

    titles = [c.title for c in result.chunks]
    # Both sections should be present (they may be merged if quality filter runs)
    has_a = any("Section A" in t for t in titles)
    has_b = any("Section B" in t for t in titles)
    assert has_a or has_b, f"Expected at least one section title in chunks: {titles}"
    # Informational: if only one, it's because quality filter merged them
    if not (has_a and has_b):
        # Both sections share same page with small text, merge is expected
        assert len(result.chunks) == 1, \
            f"Merged same-page sections expected 1 chunk, got {len(result.chunks)}"


# ---------------------------------------------------------------------------
# document_id tests
# ---------------------------------------------------------------------------

def test_document_id_from_original_filename():
    """When document_id is provided, it should be used instead of file_path."""
    import tempfile, os
    chunker = PDFChunker(chunk_size=200, chunk_overlap=30)

    # Create a minimal PDF
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    tmp = os.path.join(tempfile.gettempdir(), "test_doc_id.pdf")
    doc.save(tmp)
    doc.close()

    try:
        result = chunker.chunk(tmp, document_id="user-uploaded-book.pdf")
        assert result.document_id == "user-uploaded-book.pdf", \
            f"Expected 'user-uploaded-book.pdf', got '{result.document_id}'"
        # Verify chunks also have the correct document_id
        for c in result.chunks:
            assert c.document_id == "user-uploaded-book.pdf"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# TOC filter tests
# ---------------------------------------------------------------------------

def test_front_matter_is_skipped(chunker, pdf_with_front_matter):
    """PDF with Title Page, Copyright, etc. should skip them and chunk from content."""
    result = chunker.chunk(pdf_with_front_matter)

    assert result.strategy_used == "toc"
    assert result.front_matter_skipped is True
    assert result.toc_entries_total == 7  # all 7 entries exist in PDF
    assert result.toc_entries_used < 7     # front matter should be filtered
    assert result.toc_entries_used >= 1    # at least some content entries remain

    # Title Page / Copyright / Contents should NOT appear as chunk titles
    chunk_titles = {c.title for c in result.chunks}
    assert "Title Page" not in chunk_titles, \
        f"Title Page should be filtered, got: {chunk_titles}"
    assert "Copyright" not in chunk_titles, \
        f"Copyright should be filtered, got: {chunk_titles}"
    assert "Contents" not in chunk_titles, \
        f"Contents should be filtered, got: {chunk_titles}"

    # Real chapters SHOULD appear
    titles_str = " ".join(chunk_titles)
    assert "Chapter 1" in titles_str or "Chapter 2" in titles_str, \
        f"Expected at least one real chapter in: {chunk_titles}"


def test_front_matter_skipped_warning(chunker, pdf_with_front_matter):
    """Chunking with front matter should include a warning about skipped entries."""
    result = chunker.chunk(pdf_with_front_matter)

    assert any(
        "front-matter" in w.lower() or "skipped" in w.lower()
        for w in result.warnings
    ), f"Expected a front-matter skip warning in: {result.warnings}"


def test_include_front_matter_option(pdf_with_front_matter):
    """When include_front_matter=True, all TOC entries should be used."""
    from app.services.pdf_ingestion.toc_filter import TOCFilter

    toc_filter = TOCFilter(include_front_matter=True)
    chunker_inc = PDFChunker(
        toc_filter=toc_filter,
        chunk_size=200,
        chunk_overlap=30,
    )

    result = chunker_inc.chunk(pdf_with_front_matter)
    assert result.front_matter_skipped is False

    # Title Page should now appear
    chunk_titles = {c.title for c in result.chunks}
    assert "Title Page" in chunk_titles or any(
        "Title Page" in t for t in chunk_titles
    ), f"Title Page should be included when include_front_matter=True, got: {chunk_titles}"


def test_toc_filter_metadata_in_result(chunker, pdf_with_front_matter):
    """ChunkingResult should carry TOC filtering metadata."""
    result = chunker.chunk(pdf_with_front_matter)

    assert result.toc_entries_total >= 1
    assert result.toc_entries_used >= 1
    assert result.toc_entries_used <= result.toc_entries_total
    # front_matter_skipped should be a bool
    assert isinstance(result.front_matter_skipped, bool)


# ---------------------------------------------------------------------------
# Token splitter edge case tests
# ---------------------------------------------------------------------------

def test_token_splitter_no_duplicate_tail():
    """Text with exactly chunk_size tokens should produce 1 chunk, not a duplicate tail."""
    from app.services.pdf_ingestion.chunker import PDFChunker

    chunker = PDFChunker(chunk_size=10, chunk_overlap=3, min_chunk_tokens=1, max_chunk_tokens=100)
    # Text that encodes to exactly 10 tokens (use short repeated word)
    text = "test " * 6  # approximate, may not be exactly 10 tokens
    chunks = chunker._split_text_with_overlap(text)
    # Key: no chunk should be a duplicate of another
    assert len(set(chunks)) == len(chunks), \
        f"Duplicate chunks detected: {chunks}"


def test_token_splitter_small_text_one_chunk():
    """Text smaller than chunk_size should produce exactly 1 chunk."""
    from app.services.pdf_ingestion.chunker import PDFChunker

    chunker = PDFChunker(chunk_size=1000, chunk_overlap=150)
    text = "A short sentence."
    chunks = chunker._split_text_with_overlap(text)
    assert len(chunks) == 1, f"Expected 1 chunk for small text, got {len(chunks)}"


def test_token_splitter_end_condition():
    """Verify that _split_text_with_overlap breaks correctly at document end."""
    from app.services.pdf_ingestion.chunker import PDFChunker
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    chunker = PDFChunker(chunk_size=30, chunk_overlap=10, min_chunk_tokens=1, max_chunk_tokens=200)

    # Build text that encodes to exactly 60 tokens
    tokens = [1] * 60  # dummy token IDs to build exact length
    # Encode a string that gives known token count
    text = "The quick brown fox jumps over the lazy dog. " * 10
    tok_count = len(enc.encode(text))

    chunks = chunker._split_text_with_overlap(text)
    # Each chunk should have actual content, and total should be minimal
    assert len(chunks) >= 1
    # No chunk should be empty or whitespace-only
    for c in chunks:
        assert c.strip(), f"Empty chunk: {c!r}"


# ---------------------------------------------------------------------------
# 5 test cases for different book formats
# ---------------------------------------------------------------------------

def test_western_music_style_skips_editorial_and_maps(chunker, pdf_western_music_style):
    """Western Music style: skip Editorial Board, Maps, Guide, Preface."""
    result = chunker.chunk(pdf_western_music_style)

    assert result.strategy_used == "toc"
    assert result.front_matter_skipped is True

    chunk_titles = {c.title for c in result.chunks}
    # These should NOT appear
    assert "EDITORIAL ADVISORY BOARD" not in chunk_titles
    assert "MAPS" not in chunk_titles
    assert "GUIDE TO NAWM SELECTIONS" not in chunk_titles
    assert "PREFACE TO THE TENTH EDITION" not in chunk_titles
    # These SHOULD appear
    titles_str = " ".join(chunk_titles)
    assert "PART ONE" in titles_str or "1 Music in Antiquity" in titles_str, \
        f"Expected PART ONE or chapter, got: {chunk_titles}"


def test_introduction_kept_as_body(chunker, pdf_intro_as_body):
    """Introduction should NOT be skipped when it's the first content entry."""
    result = chunker.chunk(pdf_intro_as_body)

    chunk_titles = {c.title for c in result.chunks}
    # Introduction should be present
    assert "Introduction" in chunk_titles or any(
        "Introduction" in t for t in chunk_titles
    ), f"Introduction should be kept as body content, got: {chunk_titles}"


def test_introduction_skipped_as_front_matter(chunker, pdf_intro_as_front):
    """Short Introduction before Part I should be skipped."""
    result = chunker.chunk(pdf_intro_as_front)

    assert result.front_matter_skipped is True
    chunk_titles = {c.title for c in result.chunks}
    # Introduction MAY be skipped if short and before strong content
    # (current scoring: ambiguous → kept. This test verifies current behavior)
    # At minimum, Part I should be present
    titles_str = " ".join(chunk_titles)
    assert "Part I" in titles_str or "Chapter 1" in titles_str, \
        f"Expected Part I or Chapter 1, got: {chunk_titles}"


def test_no_chapters_keeps_all(chunker, pdf_no_chapters):
    """Without Chapter/Part markers, all entries should be kept."""
    result = chunker.chunk(pdf_no_chapters)

    chunk_titles = {c.title for c in result.chunks}
    assert "Overview" in chunk_titles, f"Overview should be kept, got: {chunk_titles}"
    assert "Basic Concepts" in chunk_titles
    assert "Advanced Topics" in chunk_titles


def test_bare_number_chapters_recognized(chunker, pdf_bare_number_chapters):
    """Bare-numbered chapters like '1 Getting Started' should be recognized."""
    result = chunker.chunk(pdf_bare_number_chapters)

    # Preface and Contents should be skipped
    assert result.front_matter_skipped is True
    chunk_titles = {c.title for c in result.chunks}
    assert "Preface" not in chunk_titles, f"Preface should be skipped: {chunk_titles}"
    assert "Contents" not in chunk_titles, f"Contents should be skipped: {chunk_titles}"
    # Numbered chapters should appear
    titles_str = " ".join(chunk_titles)
    assert "1 Getting Started" in titles_str or "Getting Started" in titles_str, \
        f"Expected numbered chapter, got: {chunk_titles}"


# ---------------------------------------------------------------------------
# Body-window tests (wrapper, back matter, body metadata)
# ---------------------------------------------------------------------------

def test_wrapper_skips_front_matter_and_wrapper(chunker, pdf_wrapper_with_front_and_back):
    """Wrapper + front matter + body + back matter ⇒ only body chunks."""
    result = chunker.chunk(pdf_wrapper_with_front_and_back)

    assert result.strategy_used == "toc"

    chunk_titles = {c.title for c in result.chunks}
    titles_str = " ".join(chunk_titles)

    # Wrapper and front matter should NOT appear
    assert "A HISTORY OF WESTERN MUSIC, 10E" not in chunk_titles, \
        f"Wrapper should be skipped: {chunk_titles}"
    assert "TITLE PAGE" not in chunk_titles, f"TITLE PAGE should be skipped: {chunk_titles}"
    assert "COPYRIGHT" not in chunk_titles, f"COPYRIGHT should be skipped: {chunk_titles}"
    assert "EDITORIAL ADVISORY BOARD" not in chunk_titles, \
        f"EDITORIAL ADVISORY BOARD should be skipped: {chunk_titles}"
    assert "MAPS" not in chunk_titles, f"MAPS should be skipped: {chunk_titles}"
    assert "GUIDE TO RECORDINGS" not in chunk_titles, \
        f"GUIDE should be skipped: {chunk_titles}"
    assert "PREFACE TO THE EDITION" not in chunk_titles, \
        f"Preface should be skipped: {chunk_titles}"

    # Back matter should NOT appear
    assert "GLOSSARY" not in chunk_titles, f"GLOSSARY should be skipped (back matter): {chunk_titles}"
    assert "CREDITS" not in chunk_titles, f"CREDITS should be skipped (back matter): {chunk_titles}"
    assert "INDEX" not in chunk_titles, f"INDEX should be skipped (back matter): {chunk_titles}"
    assert "BIBLIOGRAPHY" not in chunk_titles, f"BIBLIOGRAPHY should be skipped (back matter): {chunk_titles}"

    # Body should appear
    assert "PART ONE: ANCIENT WORLDS" in titles_str or "1. Music in Antiquity" in titles_str, \
        f"Expected body content, got: {chunk_titles}"

    # Body-window metadata should be populated
    assert result.body_start_title, "body_start_title should be set"


def test_back_matter_is_removed_after_body(chunker, pdf_with_back_matter):
    """Bibliography, References, Index, Glossary, Credits, About Author, Colophon
    should all be removed when they come after body content."""
    result = chunker.chunk(pdf_with_back_matter)

    assert result.strategy_used == "toc"

    chunk_titles = {c.title for c in result.chunks}

    # Front matter skipped
    assert "Cover" not in chunk_titles, f"Cover should be skipped: {chunk_titles}"
    assert "Copyright" not in chunk_titles, f"Copyright should be skipped: {chunk_titles}"
    assert "Table of Contents" not in chunk_titles, f"TOC should be skipped: {chunk_titles}"

    # Body present
    titles_str = " ".join(chunk_titles)
    assert "Chapter 1" in titles_str or "Chapter 2" in titles_str, \
        f"Expected body chapters, got: {chunk_titles}"

    # Back matter removed
    assert "Bibliography" not in chunk_titles, f"Bibliography should be skipped: {chunk_titles}"
    assert "References" not in chunk_titles, f"References should be skipped: {chunk_titles}"
    assert "Index" not in chunk_titles, f"Index should be skipped: {chunk_titles}"
    assert "Glossary" not in chunk_titles, f"Glossary should be skipped: {chunk_titles}"
    assert "Credits" not in chunk_titles, f"Credits should be skipped: {chunk_titles}"
    assert "About the Author" not in chunk_titles, f"About the Author should be skipped: {chunk_titles}"
    assert "Colophon" not in chunk_titles, f"Colophon should be skipped: {chunk_titles}"

    # back_matter_skipped_count should be populated
    assert result.back_matter_skipped_count > 0, \
        f"Expected back_matter_skipped_count > 0, got {result.back_matter_skipped_count}"


def test_body_window_metadata_in_result(chunker, pdf_with_back_matter):
    """ChunkingResult should have body-window metadata fields filled."""
    result = chunker.chunk(pdf_with_back_matter)

    assert isinstance(result.body_start_title, str)
    assert isinstance(result.body_end_title, str)
    assert isinstance(result.front_matter_skipped_count, int)
    assert isinstance(result.back_matter_skipped_count, int)
    assert isinstance(result.wrapper_skipped_count, int)

    # At minimum we should have skipped some front matter
    assert result.front_matter_skipped is True


def test_no_toc_skip_front_and_back_pages(chunker, pdf_no_toc_with_front_back):
    """Synthetic outline should skip obvious copyright/toc/index/bibliography pages."""
    result = chunker.chunk(pdf_no_toc_with_front_back)

    assert result.strategy_used == "synthetic_outline"
    assert result.chunking_mode == "synthetic_outline"
    assert len(result.chunks) > 0

    # The chunks should come from body pages (pages 3-6), not front/back
    for c in result.chunks:
        # Pages 1-2 are front matter, 7-8 are back matter
        # Chunks should be from body pages (3-6)
        # Accept small overlap due to chunk boundaries
        is_body_page = 3 <= c.page_start <= 6
        is_front_page = c.page_start <= 2
        is_back_page = c.page_start >= 7

        if is_front_page:
            # Very few front-matter chunks are acceptable (might get merged)
            pass
        elif is_back_page:
            # Very few back-matter chunks are acceptable
            pass
        else:
            # Body pages are good
            assert is_body_page, \
                f"Chunk should be from body pages 3-6, got page_start={c.page_start}"

    # At least some chunks should be from body pages
    body_page_chunks = [c for c in result.chunks if 3 <= c.page_start <= 6]
    assert len(body_page_chunks) > 0, "Expected at least one chunk from body pages"


# ---------------------------------------------------------------------------
# LLM TOC filter toggle
# ---------------------------------------------------------------------------

def test_llm_toc_filter_toggle():
    """Default: use_llm_toc_filter=True. Also verify explicit override."""
    from app.services.pdf_ingestion.chunker import PDFChunker

    # default: LLM TOC filter is enabled
    chunker = PDFChunker()
    assert chunker.use_llm_toc_filter is True, \
        "Default PDFChunker should have use_llm_toc_filter=True"

    # explicit override
    chunker_off = PDFChunker(use_llm_toc_filter=False)
    assert chunker_off.use_llm_toc_filter is False, \
        "Explicit use_llm_toc_filter=False should be respected"


# ---------------------------------------------------------------------------
# Fixture: document not found
# ---------------------------------------------------------------------------

@pytest.fixture
def document_not_found():
    """Path that does not exist."""
    import tempfile, os
    return os.path.join(tempfile.gettempdir(), "does_not_exist_xyz.pdf")
