"""Tests for body-aware chunk sampling."""

from __future__ import annotations

import pytest

from app.services.generation.sampling import (
    SectionGroup,
    allocate_flashcard_budget,
    group_chunks_by_major_section,
    is_body_chunk,
    major_section_key,
    sample_body_chunks,
    sample_section_chunks,
)
from app.services.retrieval.base import RetrievedChunk


def make_chunk(title="", section_path="", text="test"):
    return RetrievedChunk(
        chunk_id="c",
        document_id="d1",
        text=text,
        title=title,
        section_path=section_path,
        page_start=1,
        page_end=1,
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Copyright", False),
        ("INDEX", False),
        ("Bibliography", False),
        ("Glossary", False),
        ("Credits", False),
        ("About the Author", False),
        ("Colophon", False),
        ("Preface", False),
        ("Table of Contents", False),
        ("Title Page", False),
        ("Chapter 1", True),
        ("PART ONE", True),
        ("15. The Madrigal", True),
        ("", True),
        ("Notes", False),
        ("Biographical Notes", False),
        ("List of Illustrations", False),
        ("List of Texts", False),
    ],
    ids=['is_body_chunk_returns_false_for_copyright', 'is_body_chunk_returns_false_for_index', 'is_body_chunk_returns_false_for_bibliography', 'is_body_chunk_returns_false_for_glossary', 'is_body_chunk_returns_false_for_credits', 'is_body_chunk_returns_false_for_about_the_author', 'is_body_chunk_returns_false_for_colophon', 'is_body_chunk_returns_false_for_preface', 'is_body_chunk_returns_false_for_table_of_contents', 'is_body_chunk_returns_false_for_title_page', 'is_body_chunk_returns_true_for_chapter', 'is_body_chunk_returns_true_for_part', 'is_body_chunk_returns_true_for_numbered_section', 'is_body_chunk_returns_true_for_empty_title', 'is_body_chunk_returns_false_for_notes', 'is_body_chunk_returns_false_for_biographical_notes', 'is_body_chunk_returns_false_for_list_of_illustrations', 'is_body_chunk_returns_false_for_list_of_texts'],
)
def test_is_body_chunk_returns(value, expected):
    c = make_chunk(title=value)
    assert is_body_chunk(c) is expected


@pytest.mark.parametrize(
    "arg0,arg1,arg2",
    [
        ("Some Content", "Bibliography", False),
        ("Terms", "Appendix > Glossary", False),
        ("Efficient Indexing", "Chapter 5 > Efficient Indexing Algorithms", True),
    ],
    ids=['is_body_chunk_checks_section_path', 'is_body_chunk_checks_section_path_glossary', 'is_body_chunk_does_not_exclude_indexing_body_section'],
)
def test_is_body_chunk(arg0, arg1, arg2):
    c = make_chunk(title=arg0, section_path=arg1)
    assert is_body_chunk(c) is arg2


# ── Sampling tests ──────────────────────────────────────────────────────


def test_sample_body_chunks_filters_non_body():
    """Front/back matter chunks should be excluded."""
    chunks = [
        make_chunk("Copyright", "Copyright"),
        make_chunk("Table of Contents", "TOC"),
        make_chunk("Preface", "Preface"),
        make_chunk("Chapter 1", "Ch 1"),
        make_chunk("Chapter 2", "Ch 2"),
        make_chunk("Chapter 3", "Ch 3"),
        make_chunk("Index", "Index"),
        make_chunk("Bibliography", "Bibliography"),
        make_chunk("Glossary", "Glossary"),
    ]
    sampled = sample_body_chunks(chunks, max_chunks=25)
    titles = {c.title for c in sampled}

    assert "Copyright" not in titles
    assert "Table of Contents" not in titles
    assert "Preface" not in titles
    assert "Index" not in titles
    assert "Bibliography" not in titles
    assert "Glossary" not in titles

    assert "Chapter 1" in titles
    assert "Chapter 2" in titles
    assert "Chapter 3" in titles


def test_sample_body_chunks_distributes_by_section():
    """Should sample from multiple sections, not just first ones."""
    chunks = []
    for i in range(10):
        chunks.append(make_chunk(f"Ch {i}", f"Part A > Section {i}"))
    for i in range(10):
        chunks.append(make_chunk(f"Ch {i+10}", f"Part B > Section {i}"))
    for i in range(10):
        chunks.append(make_chunk(f"Ch {i+20}", f"Part C > Section {i}"))

    sampled = sample_body_chunks(chunks, max_chunks=15)
    titles = {c.title for c in sampled}

    # Should have chunks from multiple top-level sections
    has_a = any("Ch " in t and int(t.split()[-1]) < 10 for t in titles)
    has_b = any("Ch " in t and 10 <= int(t.split()[-1]) < 20 for t in titles)
    has_c = any("Ch " in t and int(t.split()[-1]) >= 20 for t in titles)

    assert has_a, f"Should sample from Part A, got {titles}"
    assert has_b or has_c, "Should sample from at least two different parts"


def test_sample_body_chunks_empty_returns_empty():
    assert sample_body_chunks([]) == []


def test_sample_body_chunks_small_list_returns_all():
    chunks = [make_chunk("Chapter 1", "Ch 1") for _ in range(3)]
    sampled = sample_body_chunks(chunks, max_chunks=25)
    assert len(sampled) == 3


def test_sample_body_chunks_all_non_body_fallsback():
    """If all chunks are non-body, should return them (safety fallback)."""
    chunks = [
        make_chunk("Index", "Index"),
        make_chunk("Bibliography", "Bibliography"),
    ]
    sampled = sample_body_chunks(chunks, max_chunks=25)
    assert len(sampled) == 2


def test_sample_body_chunks_fair_across_documents():
    """Two documents must both be represented, not just the first."""
    chunks = [
        RetrievedChunk(
            chunk_id=f"a{i}", document_id="doc-a", text="x", title=f"A{i}",
            section_path=f"Part A > {i}", page_start=i, page_end=i,
        )
        for i in range(30)
    ] + [
        RetrievedChunk(
            chunk_id=f"b{i}", document_id="doc-b", text="x", title=f"B{i}",
            section_path=f"Part B > {i}", page_start=i, page_end=i,
        )
        for i in range(10)
    ]
    sampled = sample_body_chunks(chunks, max_chunks=15)
    doc_ids = {c.document_id for c in sampled}
    assert "doc-a" in doc_ids and "doc-b" in doc_ids


def test_sample_body_chunks_spreads_within_document():
    """Within one document, sampling must spread across the whole text position."""
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}", document_id="doc", text="x", title=f"S{i}",
            section_path=f"Section {i}", page_start=i, page_end=i,
        )
        for i in range(100)
    ]
    sampled = sample_body_chunks(chunks, max_chunks=10)
    assert len(sampled) == 10
    positions = sorted(c.page_start for c in sampled)
    assert positions[0] < 20       # near the beginning
    assert positions[-1] > 80      # near the end, not just the head


# ── Section grouping tests ────────────────────────────────────────────────


def test_group_chunks_by_major_section():
    chunks = [
        make_chunk("Ch 1.1", "Chapter 1 > Section 1.1", "text"),
        make_chunk("Ch 1.2", "Chapter 1 > Section 1.2", "text"),
        make_chunk("Ch 2.1", "Chapter 2 > Section 2.1", "text"),
        make_chunk("Ch 2.2", "Chapter 2 > Section 2.2", "text"),
        make_chunk("Ch 3.1", "Chapter 3 > Section 3.1", "text"),
    ]
    groups = group_chunks_by_major_section(chunks)
    assert len(groups) == 3
    assert groups[0].key == "Chapter 1"
    assert groups[0].chunk_weight == 2
    assert groups[1].key == "Chapter 2"
    assert groups[2].key == "Chapter 3"


def test_group_chunks_excludes_non_body():
    chunks = [
        make_chunk("Index", "Index", "text"),
        make_chunk("Chapter 1", "Chapter 1", "text"),
        make_chunk("Notes", "Notes", "text"),
    ]
    groups = group_chunks_by_major_section(chunks)
    assert len(groups) == 1
    assert groups[0].key == "Chapter 1"


def test_sample_section_chunks_picks_beginning_middle_end():
    chunks = [
        make_chunk(f"chunk{i}", f"Ch > Sub{i//3}", f"text{i}")
        for i in range(20)
    ]
    section = SectionGroup(key="Ch", chunks=chunks, page_start=1, page_end=20, chunk_weight=20)
    sampled = sample_section_chunks(section, max_chunks=5)
    assert len(sampled) <= 5
    assert len(sampled) >= 2
    # First and last should be present
    assert chunks[0] in sampled
    assert chunks[-1] in sampled


def test_allocate_budget_proportional():
    sections = [
        SectionGroup(key="Ch1", chunks=[], page_start=1, page_end=50, chunk_weight=50),
        SectionGroup(key="Ch2", chunks=[], page_start=51, page_end=100, chunk_weight=50),
        SectionGroup(key="Ch3", chunks=[], page_start=101, page_end=150, chunk_weight=20),
    ]
    budget = allocate_flashcard_budget(sections, 30)
    assert budget["Ch1"] >= 6
    assert budget["Ch2"] >= 6
    assert budget["Ch3"] >= 3
    assert sum(budget.values()) == 30


def test_allocate_budget_minimum_one_per_section():
    sections = [
        SectionGroup(key="Ch1", chunks=[], page_start=1, page_end=10, chunk_weight=10),
        SectionGroup(key="Ch2", chunks=[], page_start=11, page_end=20, chunk_weight=5),
        SectionGroup(key="Ch3", chunks=[], page_start=21, page_end=30, chunk_weight=5),
    ]
    budget = allocate_flashcard_budget(sections, 10)
    # Every section should get at least 1 if total >= sections
    assert budget.get("Ch1", 0) >= 1
    assert budget.get("Ch2", 0) >= 1
    assert budget.get("Ch3", 0) >= 1


def test_allocate_budget_caps_per_section():
    sections = [
        SectionGroup(key="Big", chunks=[], page_start=1, page_end=100, chunk_weight=100),
        SectionGroup(key="Small", chunks=[], page_start=101, page_end=110, chunk_weight=10),
        SectionGroup(key="Medium", chunks=[], page_start=111, page_end=130, chunk_weight=30),
    ]
    budget = allocate_flashcard_budget(sections, 30)
    # No section should get an extreme share (more than ceil(30*0.6) = 18)
    max_allowed = 18  # generous cap for 3-section test
    for k, v in budget.items():
        assert v <= max_allowed, f"Section {k} got {v}, max allowed {max_allowed}"
    # The largest section should not dominate
    assert budget.get("Big", 0) <= 15  # Big has weight 100/140 = 71%, but capped


def test_section_path_component_matching():
    """Only exact path components should be flagged, not substrings."""
    # "Index" as a section path component should be flagged
    c_bad = make_chunk(title="Topics", section_path="Index")
    assert is_body_chunk(c_bad) is False

    # "Indexing" as a section path component should NOT be flagged
    c_ok = make_chunk(title="Efficient Indexing", section_path="Chapter 5 > Efficient Indexing")
    assert is_body_chunk(c_ok) is True


# ── Major section key tests ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Chapter 1 Art as History > Introduction", "Chapter 1 Art as History"),
        ("Introduction", "Introduction"),
    ],
    ids=['major_section_key_uses_chapter', 'major_section_key_fallback_top_level'],
)
def test_major_section_key(value, expected):
    assert major_section_key(value) == expected


@pytest.mark.parametrize(
    "arg0,arg1,arg2",
    [
        ("PART ONE > 1. Music in Antiquity", "1. Music in Antiquity", "PART ONE"),
        ("PART TWO > 7. Music and the Renaissance", "7. Music and the Renaissance", "PART TWO"),
        ("Part I: Foundations > Chapter 1: Basics", "Chapter 1", "Part I: Foundations"),
    ],
    ids=['major_section_key_uses_numbered_chapter', 'major_section_key_uses_simple_number', 'major_section_key_uses_chapter_with_parent'],
)
def test_major_section_key_uses(arg0, arg1, arg2):
    # Parent Part context is included when it's a structure label
    result = major_section_key(arg0)
    assert arg1 in result
    # Groups by chapter, not by Part
    assert result != arg2


def test_grouping_uses_major_section_key():
    """Chunks with PART > chapter paths should group by chapter, not PART."""
    chunks = [
        make_chunk("text", "PART ONE > 1. Music in Antiquity", "text"),
        make_chunk("text", "PART ONE > 2. The Christian Church", "text"),
        make_chunk("text", "PART TWO > 7. Music and the Renaissance", "text"),
    ]
    groups = group_chunks_by_major_section(chunks)
    # Should have 3 groups (one per chapter), not 2 (one per PART)
    assert len(groups) == 3


def test_head_bias_always_includes_head_chunks():
    """Branch C: overview evidence is front-loaded (cover/TOC/agenda)."""
    chunks = [make_chunk(title=f"Head {i}") for i in range(3)]
    chunks += [make_chunk(title=f"Body {i}") for i in range(27)]
    sampled = sample_body_chunks(chunks, max_chunks=10, include_front_matter=True, head_bias=3)
    titles = [c.title for c in sampled]
    assert "Head 0" in titles
    assert "Head 1" in titles
    assert "Head 2" in titles
    assert len(sampled) == 10
