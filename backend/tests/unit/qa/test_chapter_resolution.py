"""Unit tests for chapter resolution — Module D."""

from __future__ import annotations

from app.services.qa.tools import resolve_chapter_page_range


class TestResolveChapterPageRange:
    def test_exact_chapter_20_in_section_path(self):
        sections = [
            {"section_path": "Chapter 20: Sonata Form", "title": "Sonata Form",
             "page_start": 245, "page_end": 260, "chunk_count": 5},
        ]
        result = resolve_chapter_page_range(20, sections)
        assert result == (245, 260)

    def test_prefix_number_match(self):
        sections = [
            {"section_path": "PART > 20. Sonata Form", "title": "20. Sonata Form",
             "page_start": 245, "page_end": 260, "chunk_count": 5},
        ]
        result = resolve_chapter_page_range(20, sections)
        assert result == (245, 260)

    def test_ch_abbreviation(self):
        sections = [
            {"section_path": "Ch. 20: Sonata Form", "title": "Ch. 20: Sonata Form",
             "page_start": 245, "page_end": 260, "chunk_count": 5},
        ]
        result = resolve_chapter_page_range(20, sections)
        assert result == (245, 260)

    def test_chapter_not_found(self):
        sections = [
            {"section_path": "Chapter 10: Basics", "title": "Chapter 10",
             "page_start": 100, "page_end": 120, "chunk_count": 3},
        ]
        result = resolve_chapter_page_range(99, sections)
        assert result is None

    def test_empty_sections_list(self):
        result = resolve_chapter_page_range(5, [])
        assert result is None

    def test_section_with_page_negative_one_skipped(self):
        sections = [
            {"section_path": "Chapter 5", "title": "Chapter 5",
             "page_start": -1, "page_end": 50, "chunk_count": 3},
            {"section_path": "Chapter 6", "title": "Chapter 6",
             "page_start": 51, "page_end": 70, "chunk_count": 5},
        ]
        result = resolve_chapter_page_range(6, sections)
        assert result == (51, 70)

    def test_first_match_returned(self):
        sections = [
            {"section_path": "Ch. 20", "title": "Chapter 20",
             "page_start": 100, "page_end": 120, "chunk_count": 3},
            {"section_path": "Ch. 20 Exercises", "title": "Ch. 20 Exercises",
             "page_start": 121, "page_end": 130, "chunk_count": 2},
        ]
        result = resolve_chapter_page_range(20, sections)
        assert result == (100, 120)

    def test_single_page_chapter(self):
        sections = [
            {"section_path": "Ch. 50", "title": "Ch. 50",
             "page_start": 50, "page_end": 50, "chunk_count": 1},
        ]
        result = resolve_chapter_page_range(50, sections)
        assert result == (50, 50)

    def test_match_in_title_only(self):
        sections = [
            {"section_path": "PART FIVE", "title": "Chapter 20",
             "page_start": 245, "page_end": 260, "chunk_count": 5},
        ]
        result = resolve_chapter_page_range(20, sections)
        assert result == (245, 260)
