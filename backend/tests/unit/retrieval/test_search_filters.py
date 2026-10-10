"""Unit tests for SearchFilters — Module E."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.services.retrieval.search_filters import SearchFilters, normalize_page_range


def _mock_chunk(document_id="d1", page_start=1, page_end=2, section_path="Ch1"):
    chunk = MagicMock()
    chunk.document_id = document_id
    chunk.page_start = page_start
    chunk.page_end = page_end
    chunk.section_path = section_path
    return chunk


class TestHasAnyFilter:
    def test_empty_filters(self):
        sf = SearchFilters()
        assert sf.has_any_filter() is False

    def test_page_range_filter(self):
        sf = SearchFilters(page_start=10, page_end=20)
        assert sf.has_any_filter() is True

    def test_document_ids_filter(self):
        sf = SearchFilters(document_ids=("a",))
        assert sf.has_any_filter() is True

    def test_section_paths_filter(self):
        sf = SearchFilters(section_paths=("Ch1",))
        assert sf.has_any_filter() is True


class TestMatches:
    def test_page_in_range(self):
        sf = SearchFilters(page_start=10, page_end=20)
        assert sf.matches(_mock_chunk(page_start=15)) is True

    def test_page_at_start_boundary(self):
        sf = SearchFilters(page_start=10, page_end=20)
        assert sf.matches(_mock_chunk(page_start=10)) is True

    def test_page_at_end_boundary(self):
        sf = SearchFilters(page_start=10, page_end=20)
        assert sf.matches(_mock_chunk(page_start=20, page_end=20)) is True

    def test_page_out_of_range(self):
        sf = SearchFilters(page_start=10, page_end=20)
        assert sf.matches(_mock_chunk(page_start=30, page_end=32)) is False

    def test_page_before_range(self):
        sf = SearchFilters(page_start=10, page_end=20)
        assert sf.matches(_mock_chunk(page_start=5, page_end=5)) is False

    def test_section_path_in_list(self):
        sf = SearchFilters(section_paths=("Ch3 > 3.1",))
        assert sf.matches(_mock_chunk(section_path="Ch3 > 3.1")) is True

    def test_section_path_not_in_list(self):
        sf = SearchFilters(section_paths=("Ch3 > 3.1",))
        assert sf.matches(_mock_chunk(section_path="Ch4")) is False

    def test_combined_filter_and_logic(self):
        sf = SearchFilters(page_start=10, page_end=20, document_ids=("d1",))
        assert sf.matches(_mock_chunk(page_start=15, document_id="d1")) is True
        assert sf.matches(_mock_chunk(page_start=15, document_id="d2")) is False

    def test_document_ids_match(self):
        sf = SearchFilters(document_ids=("doc1", "doc2"))
        assert sf.matches(_mock_chunk(document_id="doc1")) is True
        assert sf.matches(_mock_chunk(document_id="doc3")) is False

    def test_immutability(self):
        sf = SearchFilters(page_start=10, page_end=20)
        with pytest.raises(AttributeError):
            sf.page_start = 15  # type: ignore[misc]


class TestNormalizePageRange:
    def test_normal(self):
        assert normalize_page_range(10, 20) == (10, 20)

    def test_inverted(self):
        assert normalize_page_range(20, 10) == (10, 20)

    def test_none_start(self):
        assert normalize_page_range(None, 20) == (None, 20)

    def test_none_end(self):
        assert normalize_page_range(10, None) == (10, None)

    def test_equal(self):
        assert normalize_page_range(10, 10) == (10, 10)
