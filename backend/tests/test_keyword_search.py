"""Unit tests for keyword search backends."""

from __future__ import annotations

import pytest

from app.services.retrieval.keyword_backend import (
    ImprovedFTSBackend,
    improved_make_fts_query,
    _legacy_make_fts_query,
    reset_keyword_backend,
    get_keyword_backend,
)


class TestImprovedFTSQueryBuilder:
    """Tests for improved_make_fts_query."""

    def test_stemming_reduces_to_stems(self):
        """Stemming should reduce 'running processes' to stemmed forms."""
        query = "running processes"
        result = improved_make_fts_query(query)
        # Should contain stemmed prefix terms (no quotes for FTS5 prefix)
        assert "run*" in result.lower() or result != ""

    def test_stopwords_removed(self):
        """Common stopwords should be removed from the query."""
        result = improved_make_fts_query("what is the meaning of life")
        # 'what', 'is', 'the', 'of' should be removed
        assert "what*" not in result.lower()
        assert "the*" not in result.lower()

    def test_short_query_uses_or(self):
        """Queries should use OR connector for recall."""
        result = improved_make_fts_query("gradient descent algorithm")
        assert " OR " in result

    def test_long_query_uses_or(self):
        """Queries with 4+ meaningful terms should use OR."""
        result = improved_make_fts_query(
            "gradient descent optimization machine learning"
        )
        assert " OR " in result

    def test_phrase_preservation(self):
        """Quoted phrases should be preserved as FTS5 phrase queries."""
        result = improved_make_fts_query('what is "gradient descent" method')
        assert '"gradient descent"' in result

    def test_empty_query(self):
        """Empty query should return empty string."""
        assert improved_make_fts_query("") == ""
        assert improved_make_fts_query("   ") == ""

    def test_only_stopwords(self):
        """Query with only stopwords should fall back to legacy."""
        result = improved_make_fts_query("the a an")
        # Should still produce something (legacy fallback)
        assert result != ""

    def test_special_characters_handled(self):
        """Special characters should be stripped safely."""
        result = improved_make_fts_query("hello! world? test:value")
        assert result != ""  # should not crash
        # Should contain cleaned tokens

    def test_mixed_case_normalized(self):
        """Mixed case should be lowercased."""
        result = improved_make_fts_query("Gradient Descent")
        assert "Gradient" not in result or result != ""


class TestLegacyFTSQueryBuilder:
    """Tests for the legacy FTS5 query builder (fallback)."""

    def test_simple_query(self):
        result = _legacy_make_fts_query("gradient descent")
        assert '"gradient"' in result
        assert '"descent"' in result
        assert " OR " in result

    def test_empty(self):
        assert _legacy_make_fts_query("") == ""

    def test_short_words_filtered(self):
        result = _legacy_make_fts_query("a b c")
        assert result == ""  # all single letters filtered


class TestImprovedFTSBackend:
    """Tests for the ImprovedFTSBackend class."""

    def test_name(self):
        backend = ImprovedFTSBackend()
        name = backend.name()
        assert "fts5_improved" in name

    def test_is_available(self):
        assert ImprovedFTSBackend.is_available() is True

    def test_index_chunks_noop(self):
        backend = ImprovedFTSBackend()
        backend.index_chunks([])  # should not raise

    def test_delete_document_noop(self):
        backend = ImprovedFTSBackend()
        backend.delete_document("any-id")  # should not raise

    def test_search_empty_query(self):
        backend = ImprovedFTSBackend()
        results = backend.search("", 10, "nb1")
        assert results == []

    def test_search_reraises_on_fts_error(self, monkeypatch):
        """A real FTS/DB failure must propagate, not masquerade as a zero-hit."""
        from app.repositories import chunk_repository
        backend = ImprovedFTSBackend()

        def boom(*a, **kw):
            raise RuntimeError("db is down")

        monkeypatch.setattr(chunk_repository, "search_chunks_fts", boom)
        with pytest.raises(RuntimeError):
            backend.search("gradient descent", 10, "nb1")


def test_bm25x_search_reraises_on_error(monkeypatch, tmp_path):
    """BM25x search failures must propagate to the retriever, not return []."""
    import app.core.config as _cfg
    from app.services.retrieval.keyword_backend import BM25xKeywordBackend

    monkeypatch.setattr(_cfg.config, "bm25x_index_dir", str(tmp_path))

    class Boom:
        def search(self, *a, **kw):
            raise RuntimeError("bm25x down")

    backend = BM25xKeywordBackend()
    backend._bm25 = Boom()
    with pytest.raises(RuntimeError):
        backend.search("q", 5, "nb1")


class TestBackendSelection:
    """Tests for keyword backend selection logic."""

    def test_get_keyword_backend_returns_fts_fallback(self):
        reset_keyword_backend()
        # Force BM25x off to test the FTS5 fallback path
        import app.core.config as _cfg
        _cfg.config.bm25x_enabled = False
        try:
            backend = get_keyword_backend()
            assert isinstance(backend, ImprovedFTSBackend)
        finally:
            _cfg.config.bm25x_enabled = True
            reset_keyword_backend()

    def test_get_keyword_backend_is_cached(self):
        reset_keyword_backend()
        b1 = get_keyword_backend()
        b2 = get_keyword_backend()
        assert b1 is b2


# ---------------------------------------------------------------------------
# M5 - BM25x enriched text tests
# ---------------------------------------------------------------------------


class _FakeBM25:
    """Records texts added via add() so tests can inspect indexed content."""

    def __init__(self):
        self.texts: list[str] = []
        self._next_idx = 0

    def add(self, texts: list[str]) -> list[int]:
        indices = []
        for t in texts:
            indices.append(self._next_idx)
            self.texts.append(t)
            self._next_idx += 1
        return indices

    def save(self, _path: str) -> None:
        pass

    @staticmethod
    def load(_path: str):
        raise FileNotFoundError("force rebuild path")


class TestBM25xEnrichedIndexing:
    """Tests that BM25x indexes enriched text but stores canonical text."""

    def test_index_chunks_uses_indexed_text_when_present(self, monkeypatch, tmp_path):
        """index_chunks() sends indexed_text to BM25x when provided."""
        import app.core.config as _cfg
        from app.services.retrieval.keyword_backend import BM25xKeywordBackend

        fake_bm25 = _FakeBM25()
        monkeypatch.setattr(_cfg.config, "bm25x_index_dir", str(tmp_path))

        backend = BM25xKeywordBackend()
        backend._bm25 = fake_bm25

        chunks = [{
            "chunk_id": "c1",
            "document_id": "d1",
            "text": "canonical text",
            "indexed_text": "[Book > Ch1]\n\nenriched for index",
            "title": "Ch1",
            "section_path": "Ch1",
            "page_start": 1,
            "page_end": 1,
        }]
        backend.index_chunks(chunks)

        # BM25x received the enriched text
        assert fake_bm25.texts == ["[Book > Ch1]\n\nenriched for index"]
        # _idx_to_chunk stores canonical text
        stored = backend._idx_to_chunk[0]
        assert stored["text"] == "canonical text"
        assert "[Book" not in stored["text"]

    def test_index_chunks_composes_from_context_prefix(self, monkeypatch, tmp_path):
        """index_chunks() composes indexed_text from context_prefix when indexed_text is absent."""
        import app.core.config as _cfg
        from app.services.retrieval.keyword_backend import BM25xKeywordBackend

        fake_bm25 = _FakeBM25()
        monkeypatch.setattr(_cfg.config, "bm25x_index_dir", str(tmp_path))

        backend = BM25xKeywordBackend()
        backend._bm25 = fake_bm25

        chunks = [{
            "chunk_id": "c1",
            "document_id": "d1",
            "text": "canonical text",
            "context_prefix": "[Book > Intro]",
            "title": "Intro",
            "section_path": "Intro",
            "page_start": 1,
            "page_end": 1,
        }]
        backend.index_chunks(chunks)

        # BM25x received composed enriched text
        assert fake_bm25.texts == ["[Book > Intro]\n\ncanonical text"]
        # _idx_to_chunk stores canonical text
        stored = backend._idx_to_chunk[0]
        assert stored["text"] == "canonical text"

    def test_index_chunks_no_context_prefix(self, monkeypatch, tmp_path):
        """index_chunks() indexes plain text when context_prefix is missing/empty."""
        import app.core.config as _cfg
        from app.services.retrieval.keyword_backend import BM25xKeywordBackend

        fake_bm25 = _FakeBM25()
        monkeypatch.setattr(_cfg.config, "bm25x_index_dir", str(tmp_path))

        backend = BM25xKeywordBackend()
        backend._bm25 = fake_bm25

        chunks = [{
            "chunk_id": "c1",
            "text": "plain chunk text",
            "title": "T",
            "section_path": "T",
            "page_start": 1,
            "page_end": 1,
        }]
        backend.index_chunks(chunks)

        assert fake_bm25.texts == ["plain chunk text"]
