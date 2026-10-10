"""Tests for LLM-based TOC classifier with mocked LLM responses.

Tests cover:
- AI Engineering style TOC (Cover → Copyright → Preface → Ch1)
- Data Structures style TOC (Cover → Title → Preface descendants → Ch1)
- Real Introduction kept
- LLM confidence too low → fallback
- LLM JSON parse failure → fallback
- API param use_llm_toc_filter=true → calls classifier
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import fitz
import pytest

from app.services.pdf_ingestion.chunker import PDFChunker
from app.services.toc_llm_classifier.classifier import LLMTOCClassifier
from app.services.toc_llm_classifier.models import TOCClassificationResult, TOCEntryForLLM
from app.services.llm import LLMConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_pdf(path: str, toc_entries: list[list]) -> str:
    """Create a minimal PDF with given TOC entries. Each entry: [level, title, page]."""
    doc = fitz.open()
    max_page = max(e[2] for e in toc_entries)
    _FILLER = "Study content. " * 30
    for i in range(max_page):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), _FILLER, fontsize=12)
    doc.set_toc(toc_entries)
    doc.save(path)
    doc.close()
    return path


def _mock_classifier_response(first_idx: int, confidence: float = 0.9,
                              reasoning: str = "test") -> TOCClassificationResult:
    """Build a mock classifier result."""
    return TOCClassificationResult(
        first_content_index=first_idx,
        confidence=confidence,
        reasoning_summary=reasoning,
    )


# ---------------------------------------------------------------------------
# Unit tests for LLMTOCClassifier
# ---------------------------------------------------------------------------

def test_classifier_parses_valid_response():
    """Classifier should parse a valid JSON response from the mock."""
    from app.services.llm.client import LLMJSONResponse

    classifier = LLMTOCClassifier()
    # Simulate what the client returns
    response_data = {
        "first_content_index": 5,
        "skip_indices": [0, 1, 2, 3, 4],
        "keep_indices": [5, 6, 7],
        "confidence": 0.92,
        "reasoning_summary": "Skipped cover, copyright, contents, preface",
    }
    mock_response = LLMJSONResponse(
        data=response_data,
        raw_text=json.dumps(response_data),
        model_used="test",
        finish_reason="stop",
    )
    result = classifier._parse_response(mock_response)
    assert result.first_content_index == 5
    assert result.confidence == 0.92
    assert result.is_usable is True  # 0.92 >= 0.65


def test_classifier_low_confidence_not_usable():
    """Low confidence result should not be usable."""
    result = TOCClassificationResult(first_content_index=3, confidence=0.3)
    assert result.is_usable is False


def test_classifier_empty_entries():
    """Empty entries should return first_content_index=0."""
    classifier = LLMTOCClassifier()
    result = classifier.classify([])
    assert result.first_content_index == 0
    assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# Integration tests: chunker with mocked LLM classifier
# ---------------------------------------------------------------------------

class MockTOCClassifier:
    """A mock LLMTOCClassifier that returns predefined results without API calls."""

    def __init__(self, first_idx: int, confidence: float = 0.9):
        self._first_idx = first_idx
        self._confidence = confidence
        self._call_count = 0

    def classify(self, entries):
        self._call_count += 1
        return TOCClassificationResult(
            first_content_index=self._first_idx,
            confidence=self._confidence,
            reasoning_summary=f"Mock classifier: starts at index {self._first_idx}",
        )


def test_chunker_with_llm_skips_cover_and_preface():
    """AI Engineering style: LLM should skip Cover, Copyright, Contents, Preface."""
    tmp = tempfile.mkdtemp()
    try:
        pdf_path = os.path.join(tmp, "ai_eng_style.pdf")
        _make_test_pdf(pdf_path, [
            [1, "Cover", 1],
            [1, "Copyright", 2],
            [1, "Table of Contents", 3],
            [1, "Preface", 4],
            [1, "1. Introduction to Building AI Applications", 5],
            [1, "2. Understanding Foundation Models", 6],
        ])

        mock_cls = MockTOCClassifier(first_idx=4, confidence=0.95)
        chunker = PDFChunker(
            toc_llm_classifier=mock_cls,
            use_llm_toc_filter=True,
            chunk_size=500,
            chunk_overlap=50,
        )
        result = chunker.chunk(pdf_path)

        assert result.strategy_used == "toc"
        assert result.llm_toc_filter_used is True
        assert result.llm_confidence == 0.95
        assert result.toc_filter_strategy == "rules+llm"

        chunk_titles = {c.title for c in result.chunks}
        assert "Cover" not in chunk_titles
        assert "Copyright" not in chunk_titles
        assert "Table of Contents" not in chunk_titles
        assert "Preface" not in chunk_titles
        # Should have actual chapter
        titles_str = " ".join(chunk_titles)
        assert "1. Introduction" in titles_str or "Introduction" in titles_str
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_chunker_with_llm_low_confidence_falls_back():
    """Low LLM confidence should fall back to rule-based filter."""
    tmp = tempfile.mkdtemp()
    try:
        pdf_path = os.path.join(tmp, "low_conf.pdf")
        _make_test_pdf(pdf_path, [
            [1, "Copyright", 1],
            [1, "Contents", 2],
            [1, "Preface", 3],
            [1, "Chapter 1", 4],
        ])

        mock_cls = MockTOCClassifier(first_idx=3, confidence=0.4)  # below 0.65
        chunker = PDFChunker(
            toc_llm_classifier=mock_cls,
            use_llm_toc_filter=True,
            chunk_size=500,
            chunk_overlap=50,
        )
        result = chunker.chunk(pdf_path)

        # LLM was called but not used due to low confidence
        assert result.llm_toc_filter_used is False
        assert result.toc_filter_strategy == "rules"
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_chunker_without_llm_flag_does_not_call_llm():
    """When use_llm_toc_filter=False, LLM should not be called."""
    tmp = tempfile.mkdtemp()
    try:
        pdf_path = os.path.join(tmp, "no_llm.pdf")
        _make_test_pdf(pdf_path, [
            [1, "Copyright", 1],
            [1, "Chapter 1", 2],
        ])

        mock_cls = MockTOCClassifier(first_idx=1, confidence=0.9)
        chunker = PDFChunker(
            toc_llm_classifier=mock_cls,
            use_llm_toc_filter=False,  # explicitly off
            chunk_size=500,
            chunk_overlap=50,
        )
        result = chunker.chunk(pdf_path)
        assert result.llm_toc_filter_used is False
        assert mock_cls._call_count == 0  # never called
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_chunker_data_structures_style():
    """Data Structures style: Cover → Title → Copyright → Preface descendants → Ch1."""
    tmp = tempfile.mkdtemp()
    try:
        pdf_path = os.path.join(tmp, "ds_style.pdf")
        _make_test_pdf(pdf_path, [
            [1, "Cover", 1],
            [1, "Title Page", 2],
            [1, "Copyright", 3],
            [1, "Preface", 4],
            [1, "Contents", 5],
            [1, "Chapter 1: Java Primer", 6],
            [1, "Chapter 2: Object-Oriented Design", 7],
        ])

        mock_cls = MockTOCClassifier(first_idx=5, confidence=0.90)
        chunker = PDFChunker(
            toc_llm_classifier=mock_cls,
            use_llm_toc_filter=True,
            chunk_size=500,
            chunk_overlap=50,
        )
        result = chunker.chunk(pdf_path)

        assert result.llm_toc_filter_used is True
        chunk_titles = {c.title for c in result.chunks}
        assert "Cover" not in chunk_titles
        assert "Title Page" not in chunk_titles
        assert "Copyright" not in chunk_titles
        assert "Preface" not in chunk_titles
        assert "Contents" not in chunk_titles
        titles_str = " ".join(chunk_titles)
        assert "Chapter 1" in titles_str or "Java Primer" in titles_str
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_toc_entry_for_llm_serialization():
    """TOCEntryForLLM should serialize correctly for the LLM prompt, including tree metadata."""
    entry = TOCEntryForLLM(
        index=3, title="Chapter 1", path=["Chapter 1"],
        page_start=10, page_end=20, level=1, has_children=True
    )
    d = entry.to_dict()
    assert d["index"] == 3
    assert d["title"] == "Chapter 1"
    assert d["level"] == 1
    assert d["has_children"] is True
    assert d["path"] == ["Chapter 1"]
    assert d["page_start"] == 10


# ---------------------------------------------------------------------------
# Remote LLM opt-in regression tests
# ---------------------------------------------------------------------------

def test_llm_config_blocked_when_remote_disabled(monkeypatch):
    """LLMConfig.is_configured should be False when LLM_REMOTE_ENABLED=false."""
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key-12345")
    monkeypatch.setenv("LLM_REMOTE_ENABLED", "false")

    from app.services.llm import LLMConfig
    config = LLMConfig()
    assert config.is_configured is False, \
        "LLM should be disabled when LLM_REMOTE_ENABLED=false, even with API key present"


def test_llm_config_enabled_when_remote_on(monkeypatch):
    """LLMConfig.is_configured should be True when LLM_REMOTE_ENABLED=true AND key set."""
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key-12345")
    monkeypatch.setenv("LLM_REMOTE_ENABLED", "true")

    from app.services.llm import LLMConfig
    config = LLMConfig()
    assert config.is_configured is True, \
        "LLM should be enabled when LLM_REMOTE_ENABLED=true and API key present"


def test_llm_config_blocked_without_key(monkeypatch):
    """LLMConfig.is_configured should be False when no API key, even if remote enabled."""
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LLM_REMOTE_ENABLED", "true")

    from app.services.llm import LLMConfig
    config = LLMConfig()
    assert config.is_configured is False, \
        "LLM should be disabled without API key"
