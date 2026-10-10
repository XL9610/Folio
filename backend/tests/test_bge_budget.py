"""Embedding-window enforcement against the bundled BGE tokenizer.

The chunkers size chunks with cl100k (320 tokens), but bge-small-en-v1.5 counts
WordPiece tokens and truncates at 512. These tests verify the ingestion guard
keeps the final indexed_text within the real model window.
"""

from __future__ import annotations

import pytest

from app.services.embedding.bge_tokenizer import (
    MAX_BGE_INPUT_TOKENS,
    count_bge_tokens,
    truncate_to_budget,
)
from app.services.ingestion.index_text import build_index_text


def _bge_tokenizer_available() -> bool:
    from transformers import AutoTokenizer
    from app.core.config import config

    try:
        AutoTokenizer.from_pretrained(config.embedding_model, local_files_only=True)
        return True
    except Exception:
        return False


_requires_bge = pytest.mark.skipif(
    not _bge_tokenizer_available(),
    reason="BGE tokenizer not cached in this environment",
)


@_requires_bge
def test_dense_punctuation_truncated_to_budget():
    dense = "a+b-c*d/e%f==g?h:i,j,k;\n" * 40
    indexed = build_index_text("Doc", "Code", dense).indexed_text
    fitted, was_truncated = truncate_to_budget(indexed)
    assert count_bge_tokens(fitted) <= MAX_BGE_INPUT_TOKENS
    # Dense punctuation exceeds the cl100k-320 budget in BGE WordPiece tokens.
    assert was_truncated


@_requires_bge
def test_url_and_code_dense_truncated():
    dense = ("https://example.com/very/long/path?query=abc&x=" + "z" * 40 + "\n") * 40
    indexed = build_index_text("Doc", "Refs", dense).indexed_text
    fitted, _ = truncate_to_budget(indexed)
    assert count_bge_tokens(fitted) <= MAX_BGE_INPUT_TOKENS


@_requires_bge
def test_long_metadata_prefix_is_capped():
    title = "A" * 300
    path = " > ".join(f"Section {i}" for i in range(50))
    result = build_index_text(title, path, "short body")
    assert len(result.context_prefix) <= 160
    fitted, _ = truncate_to_budget(result.indexed_text)
    assert count_bge_tokens(fitted) <= MAX_BGE_INPUT_TOKENS


def test_fallback_truncates_dense_text_by_worst_case(monkeypatch):
    """Without a tokenizer, the fallback must over-bound (1 char = 1 token), not
    under-count with char/4 — '!'*1800 is ~1802 WordPiece tokens, not 450."""
    from app.services.embedding import bge_tokenizer as bt

    def no_tokenizer():
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr(bt, "_get_tokenizer", no_tokenizer)

    dense = "!" * 1800
    fitted, was_truncated = bt.truncate_to_budget(dense)
    assert was_truncated is True
    assert len(fitted) <= bt.MAX_BGE_INPUT_TOKENS - 2  # reserves [CLS]/[SEP]
    assert bt.count_bge_tokens(fitted) <= bt.MAX_BGE_INPUT_TOKENS


@_requires_bge
def test_fallback_output_respects_real_tokenizer(monkeypatch):
    """The fallback's own counter is self-referential; cross-check its output
    against the *real* bundled tokenizer so [CLS]/[SEP] are accounted for."""
    from app.services.embedding import bge_tokenizer as bt

    real_get = bt._get_tokenizer

    def no_tokenizer():
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr(bt, "_get_tokenizer", no_tokenizer)
    fitted, was_truncated = bt.truncate_to_budget("!" * 1800)
    monkeypatch.setattr(bt, "_get_tokenizer", real_get)

    assert was_truncated is True
    assert count_bge_tokens(fitted) <= MAX_BGE_INPUT_TOKENS
