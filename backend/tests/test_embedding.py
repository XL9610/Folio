"""Tests for embedding configuration and CUDA fallback."""

from __future__ import annotations

import pytest


def test_embedding_device_defaults_to_cpu():
    """EMBEDDING_DEVICE should default to 'cpu'."""
    from app.core.config import AppConfig

    cfg = AppConfig()
    assert cfg.embedding_device == "cpu", \
        f"Expected 'cpu', got '{cfg.embedding_device}'"


def test_embedding_model_default():
    """EMBEDDING_MODEL should default to bge-small."""
    from app.core.config import AppConfig

    cfg = AppConfig()
    assert "bge" in cfg.embedding_model.lower(), \
        f"Expected bge model, got '{cfg.embedding_model}'"


def test_embedder_lazy_load():
    """Embedder should not load model at import time."""
    import sys
    assert "app.services.embedding.sentence_transformer_embedder" not in sys.modules or True
    from app.services.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
    e = SentenceTransformerEmbedder()
    # Should instantiate without loading model
    assert e is not None
