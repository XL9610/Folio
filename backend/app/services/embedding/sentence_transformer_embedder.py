"""Local sentence-transformers embedder.

Uses a small, fast model by default (bge-small-en-v1.5).
Model is lazy-loaded on first call — keeps startup fast.
Default device is CPU to avoid CUDA issues on varied hardware.
"""

from __future__ import annotations

import logging
import sys

from ...core.config import config
from .bge_tokenizer import EMBEDDING_MODEL_REVISION

logger = logging.getLogger(__name__)

# Lazy-loaded singleton
_model: object | None = None
_model_name: str = ""
_current_device: str = ""

# Load status, surfaced by GET /api/config/embedding-status and used by the
# frontend to show a "preparing semantic search" state on first run. Transitions:
#   not_loaded -> loading -> ready | error
_LOAD_STATE = "not_loaded"
_LOAD_MODEL = ""
_LOAD_DEVICE = ""
_LOAD_MESSAGE = ""

# bge-small-en-v1.5 (BAAI) query instruction; passages use no prefix.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _load_model(device: str):
    """Load the SentenceTransformer model on the specified device."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model '%s' on device '%s'", config.embedding_model, device)
    # Frozen app: the model is bundled locally, so forbid any network fallback.
    load_kwargs = {
        "device": device,
        "local_files_only": getattr(sys, "frozen", False),
    }
    if not getattr(sys, "frozen", False):
        # Source mode downloads from HF; pin to the bundled commit for parity.
        load_kwargs["revision"] = EMBEDDING_MODEL_REVISION
    model = SentenceTransformer(config.embedding_model, **load_kwargs)
    dim = model.get_sentence_embedding_dimension()
    logger.info("Embedding model loaded successfully (dim=%d, device=%s)", dim, device)
    return model


def _get_model():
    """Return the cached SentenceTransformer model.

    First tries the configured device (default: cpu).
    If that fails with a CUDA error, falls back to cpu automatically.
    """
    global _model, _model_name, _current_device
    global _LOAD_STATE, _LOAD_MODEL, _LOAD_DEVICE, _LOAD_MESSAGE
    target_name = config.embedding_model
    target_device = config.embedding_device
    _LOAD_MODEL = target_name
    _LOAD_DEVICE = target_device

    if _model is not None and _model_name == target_name and _current_device == target_device:
        _LOAD_STATE = "ready"
        return _model  # already loaded

    _LOAD_STATE = "loading"
    _LOAD_MESSAGE = ""
    try:
        _model = _load_model(target_device)
        _model_name = target_name
        _current_device = target_device
        _LOAD_STATE = "ready"
        return _model
    except Exception as exc:
        is_cuda_err = "cuda" in str(exc).lower() or "CUDA" in str(exc)
        if is_cuda_err and target_device != "cpu":
            logger.warning(
                "CUDA loading failed for '%s': %s. Falling back to CPU.",
                target_device, exc,
            )
            _LOAD_DEVICE = "cpu"
            try:
                _model = _load_model("cpu")
                _model_name = target_name
                _current_device = "cpu"
                _LOAD_STATE = "ready"
                return _model
            except Exception as cpu_exc:
                logger.error("CPU fallback also failed: %s", cpu_exc)
                _LOAD_STATE = "error"
                _LOAD_MESSAGE = str(cpu_exc)
                raise
        _LOAD_STATE = "error"
        _LOAD_MESSAGE = str(exc)
        raise


class SentenceTransformerEmbedder:
    """Local embedding via sentence-transformers.

    Implements the Embedder protocol:
      - embed_texts: pass multiple passages for batch encoding
      - embed_query:  single query with query prefix
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode document passages.

        bge-small-en-v1.5 uses no passage prefix (unlike E5 models).
        """
        if not texts:
            return []
        model = _get_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode a search query.

        bge-small-en-v1.5 queries use a retrieval instruction prefix.
        """
        model = _get_model()
        embedding = model.encode(
            _BGE_QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


def get_status() -> dict:
    """Non-blocking load status for the embedding model."""
    return {
        "status": _LOAD_STATE,
        "model": _LOAD_MODEL,
        "device": _LOAD_DEVICE,
        "message": _LOAD_MESSAGE,
    }


def warm_up() -> None:
    """Kick off a background model load if one isn't already in flight.

    Used by the status endpoint so the first search isn't slow and the frontend
    can show a "preparing" state. Runs the actual load in a daemon thread —
    callers must never block on it.
    """
    global _LOAD_STATE, _LOAD_MESSAGE
    if _LOAD_STATE in ("loading", "ready"):
        return
    _LOAD_STATE = "loading"
    _LOAD_MESSAGE = ""

    def _load() -> None:
        try:
            SentenceTransformerEmbedder().embed_query("warmup")
        except Exception as exc:  # noqa: BLE001 — status already records the error
            logger.warning("Embedding model warm-up failed: %s", exc)

    import threading

    threading.Thread(target=_load, daemon=True, name="embedder-warmup").start()
