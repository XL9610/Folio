"""Application configuration loaded from environment variables.

All config values have sensible local-first defaults.
Remote services (LLM, etc.) must be explicitly enabled.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .env_file import ENV_PATH, user_data_dir

# Auto-load .env (dev: backend/.env; frozen: %APPDATA%/Folio/.env)
if ENV_PATH.exists():
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)


_FROZEN = getattr(sys, "frozen", False)

if _FROZEN:
    # PyInstaller bundle: data lives in a stable per-user directory, not next to
    # the (possibly read-only or replaced) executable.
    _PROJECT_ROOT = user_data_dir()
    # The embedding model is bundled into the app, so never reach out to
    # HuggingFace for it — offline installs must work. Force HF libs offline.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    # Dev: anchored to this file's location on disk (repo root).
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_DEFAULT_LOCAL_DATA = _PROJECT_ROOT / "local-data"


def _default_embedding_model() -> str:
    """Bundled local model path when frozen, the HF repo id in dev."""
    if _FROZEN:
        return str(Path(getattr(sys, "_MEIPASS", "")) / "embedding_model")
    return "BAAI/bge-small-en-v1.5"


@dataclass
class AppConfig:
    """Centralised application configuration.

    Defaults:
      - SQLite for metadata store
      - SQLite (sqlite-vec) for dense vector storage
      - Local sentence-transformers embedding
      - Remote LLM disabled by default

    Override via environment variables or .env file.
    """

    # -- Local data directory -----------------------------------------------
    local_data_dir: str = field(
        default_factory=lambda: os.getenv(
            "LOCAL_DATA_DIR", str(_DEFAULT_LOCAL_DATA)
        )
    )

    # -- Database -----------------------------------------------------------
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            f"sqlite:///{os.getenv('LOCAL_DATA_DIR', str(_DEFAULT_LOCAL_DATA))}/rag_companion.db"
        )
    )

    # -- Embedding ----------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", _default_embedding_model()
        )
    )
    embedding_device: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_DEVICE", "cpu"
        )
    )

    # -- Reranker (opt-in) -------------------------------------------------
    rerank_enabled: bool = field(
        default_factory=lambda: os.getenv("RERANK_ENABLED", "false").lower() == "true"
    )
    rerank_top_n: int = field(
        default_factory=lambda: int(os.getenv("RERANK_TOP_N", "20"))
    )
    rerank_concurrency: int = field(
        default_factory=lambda: int(os.getenv("RERANK_CONCURRENCY", "10"))
    )
    rerank_max_doc_chars: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_DOC_CHARS", "2000"))
    )

    # -- Retrieval profile --------------------------------------------------
    retrieval_profile: str = field(
        default_factory=lambda: os.getenv("RETRIEVAL_PROFILE", "baseline")
    )

    chunking_llm_toc_filter: bool = field(
        default_factory=lambda: os.getenv("CHUNKING_LLM_TOC_FILTER", "true").lower() == "true"
    )

    # -- BM25x Keyword Search -----------------------------------------------
    # Default off: ImprovedFTS5 is the deterministic, reproducible default.
    # BM25x is opt-in and requires a separately installed `bm25x` wheel
    # (see ADR 0001 — its Rust source is not vendored).
    bm25x_enabled: bool = field(
        default_factory=lambda: os.getenv("BM25X_ENABLED", "false").lower() == "true"
    )
    bm25x_index_dir: str = field(
        default_factory=lambda: os.getenv(
            "BM25X_INDEX_DIR",
            f"{os.getenv('LOCAL_DATA_DIR', str(_DEFAULT_LOCAL_DATA))}/bm25x_data"
        )
    )
    bm25x_method: str = field(
        default_factory=lambda: os.getenv("BM25X_METHOD", "lucene")
    )
    bm25x_k1: float = field(
        default_factory=lambda: float(os.getenv("BM25X_K1", "1.5"))
    )
    bm25x_b: float = field(
        default_factory=lambda: float(os.getenv("BM25X_B", "0.75"))
    )
    bm25x_max_n: int = field(
        default_factory=lambda: int(os.getenv("BM25X_MAX_N", "2"))
    )


# Singleton config instance
config = AppConfig()
