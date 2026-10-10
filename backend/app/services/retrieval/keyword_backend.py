"""Keyword search backends for hybrid retrieval.

Protocol KeywordBackend:
  - search(query, top_k, notebook_id, document_ids) -> list[dict]
  - index_chunks(chunks) -> None
  - delete_document(document_id) -> None
  - name() -> str
  - is_available() -> bool

Implementations:
  - BM25xKeywordBackend: SIRA BM25x Rust library (stemming, n-grams, query parsing)
  - ImprovedFTSBackend:   SQLite FTS5 with improved query preprocessing
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Protocol

from ...core.config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class KeywordBackend(Protocol):
    """Protocol for keyword search backends."""

    def search(
        self,
        query: str,
        top_k: int,
        notebook_id: str,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None: ...

    def delete_document(self, document_id: str) -> None: ...

    @staticmethod
    def name() -> str: ...

    @staticmethod
    def is_available() -> bool: ...


# ---------------------------------------------------------------------------
# Improved FTS5 Backend
# ---------------------------------------------------------------------------

_ENGLISH_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "you", "your",
    "he", "she", "it", "they", "we", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "also", "into", "over", "after", "before", "between",
    "under", "again", "further", "here", "there", "each", "both", "few",
    "more", "most", "other", "some", "such", "only", "own", "same",
    "any", "all", "every", "one", "two", "three", "now", "up", "out",
}

_stemmer: object | None = None


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            from nltk.stem import SnowballStemmer
            _stemmer = SnowballStemmer("english")
        except ImportError:
            _stemmer = False
        except LookupError:
            _stemmer = False
    return _stemmer if _stemmer is not False else None


def _stem(word: str) -> str:
    stemmer = _get_stemmer()
    if stemmer is None:
        return word
    try:
        return stemmer.stem(word)
    except Exception:
        return word


def _legacy_make_fts_query(query: str) -> str:
    """Original simple FTS5 query builder (fallback)."""
    words = query.strip().split()
    if not words:
        return ""
    cleaned = []
    for w in words:
        w = "".join(c for c in w if c.isalnum() or c in ("-", "_"))
        if w and len(w) > 1:
            cleaned.append(f'"{w}"')
    if not cleaned:
        return ""
    return " OR ".join(cleaned)


def improved_make_fts_query(query: str) -> str:
    """Build an improved FTS5 query with stemming and stopword removal.

    Strategy:
    1. Extract quoted phrases (preserve as FTS5 phrase queries)
    2. Lowercase and tokenize remaining words
    3. Remove English stopwords
    4. Stem remaining terms (Snowball)
    5. For short queries (<=3 meaningful terms): AND the stemmed prefix terms
    6. For longer queries: OR the stemmed prefix terms
    7. Combine phrase queries with term queries
    """
    if not query or not query.strip():
        return ""

    # Extract quoted phrases
    phrases = re.findall(r'"([^"]+)"', query)
    unquoted = re.sub(r'"[^"]+"', " ", query).strip()

    words = re.findall(r"[a-zA-Z]{2,}", unquoted.lower())
    meaningful = [w for w in words if w not in _ENGLISH_STOPWORDS]

    if not meaningful and not phrases:
        return _legacy_make_fts_query(query)

    stemmed_unique: list[str] = []
    seen: set[str] = set()
    for w in meaningful:
        s = _stem(w)
        if s not in seen:
            seen.add(s)
            stemmed_unique.append(s)

    if not stemmed_unique and not phrases:
        return _legacy_make_fts_query(query)

    parts: list[str] = []

    # Phrase terms
    for phrase in phrases:
        parts.append(f'"{phrase}"')

    # Stemmed terms with FTS5 prefix matching (no quotes — quotes make it literal)
    # Only prefix terms that are safe (alphanumeric) for FTS5 syntax
    stem_terms = []
    for s in stemmed_unique:
        if re.match(r"^[a-zA-Z0-9]{2,}$", s):
            stem_terms.append(f"{s}*")  # FTS5 prefix: run* matches run, running, runs
        else:
            stem_terms.append(f'"{s}"')  # fallback: quoted exact match

    if stem_terms:
        parts.extend(stem_terms)
        # Use OR for recall (precision comes from BM25 ranking + RRF fusion)
        connector = " OR "
    else:
        connector = " OR "

    if parts:
        return connector.join(parts)
    return ""


class ImprovedFTSBackend:
    """Keyword search via SQLite FTS5 with improved query preprocessing.

    Uses stemming, stopword removal, and adaptive AND/OR logic.
    Always available — no external dependencies beyond SQLite.
    """

    @staticmethod
    def name() -> str:
        stemmer = _get_stemmer()
        suffix = "+stem" if stemmer else ""
        return f"fts5_improved{suffix}"

    @staticmethod
    def is_available() -> bool:
        return True

    def search(
        self,
        query: str,
        top_k: int,
        notebook_id: str,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        from ...repositories import chunk_repository

        safe_query = improved_make_fts_query(query)
        if not safe_query:
            return []

        try:
            rows = chunk_repository.search_chunks_fts(
                notebook_id, safe_query, top_k, document_ids
            )
            return [
                {
                    "chunk_id": r["id"],
                    "document_id": r["document_id"],
                    "text": r["text"],
                    "title": r["title"],
                    "section_path": r["section_path"],
                    "page_start": r["page_start"],
                    "page_end": r["page_end"],
                    "score": -float(r.get("rank", 1.0)),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.warning("Improved FTS search failed: %s", exc)
            raise

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        pass  # FTS5 auto-syncs via SQLite triggers

    def delete_document(self, document_id: str) -> None:
        pass  # FTS5 triggers handle cleanup


# ---------------------------------------------------------------------------
# BM25x Backend (SIRA Rust library)
# ---------------------------------------------------------------------------


class BM25xKeywordBackend:
    """Keyword search via SIRA BM25x Rust library.

    Uses unicode_stem tokenizer (NFKD normalization + Snowball stemming),
    configurable n-grams, and BM25 Lucene scoring.

    Falls back to ImprovedFTSBackend if the bm25x package is not installed.
    """

    def __init__(self):
        self._bm25: Any = None
        self._chunk_id_to_idx: dict[str, int] = {}
        self._idx_to_chunk: dict[int, dict[str, Any]] = {}
        self._index_path = os.path.join(config.bm25x_index_dir, "bm25x_index")

    @staticmethod
    def name() -> str:
        return (
            f"bm25x_{config.bm25x_method}_n{config.bm25x_max_n}_"
            f"k{config.bm25x_k1}_b{config.bm25x_b}"
        )

    @staticmethod
    def is_available() -> bool:
        try:
            import bm25x  # type: ignore # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_index(self) -> bool:
        if self._bm25 is not None:
            return True
        return self._load_or_build()

    def _load_or_build(self) -> bool:
        try:
            from bm25x import BM25  # type: ignore
        except ImportError:
            logger.warning("bm25x package not installed, cannot use BM25x backend")
            return False

        if os.path.exists(self._index_path):
            try:
                self._bm25 = BM25.load(self._index_path)
                self._rebuild_mappings()
                logger.info(
                    "Loaded BM25x index from %s (%d docs)",
                    self._index_path, self._bm25.n_features,
                )
                return True
            except Exception as exc:
                logger.warning("Failed to load BM25x index, rebuilding: %s", exc)

        return self._build_from_db()

    def _build_from_db(self) -> bool:
        try:
            from bm25x import BM25  # type: ignore
        except ImportError:
            return False

        from ..ingestion.index_text import compose_indexed_text

        from ...repositories import chunk_repository
        from ...repositories import document_repository

        os.makedirs(config.bm25x_index_dir, exist_ok=True)

        # Collect all completed documents' chunks
        self._bm25 = BM25(
            method=config.bm25x_method,
            k1=config.bm25x_k1,
            b=config.bm25x_b,
            max_n=config.bm25x_max_n,
            tokenizer="unicode_stem",
            use_stopwords=True,
        )

        self._chunk_id_to_idx = {}
        self._idx_to_chunk = {}
        texts: list[str] = []
        chunk_ids: list[str] = []

        # Walk through all notebooks to gather chunks
        from ...repositories.notebook_repository import list_notebooks
        notebooks = list_notebooks()
        for nb in notebooks:
            docs = document_repository.list_documents(nb["id"])
            for doc in docs:
                chunks = chunk_repository.get_chunks_by_document(doc["id"])
                for c in chunks:
                    prefix = c.get("context_prefix", "")
                    texts.append(compose_indexed_text(prefix, c["text"]))
                    chunk_ids.append(c["id"])

        if texts:
            indices = self._bm25.add(texts)
            for i, cid in enumerate(chunk_ids):
                idx = indices[i]
                self._chunk_id_to_idx[cid] = idx
                self._idx_to_chunk[idx] = {"chunk_id": cid}

            self._bm25.save(self._index_path)
            logger.info(
                "Built BM25x index: %d chunks, saved to %s",
                len(texts), self._index_path,
            )
        else:
            logger.info("BM25x index: no chunks to index yet")
            self._bm25.save(self._index_path)

        return True

    def _rebuild_mappings(self) -> None:
        from ...repositories import chunk_repository
        from ...repositories import document_repository
        from ...repositories.notebook_repository import list_notebooks

        self._chunk_id_to_idx = {}
        self._idx_to_chunk = {}
        idx = 0
        notebooks = list_notebooks()
        for nb in notebooks:
            docs = document_repository.list_documents(nb["id"])
            for doc in docs:
                chunks = chunk_repository.get_chunks_by_document(doc["id"])
                for c in chunks:
                    self._chunk_id_to_idx[c["id"]] = idx
                    self._idx_to_chunk[idx] = {
                        "chunk_id": c["id"],
                        "document_id": c["document_id"],
                        "text": c["text"],
                        "title": c.get("title", ""),
                        "section_path": c.get("section_path", ""),
                        "page_start": c.get("page_start", -1),
                        "page_end": c.get("page_end", -1),
                    }
                    idx += 1

    def search(
        self,
        query: str,
        top_k: int,
        notebook_id: str,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._ensure_index() or self._bm25 is None:
            return []

        try:
            # Determine which BM25x indices to restrict to
            subset: list[int] | None = None
            if document_ids:
                subset = []
                for cid, idx in self._chunk_id_to_idx.items():
                    if self._idx_to_chunk.get(idx, {}).get("document_id") in document_ids:
                        subset.append(idx)

            results = self._bm25.search(query, k=top_k, subset=subset)
            output: list[dict[str, Any]] = []
            for doc_idx, score in results:
                meta = self._idx_to_chunk.get(doc_idx, {})
                if not meta:
                    continue
                output.append({
                    "chunk_id": meta["chunk_id"],
                    "document_id": meta.get("document_id", ""),
                    "text": meta.get("text", ""),
                    "title": meta.get("title", ""),
                    "section_path": meta.get("section_path", ""),
                    "page_start": meta.get("page_start", -1),
                    "page_end": meta.get("page_end", -1),
                    "score": float(score),
                })
            return output
        except Exception as exc:
            logger.warning("BM25x search failed: %s", exc)
            raise

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if self._bm25 is None:
            return
        try:
            from ..ingestion.index_text import compose_indexed_text

            texts: list[str] = []
            for c in chunks:
                if "indexed_text" in c:
                    texts.append(c["indexed_text"])
                else:
                    prefix = c.get("context_prefix", "")
                    text = c.get("text", "")
                    texts.append(compose_indexed_text(prefix, text))

            indices = self._bm25.add(texts)
            for c, idx in zip(chunks, indices):
                self._chunk_id_to_idx[c["chunk_id"]] = idx
                self._idx_to_chunk[idx] = {
                    "chunk_id": c["chunk_id"],
                    "document_id": c.get("document_id", ""),
                    "text": c.get("text", ""),
                    "title": c.get("title", ""),
                    "section_path": c.get("section_path", ""),
                    "page_start": c.get("page_start", -1),
                    "page_end": c.get("page_end", -1),
                }
            self._bm25.save(self._index_path)
        except Exception as exc:
            logger.warning("BM25x index_chunks failed: %s", exc)

    def delete_document(self, document_id: str) -> None:
        if self._bm25 is None:
            return
        try:
            indices_to_delete = [
                idx
                for idx, meta in self._idx_to_chunk.items()
                if meta.get("document_id") == document_id
            ]
            if indices_to_delete:
                self._bm25.delete(indices_to_delete)
                for idx in indices_to_delete:
                    cid = self._idx_to_chunk[idx]["chunk_id"]
                    self._chunk_id_to_idx.pop(cid, None)
                    self._idx_to_chunk.pop(idx, None)
                self._bm25.save(self._index_path)
        except Exception as exc:
            logger.warning("BM25x delete_document failed: %s", exc)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_backend: KeywordBackend | None = None


def get_keyword_backend() -> KeywordBackend:
    global _backend
    if _backend is not None:
        return _backend

    if config.bm25x_enabled and BM25xKeywordBackend.is_available():
        _backend = BM25xKeywordBackend()
        logger.info("Keyword backend: %s", _backend.name())
    else:
        if config.bm25x_enabled:
            logger.info(
                "BM25x not available (package not installed), falling back to improved FTS5"
            )
        _backend = ImprovedFTSBackend()

    return _backend


def reset_keyword_backend() -> None:
    """Reset the cached backend (useful for testing)."""
    global _backend
    _backend = None
