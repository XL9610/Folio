"""Abstract interface for text embedding.

Implementations:
  - SentenceTransformerEmbedder (local, default)
  - Future: remote API embedders
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Protocol: text → vector embeddings.

    Implementations must be safe to call from multiple threads
    (e.g., sentence-transformers is thread-safe after model load).
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple document passages into dense vectors."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Encode a single search query into a dense vector."""
        ...
