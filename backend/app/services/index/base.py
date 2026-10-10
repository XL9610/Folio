"""Abstract interface for dense vector storage and search.

Implementations:
  - SqliteVecStore (default, via sqlite-vec)
  - Future: FAISS, Pinecone, etc.
"""

from __future__ import annotations

from typing import Any, Protocol


class DenseIndexStore(Protocol):
    """Protocol: dense vector index for RAG retrieval.

    Each chunk is stored with a unique ID plus payload metadata
    (document_id, notebook_id, title, pages, etc.).
    """

    def ensure_collection(self, vector_size: int) -> None:
        """Create or verify the collection exists, with given vector dimensions.

        Called once on startup / first use.
        """
        ...

    def upsert_chunks(
        self, chunks: list[dict[str, Any]], vectors: list[list[float]]
    ) -> None:
        """Insert or update chunks with their embedding vectors.

        Args:
            chunks: List of chunk payload dicts. Must include:
                - chunk_id
                - document_id
                - notebook_id
                - title
                - section_path
                - page_start
                - page_end
                - chunk_strategy
            vectors: Corresponding embedding vectors (same order).
        """
        ...

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all vector points belonging to a document.

        Used when a source document is removed, to keep the dense index
        consistent with SQLite.
        """
        ...

    def search(
        self,
        query_vector: list[float],
        notebook_id: str | None,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for nearest neighbors to query_vector.

        Args:
            query_vector: The query embedding.
            notebook_id: Optional filter — only return chunks in this notebook.
            top_k: Max results to return.
            document_ids: Optional restrict results to these documents.

        Returns:
            List of dicts with at least: chunk_id, document_id, score, plus
            all original payload fields.
        """
        ...
