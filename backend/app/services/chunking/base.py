"""Document chunker interface — abstract base for all Phase 1 chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..extraction.models import ExtractedDocument
from ..pdf_ingestion.models import ChunkingResult


class DocumentChunker(ABC):
    """Abstract base for chunkers that consume ExtractedDocument.

    Subclasses must set strategy_name and implement can_chunk / chunk.
    """

    strategy_name: str

    @abstractmethod
    def can_chunk(self, document: ExtractedDocument) -> bool:
        """Return True if this chunker can process the given document."""

    @abstractmethod
    def chunk(self, document: ExtractedDocument, document_id: str) -> ChunkingResult:
        """Chunk the extracted document and return a ChunkingResult."""
