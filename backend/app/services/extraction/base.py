"""Content extractor interface — abstract base for all Phase 1 extractors.

Every extractor must:
  - Preserve all structure it can reliably extract.
  - Populate extraction_engine.
  - Return warnings instead of silently dropping structure.
  - Avoid indexing, embedding, or database writes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import ExtractedDocument, SourceInput


class ExtractionError(Exception):
    """Base exception for extraction failures."""


class UnsupportedSourceError(ExtractionError):
    """Raised when a source type or format is not supported by the extractor."""


class ExtractorDependencyError(ExtractionError):
    """Raised when an optional dependency is missing.

    The message must name the missing package and include the install command:
        pip install -r backend/requirements-extractors.txt
    """


class ContentExtractor(ABC):
    """Abstract base for all content extractors.

    Subclasses must set engine_name and implement can_handle / extract.
    """

    engine_name: str

    @abstractmethod
    def can_handle(self, source: SourceInput) -> bool:
        """Return True if this extractor can process the given source."""

    @abstractmethod
    async def extract(self, source: SourceInput) -> ExtractedDocument:
        """Extract content and structure from the source.

        Returns ExtractedDocument with all preserved structure.
        Raises ExtractionError (or subclass) on failure.
        """
