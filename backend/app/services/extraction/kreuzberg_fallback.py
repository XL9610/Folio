"""Kreuzberg fallback extractor — handles unsupported formats as a last resort.

Used only when dedicated extractors cannot handle the source or PDF fallback
conditions are met. Emits an explicit fallback warning.
"""

from __future__ import annotations

from .base import ContentExtractor, ExtractionError, ExtractorDependencyError
from .models import ExtractedDocument, SourceInput

_INSTALL_HINT = "pip install -r backend/requirements-extractors.txt"


class KreuzbergFallbackExtractor(ContentExtractor):
    """Fallback extractor for unsupported formats and text-poor PDFs.

    Returns plain text with warnings. Does not claim to preserve structure.
    Selected by the registry only after dedicated extractors do not match.
    """

    engine_name = "kreuzberg"

    # ------------------------------------------------------------------
    # ContentExtractor interface
    # ------------------------------------------------------------------

    def can_handle(self, source: SourceInput) -> bool:
        """Always returns True as the final fallback for file-backed sources.

        The registry must only select this after dedicated extractors fail.
        """
        return source.file_path is not None

    async def extract(self, source: SourceInput) -> ExtractedDocument:
        file_path = source.file_path
        if not file_path:
            raise ExtractionError("Kreuzberg fallback requires a file_path")

        try:
            import kreuzberg
        except ImportError:
            raise ExtractorDependencyError(
                f"kreuzberg is required for fallback extraction. Install with: {_INSTALL_HINT}"
            ) from None

        try:
            # Force OCR — this is a fallback path for text-poor PDFs,
            # so we need Tesseract regardless of what the auto-detection says.
            from kreuzberg import ExtractionConfig
            config = ExtractionConfig(force_ocr=True)
            result = await kreuzberg.extract_file(file_path, config=config)
            text = result.content.strip() if result and result.content else ""
        except Exception as exc:
            raise ExtractionError(f"Kreuzberg extraction failed: {exc}") from exc

        if not text:
            raise ExtractionError(
                f"No extractable text from {source.filename or file_path} via fallback extractor"
            )

        filename = source.filename or file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        return ExtractedDocument(
            title=filename,
            full_text=text,
            extraction_engine=self.engine_name,
            warnings=[
                "Used fallback extractor; source-native structure may be unavailable."
            ],
            metadata={
                "source_type": source.source_type,
                "original_filename": filename,
                "fallback": True,
            },
        )
