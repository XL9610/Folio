"""Extractor registry — deterministic selection of content extractors.

Order: PDF, EPUB, PPTX, DOCX, URL, plain text, Kreuzberg fallback.
First matching dedicated extractor wins. Unsupported file-backed sources resolve to fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import ContentExtractor
from .docx import DocxExtractor
from .epub import EpubExtractor
from .kreuzberg_fallback import KreuzbergFallbackExtractor
from .models import SourceInput, SourceType
from .pdf import PdfTextbookExtractor
from .pptx import PptxExtractor
from .text import PlainTextExtractor
from .url import UrlExtractor


@dataclass(frozen=True)
class ExtractorSelection:
    """Result of registry selection — the chosen extractor and metadata."""

    extractor: ContentExtractor
    engine_name: str
    fallback_reason: str = ""


class ExtractorRegistry:
    """Selects the best ContentExtractor for a given SourceInput.

    Extractors are tried in registration order. The first one whose
    can_handle() returns True wins. If no dedicated extractor matches,
    the fallback extractor is used.
    """

    def __init__(
        self,
        extractors: list[ContentExtractor] | None = None,
        fallback_extractor: ContentExtractor | None = None,
    ) -> None:
        self._extractors = extractors or []
        self._fallback = fallback_extractor or KreuzbergFallbackExtractor()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def select(self, source: SourceInput) -> ExtractorSelection:
        """Return the best ExtractorSelection for *source*.

        PDF fallback is NOT handled here — the orchestrator checks text
        stats after extraction and may re-run through Kreuzberg.
        """
        for extractor in self._extractors:
            if extractor.can_handle(source):
                return ExtractorSelection(
                    extractor=extractor,
                    engine_name=extractor.engine_name,
                )

        # Try fallback
        if self._fallback.can_handle(source):
            return ExtractorSelection(
                extractor=self._fallback,
                engine_name=self._fallback.engine_name,
                fallback_reason="No dedicated extractor matched; using fallback.",
            )

        # Last resort — construct a meaningful error upstream
        return ExtractorSelection(
            extractor=self._fallback,
            engine_name=self._fallback.engine_name,
            fallback_reason="Source has no file_path; only fallback available.",
        )

    def supported_extensions(self) -> set[str]:
        """Return the set of file extensions that dedicated extractors handle.

        Used by the API layer to determine which uploads are supported.
        """
        exts: set[str] = set()
        for ext, source_type in _KNOWN_EXTENSIONS.items():
            test_source = SourceInput(source_type=source_type, filename=f"test{ext}")
            for extractor in self._extractors:
                if extractor is not self._fallback and extractor.can_handle(test_source):
                    exts.add(ext)
                    break
        return exts


# ---------------------------------------------------------------------------
# Default factory
# ---------------------------------------------------------------------------


def build_default_extractor_registry() -> ExtractorRegistry:
    """Create an ExtractorRegistry with the standard Phase 1 extractor order.

    Order: PDF, EPUB, PPTX, DOCX, URL, plain text, Kreuzberg fallback.
    """
    return ExtractorRegistry(
        extractors=[
            PdfTextbookExtractor(),
            EpubExtractor(),
            PptxExtractor(),
            DocxExtractor(),
            UrlExtractor(),
            PlainTextExtractor(),
        ],
        fallback_extractor=KreuzbergFallbackExtractor(),
    )


# ---------------------------------------------------------------------------
# Source type inference
# ---------------------------------------------------------------------------

_KNOWN_EXTENSIONS: dict[str, SourceType] = {
    ".pdf": "pdf",
    ".epub": "epub",
    ".pptx": "pptx",
    ".docx": "docx",
    ".md": "text",
    ".txt": "text",
}


def infer_source_type(
    filename: str = "",
    url: str = "",
    raw_text: str | None = None,
) -> SourceType:
    """Infer the SourceType from available source hints.

    Priority:
      1. URL with http/https → "url"
      2. raw_text without file path → "text"
      3. Known file extension → mapped type
      4. Otherwise → "unknown"
    """
    if url and url.startswith(("http://", "https://")):
        return "url"

    if raw_text is not None and not filename:
        return "text"

    if filename:
        ext = Path(filename).suffix.lower()
        if ext in _KNOWN_EXTENSIONS:
            return _KNOWN_EXTENSIONS[ext]

    return "unknown"
