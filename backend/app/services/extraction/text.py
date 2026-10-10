"""Plain text / Markdown extractor — handles .txt, .md, and pasted raw text."""

from __future__ import annotations

from pathlib import Path

from .base import ContentExtractor, ExtractionError
from .models import ExtractedDocument, SourceInput


class PlainTextExtractor(ContentExtractor):
    """Extract plain text or Markdown without inventing structure.

    - .md files: markdown is preserved as-is.
    - .txt files: only full_text, no markdown.
    - Pasted text: treated as plain text with display name "pasted text".
    """

    engine_name = "plain_text"

    # ------------------------------------------------------------------
    # ContentExtractor interface
    # ------------------------------------------------------------------

    def can_handle(self, source: SourceInput) -> bool:
        if source.source_type == "text":
            return True
        if source.raw_text is not None:
            return True
        ext = Path(source.filename or source.file_path or "").suffix.lower()
        return ext in (".txt", ".md")

    async def extract(self, source: SourceInput) -> ExtractedDocument:
        if source.raw_text is not None:
            text = source.raw_text
        elif source.file_path:
            text = Path(source.file_path).read_text(encoding="utf-8", errors="replace")
        else:
            raise ExtractionError("PlainTextExtractor requires raw_text or file_path")

        if not text.strip():
            raise ExtractionError("No text content in source")

        is_markdown = _looks_like_markdown(source)
        return ExtractedDocument(
            title=source.filename or "Pasted Text",
            full_text=text,
            markdown=text if is_markdown else None,
            extraction_engine=self.engine_name,
            metadata={
                "source_type": "text",
                "original_filename": source.filename or "",
                "is_markdown": is_markdown,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_markdown(source: SourceInput) -> bool:
    """Detect whether a source is likely Markdown."""
    ext = Path(source.filename or source.file_path or "").suffix.lower()
    return ext == ".md"
