"""URL extractor — uses trafilatura to extract readable article text and markdown."""

from __future__ import annotations

from .base import ContentExtractor, ExtractionError
from .models import ExtractedDocument, SourceInput


class UrlExtractor(ContentExtractor):
    """Extract readable article content from URLs using trafilatura.

    Stores the URL in metadata. Trafilatura is already in requirements.txt.
    """

    engine_name = "trafilatura"

    # ------------------------------------------------------------------
    # ContentExtractor interface
    # ------------------------------------------------------------------

    def can_handle(self, source: SourceInput) -> bool:
        if source.source_type == "url":
            return True
        if source.url:
            return source.url.startswith(("http://", "https://"))
        return False

    async def extract(self, source: SourceInput) -> ExtractedDocument:
        url = source.url or ""
        if not url:
            raise ExtractionError("URL extraction requires a non-empty url")
        if not url.startswith(("http://", "https://")):
            raise ExtractionError(f"Invalid URL (must start with http/https): {url}")

        try:
            markdown, plain_text, article_title = _extract_url_text(url)
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Failed to fetch or extract URL content: {exc}") from exc

        if not plain_text.strip():
            raise ExtractionError(f"No extractable content from URL: {url}")

        return ExtractedDocument(
            title=article_title or source.filename or url,
            full_text=plain_text,
            markdown=markdown if markdown else None,
            extraction_engine=self.engine_name,
            metadata={
                "source_type": "url",
                "url": url,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_url_text(url: str) -> tuple[str, str, str]:
    """Download and extract article text from a URL using trafilatura.

    Returns (markdown, plain_text, title).
    """
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ExtractionError(f"Failed to fetch URL: {url}")

    # Try markdown first
    markdown = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_links=False,
        favor_precision=True,
    ) or ""

    # Fall back to plain text
    plain_text = trafilatura.extract(
        downloaded,
        output_format="text",
        include_links=False,
        favor_precision=True,
    ) or ""

    if not markdown:
        markdown = plain_text
    if not plain_text:
        plain_text = markdown

    # Extract title from downloaded HTML
    title = ""
    try:
        meta = trafilatura.metadata.extract_metadata(downloaded)
        if meta and meta.title:
            title = meta.title
    except Exception:
        pass

    return markdown, plain_text, title
