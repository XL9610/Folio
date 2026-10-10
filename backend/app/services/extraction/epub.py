"""EPUB extractor — preserves spine order and native TOC via ebooklib."""

from __future__ import annotations

from pathlib import Path

from ..pdf_ingestion.models import TocEntry
from .base import ContentExtractor, ExtractionError, ExtractorDependencyError
from .models import ExtractedDocument, SourceInput

_INSTALL_HINT = "pip install -r backend/requirements-extractors.txt"


class EpubExtractor(ContentExtractor):
    """Extract EPUB with native TOC, spine order, and chapter text/markdown.

    Uses monotonically increasing chapter indexes as page_start/page_end
    since EPUB has no real pages. Real locators stored in metadata.
    """

    engine_name = "ebooklib"

    # ------------------------------------------------------------------
    # ContentExtractor interface
    # ------------------------------------------------------------------

    def can_handle(self, source: SourceInput) -> bool:
        ext = Path(source.filename or source.file_path or "").suffix.lower()
        return source.source_type == "epub" or ext == ".epub"

    async def extract(self, source: SourceInput) -> ExtractedDocument:
        file_path = source.file_path
        if not file_path:
            raise ExtractionError("EPUB extraction requires a file_path")

        try:
            from ebooklib import epub  # type: ignore
        except ImportError:
            raise ExtractorDependencyError(
                f"ebooklib is required for EPUB extraction. Install with: {_INSTALL_HINT}"
            ) from None

        try:
            book = epub.read_epub(file_path)
        except Exception as exc:
            raise ExtractionError(f"Failed to read EPUB: {exc}") from exc

        # Title from metadata
        title = source.filename or Path(file_path).stem
        meta_titles = book.get_metadata("DC", "title")
        if meta_titles:
            title = str(meta_titles[0][0])

        # Native TOC
        raw_toc = _epub_toc_to_toc_entries(book.toc) if book.toc else []

        # Build flat TOC href lookup for both injection layers
        toc_href_entries = _flatten_toc_for_injection(book.toc) if book.toc else []
        injected_via_href: set[str] = set()

        # Extract chapter content in spine order
        chapters_md: list[str] = []
        chapters_text: list[str] = []
        chapter_index = 0

        for item in book.get_items_of_type(9):  # ITEM_DOCUMENT = 9
            if not isinstance(item, object):
                continue
            try:
                content = item.get_content()  # type: ignore
            except Exception:
                continue
            if not content:
                continue

            chapter_index += 1
            html_str = content.decode("utf-8", errors="replace")

            # Layer 1: href-based TOC title injection
            matched_title = _match_item_to_toc_title(
                item.get_name(), toc_href_entries
            )
            if matched_title and not _html_has_heading_with_title(
                html_str, matched_title
            ):
                html_str = f"<h1>{matched_title}</h1>\n" + html_str
                injected_via_href.add(matched_title.lower())

            md_str, txt_str = _html_to_text_and_markdown(html_str)
            if txt_str.strip():
                chapters_md.append(md_str)
                chapters_text.append(txt_str)

        full_text = "\n\n".join(chapters_text)
        if not full_text.strip():
            raise ExtractionError("No extractable text content in EPUB")

        # Layer 2: text-scanning injection for unmatched TOC titles
        combined_md = "\n\n".join(chapters_md) if chapters_md else ""
        if combined_md and toc_href_entries:
            combined_md = _inject_unmatched_titles(
                combined_md, toc_href_entries, injected_via_href
            )

        return ExtractedDocument(
            title=title,
            full_text=full_text,
            markdown=combined_md if combined_md else None,
            raw_toc=raw_toc if raw_toc else None,
            extraction_engine=self.engine_name,
            metadata={
                "source_type": "epub",
                "original_filename": source.filename or Path(file_path).name,
                "chapter_count": chapter_index,
                "toc_count": len(raw_toc),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _epub_toc_to_toc_entries(raw_toc: object, depth: int = 0) -> list[TocEntry]:
    """Convert ebooklib TOC tree to our TocEntry list.

    Uses monotonically increasing chapter indexes as page_start/page_end
    for compatibility with the existing TocEntry shape.
    """
    _counter = [0]  # mutable counter for breadth-first ordering

    def _walk(node: object, level: int = 1) -> list[TocEntry]:
        result: list[TocEntry] = []
        if isinstance(node, list):
            for child in node:
                result.extend(_walk(child, level))
        elif isinstance(node, tuple) and len(node) >= 1:
            # ebooklib TOC link: often (epub.Link, [children]) or similar
            link = node[0]
            children = node[1] if len(node) > 1 else []
            _counter[0] += 1
            title = ""
            if hasattr(link, "title"):
                title = link.title or ""
            entry = TocEntry(
                title=title,
                level=level,
                page_start=_counter[0],
                page_end=_counter[0],
                children=_walk(children, level + 1),
            )
            result.append(entry)
        elif hasattr(node, "title"):
            _counter[0] += 1
            title = getattr(node, "title", "") or ""
            entry = TocEntry(
                title=title,
                level=level,
                page_start=_counter[0],
                page_end=_counter[0],
            )
            result.append(entry)
        return result

    return _walk(raw_toc)


# ---------------------------------------------------------------------------
# TOC href injection helpers
# ---------------------------------------------------------------------------


def _normalize_href_path(href_path: str) -> str:
    """Normalize an EPUB href path for cross-format suffix matching.

    NCX paths (``"Text/ch01.xhtml"``) and NAV paths
    (``"OEBPS/Text/ch01.xhtml"``) both normalize to comparable forms.
    """
    return href_path.replace("\\", "/").strip().strip("/").lower()


def _flatten_toc_for_injection(raw_toc: object) -> list[dict[str, str]]:
    """Walk the ebooklib TOC tree and return a flat list of
    ``{title, href_path, fragment}`` dicts.

    This is a parallel traversal to :func:`_epub_toc_to_toc_entries` but
    preserves *href* information that the latter discards.
    """
    result: list[dict[str, str]] = []

    def _extract(link: object) -> None:
        title = (getattr(link, "title", "") or "").strip()
        href = (getattr(link, "href", "") or "").strip()
        if not title or not href:
            return
        path, fragment = href, ""
        if "#" in href:
            path, fragment = href.rsplit("#", 1)
        result.append(
            {
                "title": title,
                "href_path": path.strip(),
                "fragment": fragment.strip(),
            }
        )

    def _walk(node: object) -> None:
        if isinstance(node, list):
            for child in node:
                _walk(child)
        elif isinstance(node, tuple) and len(node) >= 1:
            link = node[0]
            children = node[1] if len(node) > 1 else []
            _extract(link)
            _walk(children)
        elif hasattr(node, "href") and hasattr(node, "title"):
            _extract(node)

    _walk(raw_toc)
    return result


def _match_item_to_toc_title(
    item_name: str, toc_entries: list[dict[str, str]]
) -> str | None:
    """Return the first TOC title whose *href_path* is a suffix of *item_name*.

    Returns ``None`` when no entry matches.
    """
    norm_name = _normalize_href_path(item_name)
    for entry in toc_entries:
        norm_href = _normalize_href_path(entry["href_path"])
        if norm_name.endswith(norm_href):
            return entry["title"]
    return None


def _html_has_heading_with_title(html_str: str, title: str) -> bool:
    """Return ``True`` when *html_str* already contains an ``<h1>``-``<h6>``
    tag whose inner text (tags stripped) equals *title* (case-insensitive).

    This prevents double-injection when an EPUB already uses semantic
    heading tags (e.g.  Think Python).
    """
    import re

    title_lower = title.strip().lower()
    for i in range(1, 7):
        for match in re.finditer(
            rf"<h{i}[^>]*>(.*?)</h{i}>",
            html_str,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            inner = re.sub(r"<[^>]+>", "", match.group(1)).strip().lower()
            if inner == title_lower:
                return True
    return False


def _inject_unmatched_titles(
    markdown: str,
    toc_entries: list[dict[str, str]],
    injected_via_href: set[str],
) -> str:
    """Layer-2 text-scanning injection.

    Scans *markdown* line-by-line for standalone lines whose text
    (case-insensitive) equals a TOC title that was **not** already
    injected via Layer-1 href matching.  Matching lines are promoted to
    ATX headings (``# Title``).  Only the first occurrence of each title
    is promoted; lines that already start with ``#`` are skipped.
    """
    import re

    lines = markdown.split("\n")
    matched_in_scan: set[str] = set()

    # De-duplicate titles while preserving first-occurrence order
    seen: set[str] = set()
    ordered_titles: list[str] = []
    for entry in toc_entries:
        title_lower = entry["title"].lower()
        if title_lower not in seen:
            seen.add(title_lower)
            ordered_titles.append(entry["title"])

    for title in ordered_titles:
        title_lower = title.lower()
        if title_lower in injected_via_href or title_lower in matched_in_scan:
            continue
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^#{1,6}\s+", stripped):
                continue
            if stripped.lower() == title_lower:
                lines[i] = f"# {stripped}"
                matched_in_scan.add(title_lower)
                break

    return "\n".join(lines)


def _html_to_text_and_markdown(html: str) -> tuple[str, str]:
    """Convert HTML content to plain text and simple markdown.

    Returns (markdown, plain_text) tuple. HTML headings (<h1>–<h6>) are
    converted to ATX markdown (#–######) so downstream HeadingChunker can
    detect chapter/section boundaries. All other HTML tags are stripped.
    """
    import re

    # Convert HTML headings to ATX markdown BEFORE stripping other tags
    for i in range(1, 7):
        html = re.sub(
            rf"<h{i}[^>]*>(.*?)</h{i}>",
            rf"{'#' * i} \1\n",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

    # Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Decode common entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&apos;", "'")
    text = text.replace("&nbsp;", " ")
    # Collapse whitespace — preserve heading-newline pairing
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Both markdown and text now include ATX headings for the chunker
    return text, text
