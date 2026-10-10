"""Table-of-Contents extraction from PDF using PyMuPDF bookmarks/outline."""

from __future__ import annotations

import fitz  # PyMuPDF

from .models import DocumentPage, TocEntry


class TOCExtractor:
    """Extracts hierarchical Table-of-Contents from a PDF's outline/bookmarks.

    Also validates whether the TOC is usable (maps to actual pages
    with non-empty text).
    """

    def extract(self, file_path: str) -> list[TocEntry]:
        """Extract the PDF outline and return a hierarchical list of TocEntry.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Hierarchical list of TocEntry (top-level entries only;
            nested entries are stored in .children).
        """
        doc = fitz.open(file_path)
        try:
            # simple=True returns a flat list: [[level, title, page], ...]
            # Hierarchy is encoded by level numbers, rebuilt in _build_tree.
            raw_toc = doc.get_toc(simple=True)
        finally:
            doc.close()

        if not raw_toc:
            return []

        entries = self._parse_flat_toc(raw_toc)
        return self._build_tree(entries)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_toc_usable(self, toc: list[TocEntry], pages: list[DocumentPage]) -> bool:
        """Check whether a TOC can be used for chunking.

        A TOC is considered usable when:
        - It is non-empty
        - At least 50% of entries map to pages with non-empty text
        - Page ranges are within the document
        """
        if not toc:
            return False

        if not pages:
            return False

        max_page = len(pages)
        valid_count = 0
        total_count = 0

        all_entries: list[TocEntry] = []
        self._collect_leaves(toc, all_entries)

        for entry in all_entries:
            total_count += 1
            page_start = entry.page_start

            # Check page is within bounds
            if 1 <= page_start <= max_page:
                page_idx = page_start - 1
                # Check page has non-empty text
                if pages[page_idx].text.strip():
                    valid_count += 1

        if total_count == 0:
            return False

        # Require at least 50% of entries to be valid
        return (valid_count / total_count) >= 0.5

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_flat_toc(self, raw: list) -> list[TocEntry]:
        """Convert PyMuPDF's flat TOC list into TocEntry objects.

        raw is a flat list of [level, title, page] entries.
        The hierarchy is encoded solely by level numbers — not by nesting.
        """
        entries: list[TocEntry] = []
        for item in raw:
            if not isinstance(item, (list, tuple)):
                continue
            if len(item) < 3:
                continue
            # Ensure level and page are ints; title is str
            try:
                lvl = int(item[0])
                title = str(item[1])
                page = int(item[2])
            except (ValueError, TypeError):
                continue
            entries.append(TocEntry(title=title, level=lvl, page_start=page))
        return entries

    def _build_tree(self, entries: list[TocEntry]) -> list[TocEntry]:
        """Build a tree from flat entries based on level indentation."""
        if not entries:
            return []

        # Assign page_end based on next sibling's page_start
        self._assign_page_ends(entries)

        # Build tree
        root: list[TocEntry] = []
        stack: list[TocEntry] = []  # stack of parent nodes

        for entry in entries:
            # Pop from stack until we find the right parent
            while stack and stack[-1].level >= entry.level:
                stack.pop()

            if stack:
                stack[-1].children.append(entry)
            else:
                root.append(entry)

            stack.append(entry)

        return root

    def _assign_page_ends(self, entries: list[TocEntry]) -> None:
        """Assign page_end to each entry based on the next entry's page_start.

        Guarantees page_end >= page_start to avoid empty page slices
        when multiple entries begin on the same page.
        """
        for i in range(len(entries)):
            if i + 1 < len(entries):
                next_start = entries[i + 1].page_start
                # page_end should be at least page_start
                entries[i].page_end = max(entries[i].page_start, next_start - 1)
                # If next section starts on the same page, make this section also end there
                if next_start == entries[i].page_start:
                    entries[i].page_end = next_start
                elif next_start < entries[i].page_start:
                    entries[i].page_end = entries[i].page_start
            # else: leave as None (handled by chunker as "until end")

    def _collect_leaves(self, entries: list[TocEntry], result: list[TocEntry]) -> None:
        """Collect all leaf entries (those without children) into result."""
        for entry in entries:
            if entry.children:
                self._collect_leaves(entry.children, result)
            else:
                result.append(entry)
