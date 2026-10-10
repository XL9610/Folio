"""PDF Chunking Pipeline.

Orchestrates TOC-based chunking with a token-based fallback,
plus quality checks (empty-drop, short-merge, long-split).
"""

from __future__ import annotations

import re
import tiktoken

from .models import Chunk, ChunkingResult, DocumentPage, TocEntry, TOCUnit
from .parser import PDFParser
from .synthetic_outline import build_synthetic_outline
from .structure_metadata import (
    apply_structure_metadata_to_chunks,
    build_structure_metadata,
)
from .chunking_strategies.layout_hybrid import chunk_layout_hybrid
from .toc_extractor import TOCExtractor
from .toc_filter import TOCFilter, FilterDebugInfo
from ..toc_llm_classifier.classifier import LLMTOCClassifier
from ..toc_llm_classifier.models import TOCEntryForLLM
from ..ingestion.index_text import MAX_CHUNK_TEXT_TOKENS

# ---------------------------------------------------------------------------
# Default parameters (tunable)
# ---------------------------------------------------------------------------
# Chunk sizes are capped to fit the bundled bge-small-en-v1.5 embedder (512
# WordPiece tokens) once build_index_text prepends the [title > section] prefix.
DEFAULT_CHUNK_SIZE_TOKENS = MAX_CHUNK_TEXT_TOKENS   # target chunk size in tokens
DEFAULT_CHUNK_OVERLAP_TOKENS = 40                   # overlap between adjacent chunks
MIN_CHUNK_TOKENS = 60                               # chunks shorter than this are merged
MAX_CHUNK_TOKENS = MAX_CHUNK_TEXT_TOKENS            # chunks longer than this are split again


class PDFChunker:
    """Main entry point for the chunking pipeline.

    Usage::

        chunker = PDFChunker()
        result = chunker.chunk("path/to/file.pdf")
        for c in result.chunks:
            print(c.citation())

    """

    def __init__(
        self,
        parser: PDFParser | None = None,
        toc_extractor: TOCExtractor | None = None,
        toc_filter: TOCFilter | None = None,
        toc_llm_classifier: LLMTOCClassifier | None = None,
        tokenizer_name: str = "cl100k_base",
        chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
        min_chunk_tokens: int = MIN_CHUNK_TOKENS,
        max_chunk_tokens: int = MAX_CHUNK_TOKENS,
        use_llm_toc_filter: bool = True,
    ):
        self._parser = parser or PDFParser()
        self._toc_extractor = toc_extractor or TOCExtractor()
        self._toc_filter = toc_filter or TOCFilter()
        self._toc_llm_classifier = toc_llm_classifier
        self._tokenizer_name = tokenizer_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.use_llm_toc_filter = use_llm_toc_filter

    @property
    def _tokenizer(self):
        """Lazily load the tokenizer so import/instantiation never hits the network."""
        cached = self.__dict__.get("_tokenizer_cached")
        if cached is None:
            cached = tiktoken.get_encoding(self._tokenizer_name)
            self._tokenizer_cached = cached
        return cached

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        file_path: str,
        document_id: str | None = None,
        use_llm_toc_filter: bool | None = None,
        pages: list[DocumentPage] | None = None,
        raw_toc: list[TocEntry] | None = None,
    ) -> ChunkingResult:
        """Run the full chunking pipeline on a PDF file.

        Args:
            file_path: Path to the PDF file.
            document_id: Optional stable identifier. If None, derived from filename.
            use_llm_toc_filter: Override the instance's default LLM toggle.
                If None, uses self.use_llm_toc_filter.
            pages: Optional pre-extracted pages; if None, parsed from file_path.
            raw_toc: Optional pre-extracted TOC; if None, extracted from file_path.

        Returns a ChunkingResult with all pages, toc, chunks, and
        metadata about which strategy was used.
        """
        # Resolve effective LLM toggle WITHOUT mutating shared instance state
        effective_use_llm = (
            self.use_llm_toc_filter if use_llm_toc_filter is None
            else use_llm_toc_filter
        )

        # 0. Determine document ID (use provided one, or derive from file_path)
        if document_id:
            doc_id = document_id
        else:
            doc_id = file_path.replace("\\", "/").split("/")[-1]

        # 1. Parse pages (or reuse the caller's pre-extracted pages)
        if pages is None:
            pages = self._parser.parse(file_path)
        if not pages:
            return ChunkingResult(
                document_id=doc_id,
                pages=[],
                strategy_used="empty",
                warnings=["No pages extracted."],
            )

        # 2. Extract TOC (or reuse the caller's pre-extracted TOC)
        toc = raw_toc if raw_toc is not None else self._toc_extractor.extract(file_path)
        embedded_toc_present = bool(toc)

        # 3. Decide strategy & chunk
        toc_entries_total = 0
        toc_entries_used = 0
        front_matter_skipped = False
        skipped_titles: list[str] = []
        first_used_title = ""
        llm_used = False
        llm_first_title = ""
        llm_confidence = 0.0
        llm_reasoning = ""
        synthetic_meta = {}

        if self._toc_extractor.is_toc_usable(toc, pages):
            chunks, warnings, toc_entries_total, toc_entries_used, front_matter_skipped, \
                skipped_titles, first_used_title, \
                llm_used, llm_first_title, llm_confidence, llm_reasoning, \
                debug = \
                self._chunk_by_toc(pages, toc, doc_id, use_llm_toc_filter=effective_use_llm)
            strategy = "toc"
            # Capture body-window metadata from debug
            body_start_title = debug.first_used_title if hasattr(debug, 'first_used_title') else ""
            body_end_title = debug.last_used_title if hasattr(debug, 'last_used_title') else ""
            front_matter_skipped_count = len(debug.skipped_titles) if hasattr(debug, 'skipped_titles') else 0
            back_matter_skipped_count = len(debug.back_matter_skipped_titles) if hasattr(debug, 'back_matter_skipped_titles') else 0
            wrapper_skipped_count = len(debug.wrapper_skipped_titles) if hasattr(debug, 'wrapper_skipped_titles') else 0
            back_matter_skipped_titles = debug.back_matter_skipped_titles if hasattr(debug, 'back_matter_skipped_titles') else []
            wrapper_skipped_titles = debug.wrapper_skipped_titles if hasattr(debug, 'wrapper_skipped_titles') else []
            back_matter_start_title = debug.back_matter_start_title if hasattr(debug, 'back_matter_start_title') else ""
        else:
            chunks, warnings, synthetic_toc, synthetic_meta = self._chunk_by_synthetic_outline(
                pages, doc_id
            )
            if chunks:
                strategy = "synthetic_outline"
                toc = synthetic_toc
                toc_entries_total = self._count_toc_entries(toc)
                toc_entries_used = toc_entries_total
            else:
                # Try sentence-boundary-aware chunking first (M1); keep token
                # fallback for documents where the layout_hybrid strategy fails.
                try:
                    result = chunk_layout_hybrid(pages, doc_id)
                    chunks = result.chunks
                    warnings = list(result.warnings)
                    strategy = "layout_hybrid"
                except Exception:
                    chunks, warnings = self._chunk_by_tokens(pages, doc_id)
                    strategy = "token_fallback"
                synthetic_meta = {}
            body_start_title = synthetic_meta.get("body_start_title", "")
            body_end_title = synthetic_meta.get("body_end_title", "")
            front_matter_skipped_count = 0
            back_matter_skipped_count = 0
            wrapper_skipped_count = 0
            back_matter_skipped_titles = []
            wrapper_skipped_titles = []
            back_matter_start_title = ""
            if embedded_toc_present and strategy == "synthetic_outline":
                warnings.append("TOC present but unusable; built a synthetic outline instead.")
            if embedded_toc_present and strategy == "token_fallback":
                warnings.append("TOC present but unusable — fell back to token chunking.")

        # 4. Quality checks (tune and merge)
        chunks = self._quality_filter(chunks)

        # 4b. Quality gate (flag suspicious outputs)
        from .quality import assess_chunk_quality
        quality_warnings = assess_chunk_quality(chunks, page_count=len(pages))
        warnings.extend(quality_warnings)

        # Determine first-chunk title and chunking mode
        first_chunk_title = chunks[0].title if chunks else ""
        if strategy == "toc":
            chunking_mode = "hierarchical_toc"
        elif strategy == "synthetic_outline":
            chunking_mode = "synthetic_outline"
        elif strategy == "layout_hybrid":
            chunking_mode = "layout_hybrid"
        else:
            chunking_mode = "token_fallback"

        result = ChunkingResult(
            document_id=doc_id,
            pages=pages,
            toc=toc,
            chunks=chunks,
            strategy_used=strategy,
            warnings=warnings,
            toc_entries_total=toc_entries_total,
            toc_entries_used=toc_entries_used,
            front_matter_skipped=front_matter_skipped,
            skipped_titles=skipped_titles,
            first_used_title=first_used_title,
            toc_filter_strategy="rules+llm" if llm_used else "rules",
            llm_toc_filter_used=llm_used,
            llm_first_content_title=llm_first_title,
            llm_confidence=llm_confidence,
            llm_reasoning_summary=llm_reasoning,
            first_chunk_title=first_chunk_title,
            chunking_mode=chunking_mode,
            embedded_toc_present=embedded_toc_present,
            synthetic_outline_mode=synthetic_meta.get("synthetic_outline_mode", ""),
            # New boundary metadata
            body_start_title=body_start_title,
            body_end_title=body_end_title,
            front_matter_skipped_count=front_matter_skipped_count,
            back_matter_skipped_count=back_matter_skipped_count,
            wrapper_skipped_count=wrapper_skipped_count,
            back_matter_skipped_titles=back_matter_skipped_titles,
            wrapper_skipped_titles=wrapper_skipped_titles,
        )
        apply_structure_metadata_to_chunks(result, build_structure_metadata(result))
        return result

    # ------------------------------------------------------------------
    # TOC-based chunking
    # ------------------------------------------------------------------

    def _chunk_by_toc(
        self,
        pages: list[DocumentPage],
        toc: list[TocEntry],
        doc_id: str,
        use_llm_toc_filter: bool = False,
    ) -> tuple[list[Chunk], list[str], int, int, bool, list[str], str, bool, str, float, str, object]:
        """Chunk pages hierarchically using the full TOC tree.

        Collects ALL TOC entries (not just leaves), filters front matter,
        optionally consults the LLM, then builds chunks that preserve
        parent-section intro text before children are emitted.

        Returns:
            (chunks, warnings, toc_entries_total, toc_entries_used,
             front_matter_skipped, skipped_titles, first_used_title,
             llm_used, llm_first_title, llm_confidence, llm_reasoning)
        """
        warnings: list[str] = []

        # Collect ALL TOC entries (parents + children) in preorder
        units: list[TOCUnit] = []
        self._collect_all_units(toc, [], units)

        toc_entries_total = len(units)

        if not units:
            fallback_chunks, _ = self._chunk_by_tokens(pages, doc_id)
            from .toc_filter import FilterDebugInfo
            return fallback_chunks, warnings, toc_entries_total, 0, False, [], "", False, "", 0.0, "", FilterDebugInfo()

        # --- Rule-based front-matter filter (on full TOC units) ---
        self._toc_filter.page_count = len(pages)
        filtered_units, skipped, debug = self._toc_filter.filter_units(units)
        front_matter_skipped = skipped > 0

        if not filtered_units:
            filtered_units = units
            front_matter_skipped = False
            warnings.append("All TOC entries filtered as front matter — using all entries.")

        if front_matter_skipped:
            skipped_str = ", ".join(debug.skipped_titles[:6])
            if len(debug.skipped_titles) > 6:
                skipped_str += f" and {len(debug.skipped_titles) - 6} more"
            first = debug.first_used_title or "(unknown)"
            warnings.append(
                f"Skipped {skipped} front-matter entries: [{skipped_str}]. "
                f"Content starts at: \"{first}\"."
            )

        # Back-matter warnings
        if debug.back_matter_skipped_titles:
            back_str = ", ".join(debug.back_matter_skipped_titles[:6])
            if len(debug.back_matter_skipped_titles) > 6:
                back_str += f" and {len(debug.back_matter_skipped_titles) - 6} more"
            warnings.append(
                f"Skipped {len(debug.back_matter_skipped_titles)} back-matter entries: [{back_str}]. "
                f"Body ends before: \"{debug.back_matter_start_title}\"."
            )

        # Wrapper warnings
        if debug.wrapper_detected:
            wrapper_str = ", ".join(debug.wrapper_skipped_titles[:4])
            if len(debug.wrapper_skipped_titles) > 4:
                wrapper_str += f" and {len(debug.wrapper_skipped_titles) - 4} more"
            warnings.append(
                f"Skipped {len(debug.wrapper_skipped_titles)} synthetic wrapper entries: [{wrapper_str}]."
            )

        toc_entries_used = len(filtered_units)

        # --- Optional LLM TOC classification (subtree-aware) ---
        llm_used = False
        llm_first_title = ""
        llm_confidence = 0.0
        llm_reasoning = ""

        if use_llm_toc_filter and self._toc_llm_classifier:
            try:
                # Build TOC entries for LLM (ALL units with tree metadata)
                llm_entries = [
                    TOCEntryForLLM(
                        index=u.index,
                        title=u.entry.title,
                        path=u.section_path,
                        page_start=u.entry.page_start,
                        page_end=u.entry.page_end or 0,
                        level=u.level,
                        has_children=u.has_children,
                    )
                    for u in units
                ]
                llm_result = self._toc_llm_classifier.classify(llm_entries)

                if llm_result.is_usable:
                    first_idx = llm_result.first_content_index

                    # Validate bounds before marking as used
                    if not (0 <= first_idx < len(units)):
                        warnings.append(
                            f"LLM returned out-of-range first_content_index={first_idx} "
                            f"(len={len(units)}) — using rule-based filter."
                        )
                    elif first_idx == 0:
                        # LLM says everything is content — record but don't change filter
                        llm_used = True
                        llm_confidence = llm_result.confidence
                        llm_reasoning = llm_result.reasoning_summary
                        llm_first_title = units[0].entry.title
                        warnings.append(
                            "LLM TOC classifier found no front matter to skip."
                        )
                    else:
                        # Valid skip: apply LLM's first_content_index
                        llm_used = True
                        llm_confidence = llm_result.confidence
                        llm_reasoning = llm_result.reasoning_summary
                        filtered_units = units[first_idx:]
                        # Re-apply back-matter boundary on LLM-filtered prefix
                        # (the LLM only identifies body START, not END)
                        from .content_boundary import find_body_window
                        llm_decision = find_body_window(filtered_units, self._toc_filter.page_count)
                        filtered_units = llm_decision.filtered_units
                        toc_entries_used = len(filtered_units)
                        front_matter_skipped = True
                        llm_first_title = units[first_idx].entry.title
                        warnings.append(
                            f"LLM TOC classifier skipped {first_idx} entries. "
                            f"Content starts at: \"{llm_first_title}\" "
                            f"(confidence: {llm_confidence:.0%})."
                        )
                else:
                    warnings.append(
                        f"LLM TOC classifier confidence too low "
                        f"({llm_result.confidence:.0%}) — using rule-based filter."
                    )
            except Exception as exc:
                warnings.append(f"LLM TOC classification failed, using rule-based filter: {exc}")

        # --- Build hierarchical chunks from filtered units ---
        chunks = self._build_hierarchical_chunks(
            units=filtered_units,
            pages=pages,
            doc_id=doc_id,
        )

        return chunks, warnings, toc_entries_total, toc_entries_used, front_matter_skipped, \
            debug.skipped_titles, debug.first_used_title, \
            llm_used, llm_first_title, llm_confidence, llm_reasoning, \
            debug

    # ------------------------------------------------------------------
    # Synthetic-outline fallback chunking
    # ------------------------------------------------------------------

    def _chunk_by_synthetic_outline(
        self,
        pages: list[DocumentPage],
        doc_id: str,
    ) -> tuple[list[Chunk], list[str], list[TocEntry], dict]:
        """Build section-aware chunks when the PDF has no usable bookmarks."""
        outline = build_synthetic_outline(pages)
        if not outline.toc:
            return [], outline.warnings, [], {}

        units: list[TOCUnit] = []
        self._collect_all_units(outline.toc, [], units)
        if not units:
            return [], outline.warnings, outline.toc, {}

        chunks = self._build_hierarchical_chunks(
            units=units,
            pages=pages,
            doc_id=doc_id,
        )

        for chunk in chunks:
            if chunk.chunk_strategy == "toc_parent_intro":
                chunk.chunk_strategy = "synthetic_parent_intro"
            else:
                chunk.chunk_strategy = "synthetic_outline"
            chunk.metadata["synthetic_outline_mode"] = outline.mode

        meta = {
            "body_start_title": outline.toc[0].title if outline.toc else "",
            "body_end_title": outline.toc[-1].title if outline.toc else "",
            "body_start_page": outline.body_start_page,
            "body_end_page": outline.body_end_page,
            "synthetic_outline_mode": outline.mode,
        }
        return chunks, outline.warnings, outline.toc, meta

    # ------------------------------------------------------------------
    # Token-based fallback chunking
    # ------------------------------------------------------------------

    def _chunk_by_tokens(
        self,
        pages: list[DocumentPage],
        doc_id: str,
    ) -> tuple[list[Chunk], list[str]]:
        """Fallback: chunk pages sequentially using token count.

        Pages are concatenated and split at chunk_size tokens,
        with chunk_overlap tokens of overlap between adjacent chunks.

        Also applies page-level body window detection to skip obvious
        front/back matter pages when no TOC is available.
        """
        from .content_boundary import find_body_page_window

        warnings: list[str] = []

        # Page-level body window (skip obvious title/copyright/index pages)
        body_start, body_end, boundary_warnings = find_body_page_window(pages)
        warnings.extend(boundary_warnings)
        body_pages = pages[body_start:body_end]

        # Safety: if body window is empty, use all pages
        if not body_pages:
            body_pages = pages

        full_text = ""
        offsets: list[tuple[int, int]] = []  # (page_num, char_offset_start)

        for p in body_pages:
            offsets.append((p.page_number, len(full_text)))
            full_text += p.text + "\n"

        text_chunks = self._split_text_with_overlap(full_text)

        # Map each text chunk back to page ranges using sequential offset tracking
        # (avoids incorrect page mapping from full_text.find() on repeated content)
        chunks: list[Chunk] = []
        cumulative_offset = 0
        non_overlap = self.chunk_size - self.chunk_overlap

        for i, tc in enumerate(text_chunks):
            tc_len = len(tc)
            start_char = cumulative_offset
            end_char = start_char + tc_len
            page_range = self._char_range_to_pages(offsets, start_char, end_char)

            chunks.append(Chunk(
                text=tc.strip(),
                document_id=doc_id,
                chunk_index=i,
                page_start=page_range[0],
                page_end=page_range[1],
                chunk_strategy="token_fallback",
            ))

            # Advance offset: decode the non-overlap portion to count characters
            non_overlap_tokens = self._tokenizer.encode(tc)[:non_overlap]
            non_overlap_chars = len(self._tokenizer.decode(non_overlap_tokens))
            cumulative_offset += non_overlap_chars

        return chunks, warnings

    # ------------------------------------------------------------------
    # Quality filter
    # ------------------------------------------------------------------

    def _quality_filter(self, chunks: list[Chunk]) -> list[Chunk]:
        """Filter and clean chunks.

        1. Drop empty chunks
        2. Merge short chunks into neighbors
        3. Re-split overly long chunks
        """
        # 1. Drop empties
        chunks = [c for c in chunks if c.text.strip()]

        if not chunks:
            return chunks

        # 2. Merge short chunks (only within the same section)
        merged: list[Chunk] = []
        i = 0
        while i < len(chunks):
            c = chunks[i]
            token_count = len(self._tokenizer.encode(c.text))

            if token_count < self.min_chunk_tokens and i + 1 < len(chunks):
                next_c = chunks[i + 1]
                # Only merge if chunks belong to the same section
                same_section = (
                    c.section_path == next_c.section_path
                    or (not c.title and not next_c.title)
                )
                if same_section:
                    merged_text = c.text + "\n---\n" + next_c.text
                    merged_tokens = len(self._tokenizer.encode(merged_text))

                    if merged_tokens <= self.max_chunk_tokens:
                        c.text = merged_text
                        c.page_end = max(c.page_end, next_c.page_end)
                        c.title = c.title or next_c.title
                        i += 2
                        merged.append(c)
                        continue

            merged.append(c)
            i += 1

        # 3. Re-split long chunks
        result: list[Chunk] = []
        chunk_idx = 0
        for c in merged:
            tokens = len(self._tokenizer.encode(c.text))
            if tokens > self.max_chunk_tokens:
                # Re-split
                sub = self._token_chunk_text(
                    text=c.text,
                    doc_id=c.document_id,
                    start_idx=chunk_idx,
                    base_title=c.title,
                    section_path=c.section_path,
                    page_start=c.page_start,
                    page_end=c.page_end,
                )
                for sc in sub:
                    sc.chunk_strategy = c.chunk_strategy
                    result.append(sc)
                    chunk_idx += 1
            else:
                c.chunk_index = chunk_idx
                result.append(c)
                chunk_idx += 1

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_all_units(
        self,
        entries: list[TocEntry],
        path: list[str],
        result: list[TOCUnit],
    ) -> None:
        """Recursively collect ALL TOC entries (parents + leaves) in preorder.

        Each unit carries:
          - entry: the TocEntry
          - section_path: full hierarchical path
          - level: TOC nesting level
          - has_children: whether this entry has sub-entries
          - index: preorder position in the full tree

        For example, given a tree:
            Chapter 2 → 2.1 Algorithm → (leaf)
        The result will contain three units (Chapter 2, 2.1, leaf).
        """
        for e in entries:
            current_path = path + [e.title]
            unit = TOCUnit(
                entry=e,
                section_path=current_path,
                level=e.level,
                has_children=bool(e.children),
                index=len(result),
            )
            result.append(unit)
            if e.children:
                self._collect_all_units(e.children, current_path, result)

    def _count_toc_entries(self, entries: list[TocEntry]) -> int:
        """Count TOC entries recursively."""
        total = 0
        for entry in entries:
            total += 1
            if entry.children:
                total += self._count_toc_entries(entry.children)
        return total

    def _build_hierarchical_chunks(
        self,
        units: list[TOCUnit],
        pages: list[DocumentPage],
        doc_id: str,
    ) -> list[Chunk]:
        """Build chunks from sorted TOC units, preserving parent intro text.

        For entries with children:
          - Creates a parent-intro chunk for pages before the first child.
        For leaf entries (no children):
          - Creates chunk(s) for the entry's full page range.

        Long sections are sub-chunked with token-based splitting.
        Every chunk carries section_role metadata so retrievers can
        distinguish parent intros from child/leaf content.
        """
        chunks: list[Chunk] = []
        chunk_idx = 0
        total_pages = len(pages)

        for i, unit in enumerate(units):
            entry = unit.entry
            page_start = max(1, entry.page_start)
            page_end = entry.page_end if entry.page_end else total_pages
            page_end = min(page_end, total_pages)
            page_end = max(page_end, page_start)

            # Determine if this is a parent intro (has children in the list)
            is_parent_intro = False
            if unit.has_children and i + 1 < len(units):
                next_unit = units[i + 1]
                # The next unit is a child if its level is deeper
                if next_unit.level > unit.level:
                    child_start = max(1, next_unit.entry.page_start)
                    # Parent intro covers pages BEFORE first child
                    if child_start > page_start:
                        page_end = child_start - 1
                        is_parent_intro = True
                    else:
                        # Same page — no distinct intro pages; skip to avoid duplication
                        # The parent heading will be captured by the child chunk.
                        continue

            # Gather text
            section_text = ""
            for p in pages[page_start - 1 : page_end]:
                section_text += p.text + "\n"
            section_text = section_text.strip()

            if not section_text:
                continue

            chunk_strategy = "toc_parent_intro" if is_parent_intro else "toc"
            section_role = "parent_intro" if is_parent_intro else "leaf"
            token_count = len(self._tokenizer.encode(section_text))

            if token_count <= self.max_chunk_tokens:
                chunks.append(Chunk(
                    text=section_text,
                    document_id=doc_id,
                    chunk_index=chunk_idx,
                    title=entry.title,
                    section_path=unit.section_path,
                    page_start=page_start,
                    page_end=page_end,
                    chunk_strategy=chunk_strategy,
                    metadata={"section_role": section_role},
                ))
                chunk_idx += 1
            else:
                sub_chunks = self._token_chunk_text(
                    text=section_text,
                    doc_id=doc_id,
                    start_idx=chunk_idx,
                    base_title=entry.title,
                    section_path=unit.section_path,
                    page_start=page_start,
                    page_end=page_end,
                )
                for sc in sub_chunks:
                    sc.chunk_strategy = chunk_strategy
                    sc.metadata["section_role"] = section_role
                    chunks.append(sc)
                    chunk_idx += 1

        return chunks

    def _split_text_with_overlap(self, text: str) -> list[str]:
        """Split text into overlapping chunks by token count, snapping
        cut-points to sentence boundaries when possible.

        Guarantees no duplicate tail chunks: breaks immediately when
        the final chunk covers the end of the token list.
        """
        tokens = self._tokenizer.encode(text)
        chunks: list[str] = []

        start = 0
        while start < len(tokens):
            target_end = min(start + self.chunk_size, len(tokens))

            # ── Sentence-boundary snap ────────────────────────────────
            # Walk backward from target_end to the nearest sentence-
            # terminating token, up to 15 % of chunk_size.  Uses a
            # two-phase decode to avoid full-text round-trips on every
            # token index.
            snap_end = target_end
            if target_end < len(tokens):
                snap_window = int(self.chunk_size * 0.15)
                window_start = max(start, target_end - snap_window)
                window_tokens = tokens[window_start:target_end]
                window_text = self._tokenizer.decode(window_tokens)
                # Match sentence terminators followed by a space or newline
                for m in re.finditer(r'[.?!]["\'」）\)]?\s', window_text):
                    # m.end() is a *character* offset; convert back to tokens
                    prefix_text = window_text[:m.end()]
                    prefix_tokens = self._tokenizer.encode(prefix_text)
                    snap_end = window_start + len(prefix_tokens)
                # fall through to hard cut if no boundary found

            chunk_tokens = tokens[start:snap_end]
            chunks.append(self._tokenizer.decode(chunk_tokens))

            if snap_end >= len(tokens):
                break

            start = snap_end - self.chunk_overlap
            if start <= 0:
                break  # prevent infinite loop if chunk_size <= overlap

        return chunks

    def _token_chunk_text(
        self,
        text: str,
        doc_id: str,
        start_idx: int,
        base_title: str,
        section_path: list[str],
        page_start: int,
        page_end: int,
    ) -> list[Chunk]:
        """Split a long text into smaller chunks using token budget."""
        sub_texts = self._split_text_with_overlap(text)
        chunks: list[Chunk] = []
        for i, st in enumerate(sub_texts):
            chunks.append(Chunk(
                text=st.strip(),
                document_id=doc_id,
                chunk_index=start_idx + i,
                title=base_title or "",
                section_path=section_path,
                page_start=page_start,
                page_end=page_end,
                chunk_strategy="token_fallback",
            ))
        return chunks

    def _char_range_to_pages(
        self,
        offsets: list[tuple[int, int]],
        start_char: int,
        end_char: int,
    ) -> tuple[int, int]:
        """Map a character range [start_char, end_char) back to page numbers.

        offsets is a list of (page_number, char_offset_start).
        """
        if not offsets:
            return (1, 1)

        start_page = offsets[0][0]
        end_page = offsets[-1][0]

        for pn, char_start in offsets:
            if char_start <= start_char:
                start_page = pn
            if char_start < end_char:  # end_char is exclusive
                end_page = pn

        return (start_page, end_page)
