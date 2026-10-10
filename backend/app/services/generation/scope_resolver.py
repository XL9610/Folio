"""Three-layer scope resolution: page range -> TOC literal -> semantic retrieval."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ...repositories.chunk_repository import (
    get_chunks_by_document,
    get_chunks_by_page_range,
    get_document_sections,
)
from ...repositories.document_repository import get_document
from ..pdf_ingestion.structure_metadata import document_structure_fields
from ..retrieval.base import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class ScopeResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


PAGE_RANGE_RE = re.compile(
    r"(?:p\.?|pages?|pp\.?)\s*(\d+)\s*[-–to]+\s*(\d+)",
    re.IGNORECASE,
)

# Extract chapter/section numbers: "Chapter 15", "Ch. 15", "chapter 15", "第15章", "15"
CHAPTER_NUM_RE = re.compile(
    r"(?:chapter|ch\.?|section|§)\s*(\d+)",
    re.IGNORECASE,
)
# Bare number: 1-3 digits, must be at start or after a space
BARE_NUM_RE = re.compile(r"\b(\d{1,3})\b")

# Match a chapter number in a section title: "15.", "15 ", "15:", "15-", "15–"
CHAPTER_NUM_IN_TITLE_RE = re.compile(
    r"(?:^|\s)(\d{1,3})(?:\.|:|\)|[\s–-])"
)


class ScopeResolver:
    """Resolve a natural-language scope to concrete chunks.

    Three layers, tried in order:
      1. Page range: "p. 50-80", "pages 50 to 80"
      2. TOC literal: substring match against document section_path/title
      3. Semantic retrieval: hybrid search against the scope text
    """

    def __init__(self, retriever):
        self._retriever = retriever

    def resolve(
        self,
        document_id: str,
        notebook_id: str,
        scope: str,
    ) -> ScopeResult:
        """Resolve scope to chunks. Returns (chunks, resolution_meta)."""
        scope = (scope or "").strip()
        structure = self._document_structure(document_id)

        if not scope:
            result = self._all_chunks(document_id, notebook_id)
            result.metadata.update({
                "scope_resolved": True,
                "structure_status": structure.get("structure_status", "unknown_legacy"),
            })
            return result

        # Layer 1: page range
        page_chunks = self._try_page_range(document_id, scope)
        if page_chunks is not None:
            m = PAGE_RANGE_RE.search(scope)
            return ScopeResult(
                chunks=page_chunks,
                metadata={
                    "layer": "page_range",
                    "page_start": int(m.group(1)),
                    "page_end": int(m.group(2)),
                    "scope_resolved": True,
                    "structure_status": structure.get("structure_status", "unknown_legacy"),
                },
            )

        # Layer 2: TOC literal match
        toc_chunks = self._try_toc_match(document_id, scope)
        if toc_chunks:
            return ScopeResult(
                chunks=toc_chunks,
                metadata={
                    "layer": "toc",
                    "matched_sections": list({
                        c.section_path for c in toc_chunks[:5] if c.section_path
                    }),
                    "scope_resolved": True,
                    "structure_status": structure.get("structure_status", "unknown_legacy"),
                },
            )

        chapter_hint = CHAPTER_NUM_RE.search(scope)
        if chapter_hint:
            return ScopeResult(
                chunks=[],
                metadata={
                    "layer": "unresolved_section",
                    "scope": scope,
                    "scope_resolved": False,
                    "structure_status": structure.get("structure_status", "unknown_legacy"),
                    "unresolved_reason": (
                        f"Could not match {chapter_hint.group(0)} to a reliable "
                        "document section. Use a page range or topic keyword instead."
                    ),
                },
            )

        # Layer 3: semantic retrieval
        raw = self._retriever.search(
            notebook_id=notebook_id,
            query=scope,
            top_k=20,
            document_ids=[document_id],
        )
        return ScopeResult(
            chunks=raw,
            metadata={
                "layer": "semantic",
                "scope": scope,
                "n_retrieved": len(raw),
                "scope_resolved": True,
                "structure_status": structure.get("structure_status", "unknown_legacy"),
            },
        )

    # ------------------------------------------------------------------
    # Layer 1 helpers
    # ------------------------------------------------------------------

    def _try_page_range(
        self, document_id: str, scope: str
    ) -> list[RetrievedChunk] | None:
        m = PAGE_RANGE_RE.search(scope)
        if not m:
            return None
        start = int(m.group(1))
        end = int(m.group(2))
        if start > end:
            start, end = end, start

        rows = get_chunks_by_page_range(document_id, start, end)
        if not rows:
            return None
        return [self._db_row_to_chunk(r) for r in rows]

    # ------------------------------------------------------------------
    # Layer 2 helpers
    # ------------------------------------------------------------------

    def _try_toc_match(
        self, document_id: str, scope: str
    ) -> list[RetrievedChunk] | None:
        from ...repositories.chunk_repository import (
            get_chunks_for_section_paths,
        )

        sections = get_document_sections(document_id)
        if not sections:
            return None

        scope_lower = scope.lower().strip()
        matched_paths: list[str] = []

        # Extract chapter number if present: "Chapter 15" → "15"
        chapter_num = None
        cm = CHAPTER_NUM_RE.search(scope_lower)
        if cm:
            chapter_num = cm.group(1)
        else:
            bm = BARE_NUM_RE.search(scope_lower)
            if bm:
                chapter_num = bm.group(1)

        for sec in sections:
            sp = (sec.get("section_path") or "").lower()
            title = (sec.get("title") or "").lower()

            # 1) Direct substring match
            if scope_lower in sp or scope_lower in title:
                matched_paths.append(sec.get("section_path", ""))
                continue

            # 2) Chapter-number match: extracted number matches "15." / "15 " in title
            if chapter_num:
                # Check section_path or title contains the number as a chapter marker
                raw_sp = sec.get("section_path") or ""
                raw_title = sec.get("title") or ""
                combined = raw_title + " " + raw_sp
                if CHAPTER_NUM_IN_TITLE_RE.search(combined):
                    num_matches = CHAPTER_NUM_IN_TITLE_RE.findall(combined)
                    if chapter_num in num_matches:
                        matched_paths.append(sec.get("section_path", ""))

        if not matched_paths:
            return None

        rows = get_chunks_for_section_paths(
            document_id, matched_paths, limit=50
        )
        if not rows:
            return None
        return [self._db_row_to_chunk(r) for r in rows]

    # ------------------------------------------------------------------
    # Layer 0 (all chunks) helpers
    # ------------------------------------------------------------------

    def _all_chunks(
        self, document_id: str, notebook_id: str
    ) -> ScopeResult:
        from .sampling import is_body_chunk

        rows = get_chunks_by_document(document_id)
        all_chunks = [self._db_row_to_chunk(r) for r in rows]

        # Filter non-body chunks by default for whole-document scope
        body_chunks = [c for c in all_chunks if is_body_chunk(c)]
        filtered_count = len(all_chunks) - len(body_chunks)

        # Safety: if filtering removed everything, return all
        if not body_chunks:
            body_chunks = all_chunks

        return ScopeResult(
            chunks=body_chunks,
            metadata={
                "layer": "all",
                "n_total": len(all_chunks),
                "filtered_non_body_count": filtered_count,
            },
        )

    # ------------------------------------------------------------------
    # Converters
    # ------------------------------------------------------------------

    @staticmethod
    def _db_row_to_chunk(row: dict) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=row.get("id", ""),
            document_id=row.get("document_id", ""),
            text=row.get("text", ""),
            title=row.get("title", ""),
            section_path=row.get("section_path", ""),
            page_start=row.get("page_start", -1),
            page_end=row.get("page_end", -1),
        )

    @staticmethod
    def _document_structure(document_id: str) -> dict:
        return document_structure_fields(get_document(document_id))
