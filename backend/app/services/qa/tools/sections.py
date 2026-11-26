"""Section tools: find_sections, get_section_chunks."""

from __future__ import annotations

import logging
import re

from ....repositories.chunk_repository import (
    get_chunks_by_section_prefix,
    get_chunks_for_section_paths,
    get_document_sections,
)
from ....repositories.document_repository import get_document
from ...pdf_ingestion.structure_metadata import (
    document_structure_fields,
    is_unreliable_structure_status,
)
from ...retrieval.base import RetrievedChunk
from ..agent_planner import ToolCall
from ..section_labels import (
    extract_numbered_section_label,
    extract_numbered_section_prefix,
    is_page_window_section_label,
)
from .types import SectionMatch, ToolResults

logger = logging.getLogger(__name__)


class SectionTools:
    """Section-locating tools."""

    def _tool_find_sections(
        self,
        results: ToolResults,
        tc: ToolCall,
        notebook_id: str,
        doc_ids: list[str],
        top_k: int = 0,
    ) -> None:
        """Find sections matching a natural-language hint like 'Chapter 1' or 'graphs'.

        Strategy (deterministic + conservative):
        1. Exact chapter-number match: "Chapter 1" "Chapter 10" "chapter 3"
        2. Substring match on section_path / title
        3. Fallback: return all top-level sections as candidates

        Returns ordered SectionMatch list in results.section_matches.
        """
        section_query = (
            tc.args.get("section_query", "")
            or tc.args.get("query", "")
            or tc.args.get("chapter", "")
        ).strip()
        if not section_query:
            results.trace.append({
                "name": "find_sections",
                "status": "error",
                "error": "No section_query or chapter hint provided",
            })
            return

        requested = tc.args.get("document_ids", doc_ids)[:5]
        bounded = [did for did in requested if did in doc_ids]

        # Store the target for later use (generator can reference it)
        results.target_section = section_query

        # Detect explicit chapter number
        chapter_match = re.search(
            r"(?:chapter|ch|section|sec)\s*#?\s*(\d+)", section_query, re.IGNORECASE
        )
        chapter_num = int(chapter_match.group(1)) if chapter_match else None

        all_matches: list[SectionMatch] = []
        for did in bounded:
            doc = get_document(did)
            if not doc or doc.get("notebook_id") != notebook_id:
                continue
            sections = get_document_sections(did) or []
            structure = document_structure_fields(doc)
            structure_status = structure.get("structure_status")

            for s in sections:
                sp = s.get("section_path", "") or ""
                title = s.get("title", "") or ""
                page_window_label = is_page_window_section_label(sp, title)
                match_type = ""
                rank = 999

                # Page-window fallback labels are not chapters.  Never infer
                # "Chapter 20" from "Pages 20-27".
                if (
                    chapter_num is not None
                    and page_window_label
                    and is_unreliable_structure_status(structure_status)
                ):
                    continue

                # Level 1: exact chapter number match
                if chapter_num is not None:
                    numbered, label = extract_numbered_section_label(sp, title)
                    # Use segment-aware parsing so "Chapter 1" does NOT match
                    # "Chapter 10", while "PART > 20. Title" does match chapter 20.
                    if numbered == chapter_num:
                        match_type = "exact_chapter"
                        rank = 1

                # Level 2: substring match (case-insensitive)
                # Guard: if we detected a chapter number, skip sections whose chapter number differs
                if not match_type:
                    q_lower = section_query.lower()
                    sp_lower = sp.lower()
                    title_lower = title.lower()
                    if q_lower in sp_lower or q_lower in title_lower:
                        # Avoid substring false-positives for chapter numbers
                        # e.g. "Chapter 1" appearing inside "Chapter 10"
                        if chapter_num is not None:
                            other_num, _ = extract_numbered_section_label(sp, title)
                            if other_num is not None and other_num != chapter_num:
                                # Different chapter — do NOT match via substring
                                pass
                            else:
                                match_type = "substring"
                                rank = 2
                        else:
                            match_type = "substring"
                            rank = 2

                # Level 3: individual word match
                if not match_type:
                    # If chapter number was detected, skip fuzzy matching on sections
                    # with a different chapter number to avoid false positives
                    if chapter_num is not None:
                        other_num, _ = extract_numbered_section_label(sp, title)
                        if other_num is not None and other_num != chapter_num:
                            # different chapter — skip entirely
                            match_type = ""
                        else:
                            words = [w.lower() for w in section_query.split() if len(w) > 2]
                            hit_count = sum(
                                1 for w in words if w in sp.lower() or w in title.lower()
                            )
                            if hit_count >= 1:
                                match_type = "fuzzy"
                                rank = 10 - hit_count
                    else:
                        words = [w.lower() for w in section_query.split() if len(w) > 2]
                        hit_count = sum(
                            1 for w in words if w in sp.lower() or w in title.lower()
                        )
                        if hit_count >= 1:
                            match_type = "fuzzy"
                            rank = 10 - hit_count

                if match_type:
                    all_matches.append(SectionMatch(
                        document_id=did,
                        section_path=sp,
                        title=title,
                        page_start=s.get("page_start", -1),
                        page_end=s.get("page_end", -1),
                        chunk_count=s.get("chunk_count", 0),
                        rank=rank,
                        match_type=match_type,
                    ))

        # Sort: lower rank first, then by page_start
        all_matches.sort(key=lambda m: (m.rank, m.section_path.count(">"), m.page_start))

        # Truncate to reasonable count
        results.section_matches = all_matches[:15]

        status = "ok" if results.section_matches else "no_match"
        results.trace.append({
            "name": "find_sections",
            "status": status,
            "args": {
                "section_query": section_query,
                "document_ids": bounded,
                "matches": len(results.section_matches),
            },
        })

    def _tool_get_section_chunks(
        self,
        results: ToolResults,
        tc: ToolCall,
        notebook_id: str,
        doc_ids: list[str],
        top_k: int = 0,
    ) -> None:
        """Retrieve chunks for matched sections from find_sections or explicit section_paths.

        Args:
            section_paths: list of section_path strings (takes precedence)
            section_prefix: string prefix to match (used if section_paths not given)
            document_ids: optional document filter
            limit: max chunks to return (default 20)
        """
        section_paths = tc.args.get("section_paths", [])
        section_prefix = tc.args.get("section_prefix", "")
        section_prefixes: list[str] = []
        requested = tc.args.get("document_ids", doc_ids)[:5]
        bounded = [did for did in requested if did in doc_ids]
        limit = min(tc.args.get("limit", 20), 50)

        # If the planner gave us the target from find_sections, use section_matches
        if not section_paths and not section_prefix and results.section_matches:
            for match in results.section_matches:
                if match.match_type != "exact_chapter":
                    continue
                _number, prefix, _label = extract_numbered_section_prefix(
                    match.section_path,
                    match.title,
                )
                if prefix and prefix not in section_prefixes:
                    section_prefixes.append(prefix)
            if len(section_prefixes) == 1:
                section_prefix = section_prefixes[0]
            else:
                section_paths = [m.section_path for m in results.section_matches[:3]]

        if not section_paths and not section_prefix and not section_prefixes:
            fallback_query = (results.target_section or "").strip()
            if fallback_query and not re.search(r"\b(?:chapter|ch|section|sec)\s*#?\s*\d+\b", fallback_query, re.I):
                chunks = self._retriever.search(
                    notebook_id=notebook_id,
                    query=fallback_query,
                    top_k=min(max(top_k, 1), limit),
                    document_ids=bounded,
                )
                results.chunks.extend(chunks)
                for chunk in chunks[:8]:
                    snippet = chunk.text[:120].replace("\n", " ").strip()
                    if len(chunk.text) > 120:
                        snippet += "\u2026"
                    results.chunk_citations.append({
                        "source_type": "book",
                        "ref_label": "",
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "text_snippet": snippet,
                        "section_path": chunk.section_path or "",
                        "title": chunk.title or "",
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                    })
                results.trace.append({
                    "name": "get_section_chunks",
                    "status": "fallback_retrieve" if chunks else "no_match",
                    "args": {
                        "query": fallback_query,
                        "document_ids": bounded,
                        "results": len(chunks),
                    },
                })
                return
            results.trace.append({
                "name": "get_section_chunks",
                "status": "error",
                "error": "No section_paths or section_prefix provided, and no prior section_matches",
            })
            return

        all_records: list[dict] = []

        for did in bounded:
            doc = get_document(did)
            if not doc or doc.get("notebook_id") != notebook_id:
                continue

            if section_paths:
                records = get_chunks_for_section_paths(did, section_paths, limit)
            elif section_prefixes:
                records = []
                for prefix in section_prefixes[:3]:
                    records.extend(get_chunks_by_section_prefix(did, prefix, limit))
            else:
                records = get_chunks_by_section_prefix(did, section_prefix, limit)

            all_records.extend(records)

        # Sort by page_start, chunk_index
        all_records.sort(key=lambda r: (r.get("page_start", 0), r.get("chunk_index", 0)))

        # Convert to RetrievedChunk
        for rec in all_records:
            snippet = rec["text"][:120].replace("\n", " ").strip()
            if len(rec["text"]) > 120:
                snippet += "\u2026"
            results.chunks.append(RetrievedChunk(
                chunk_id=rec["id"],
                document_id=rec["document_id"],
                text=rec["text"],
                title=rec.get("title", ""),
                section_path=rec.get("section_path", ""),
                page_start=rec.get("page_start", -1),
                page_end=rec.get("page_end", -1),
            ))
            ref_label = self._next_ref_label(results, "book")
            results.chunk_citations.append({
                "source_type": "book",
                "ref_label": ref_label,
                "chunk_id": rec["id"],
                "document_id": rec["document_id"],
                "text_snippet": snippet,
                "section_path": rec.get("section_path", ""),
                "title": rec.get("title", ""),
                "page_start": rec.get("page_start", -1),
                "page_end": rec.get("page_end", -1),
            })

        status = "ok" if all_records else "no_match"
        results.trace.append({
            "name": "get_section_chunks",
            "status": status,
            "args": {
                "section_paths": section_paths[:3] if section_paths else None,
                "section_prefix": section_prefix or None,
                "section_prefixes": section_prefixes[:3] if section_prefixes else None,
                "document_ids": bounded,
                "limit": limit,
                "results": len(all_records),
            },
        })
