"""Whole-source tools: get_source_metadata, get_source_outline, get_source_chunks."""

from __future__ import annotations

import logging

from ....repositories.chunk_repository import (
    get_chunks_by_document,
    get_document_sections,
)
from ....repositories.document_repository import get_document
from ...pdf_ingestion.structure_metadata import (
    document_structure_fields,
)
from ...retrieval.base import RetrievedChunk
from ..agent_planner import ToolCall
from ..citations import (
    MAX_BOOK_CITATIONS,
)
from .types import ToolResults

logger = logging.getLogger(__name__)


class SourceTools:
    """Whole-source tools."""

    def _tool_metadata(
        self,
        results: ToolResults,
        tc: ToolCall,
        notebook_id: str,
        doc_ids: list[str],
        top_k: int = 0,
    ) -> None:
        requested = tc.args.get("document_ids", doc_ids)[:5]
        for did in requested:
            if did not in doc_ids:
                continue
            doc = get_document(did)
            if doc and doc.get("notebook_id") == notebook_id:
                results.metadata.append({
                    "document_id": doc["id"],
                    "filename": doc["filename"],
                    "page_count": doc["page_count"],
                    "chunk_count": doc["chunk_count"],
                    "status": doc["status"],
                    **document_structure_fields(doc),
                })
        results.trace.append({
            "name": "get_source_metadata",
            "status": "ok",
            "args": {"document_ids": requested, "results": len(results.metadata)},
        })

    def _tool_outline(
        self,
        results: ToolResults,
        tc: ToolCall,
        notebook_id: str,
        doc_ids: list[str],
        top_k: int = 0,
    ) -> None:
        requested = tc.args.get("document_ids", doc_ids)[:5]
        for did in requested:
            if did not in doc_ids:
                continue
            doc = get_document(did)
            if not doc or doc.get("notebook_id") != notebook_id:
                continue
            sections = get_document_sections(did) or []
            structure = document_structure_fields(doc)
            results.outline.append({
                "document_id": did,
                "filename": doc["filename"],
                "page_count": doc["page_count"],
                **structure,
                "sections": [
                    {
                        "section_path": s.get("section_path", ""),
                        "title": s.get("title", ""),
                        "page_start": s.get("page_start", -1),
                        "page_end": s.get("page_end", -1),
                        "chunk_count": s.get("chunk_count", 0),
                    }
                    for s in sections
                ],
            })
        results.trace.append({
            "name": "get_source_outline",
            "status": "ok",
            "args": {"document_ids": requested, "results": len(results.outline)},
        })

    def _tool_get_source_chunks(
        self,
        results: ToolResults,
        tc: ToolCall,
        notebook_id: str,
        doc_ids: list[str],
        top_k: int = 0,
    ) -> None:
        """Retrieve representative body chunks across the document (overview/summary).

        Branch C: calls routed from the planner's overview augmentation carry
        ``_coverage`` — only those apply the coverage sampling profile
        (front matter included: a magazine's cover story or a deck's
        title/agenda slides are exactly the overview evidence, and the even
        spread dilutes a book's copyright noise; head bias: overview
        evidence is front-loaded). Planner-initiated calls keep the classic
        body-favoring sampling.
        """
        coverage = bool(tc.args.get("_coverage"))
        from ...generation.sampling import sample_body_chunks

        requested = tc.args.get("document_ids", doc_ids)[:5]
        bounded = [did for did in requested if did in doc_ids]
        limit = max(1, min(int(tc.args.get("limit", 20) or 20), 50))

        collected: list[RetrievedChunk] = []
        for did in bounded:
            doc = get_document(did)
            if not doc or doc.get("notebook_id") != notebook_id:
                continue
            for rec in get_chunks_by_document(did):
                collected.append(RetrievedChunk(
                    chunk_id=rec["id"],
                    document_id=rec["document_id"],
                    text=rec["text"],
                    title=rec.get("title", ""),
                    section_path=rec.get("section_path", ""),
                    page_start=rec.get("page_start", -1),
                    page_end=rec.get("page_end", -1),
                ))

        sampled = sample_body_chunks(
            collected,
            max_chunks=min(limit, MAX_BOOK_CITATIONS),
            include_front_matter=coverage,
            # The coverage share is six chunks, so reserving three for the
            # head would spend half the budget on front matter. One head chunk
            # preserves cover/agenda evidence while leaving room for the body.
            head_bias=1 if coverage else 0,
        )

        seen = set()
        for chunk in sampled:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            snippet = chunk.text[:120].replace("\n", " ").strip()
            if len(chunk.text) > 120:
                snippet += "…"
            ref_label = self._next_ref_label(results, "book")
            results.chunk_citations.append({
                "source_type": "book",
                "ref_label": ref_label,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text_snippet": snippet,
                "section_path": chunk.section_path or "",
                "title": chunk.title or "",
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            })
        # Overview samples are the primary evidence for whole-document queries;
        # the generator reads only the first 10 book chunks, and retrieve_chunks
        # (which ran earlier) could crowd them out. Prepend, not append.
        results.chunks = list(sampled) + results.chunks

        results.trace.append({
            "name": "get_source_chunks",
            "status": "ok" if sampled else "no_match",
            "args": {
                "document_ids": bounded,
                "limit": limit,
                "results": len(sampled),
                "coverage": coverage,
            },
        })
