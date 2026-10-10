"""Eval/debug exports for the ask path.

Builds full passage + source-structure payloads (used by the eval harness and
the optional `debug_retrieval` flag) without changing the normal UX output.
"""

from __future__ import annotations

import json
from contextlib import suppress

from ...repositories.chunk_repository import get_chunk_by_id
from ...repositories.document_repository import get_document
from ..pdf_ingestion.structure_metadata import document_structure_fields
from .source_framing import _citation_ref_by_chunk
from .tools import ToolResults


def _safe_json_object(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _debug_source_metadata(tool_results: ToolResults) -> list[dict]:
    """Return deduplicated source metadata for eval/debug exports."""
    by_id: dict[str, dict] = {}
    for item in list(tool_results.metadata) + list(tool_results.outline):
        document_id = str(item.get("document_id", ""))
        if not document_id or document_id in by_id:
            continue
        by_id[document_id] = {
            "document_id": document_id,
            "filename": item.get("filename", ""),
            "page_count": item.get("page_count", 0),
            "chunk_count": item.get("chunk_count", 0),
            **document_structure_fields(item),
        }

    for chunk in tool_results.chunks:
        document_id = chunk.document_id
        if not document_id or document_id in by_id:
            continue
        doc = get_document(document_id)
        if not doc:
            continue
        by_id[document_id] = {
            "document_id": document_id,
            "filename": doc.get("filename", ""),
            "page_count": doc.get("page_count", 0),
            "chunk_count": doc.get("chunk_count", 0),
            **document_structure_fields(doc),
        }

    return list(by_id.values())


def _debug_retrieved_chunks(tool_results: ToolResults) -> list[dict]:
    """Build full passages for eval harnesses without changing normal UX output."""
    items: list[dict] = []
    ref_by_chunk = _citation_ref_by_chunk(tool_results)
    source_meta = {
        item["document_id"]: item
        for item in _debug_source_metadata(tool_results)
        if item.get("document_id")
    }

    for idx, chunk in enumerate(tool_results.chunks, start=1):
        db_chunk = get_chunk_by_id(chunk.chunk_id) or {}
        metadata = {}
        metadata.update(chunk.metadata or {})
        metadata.update(_safe_json_object(db_chunk.get("metadata_json")))

        structure = source_meta.get(chunk.document_id, {})
        ref_label = ref_by_chunk.get(chunk.chunk_id) or f"S{idx}"
        section_label = (
            metadata.get("section_label")
            or chunk.section_path
            or chunk.title
            or db_chunk.get("section_path", "")
            or db_chunk.get("title", "")
        )
        score = float(getattr(chunk, "rrf_score", 0.0) or 0.0)
        llm_score = getattr(chunk, "llm_rerank_score", None)
        if llm_score is not None:
            with suppress(TypeError, ValueError):
                score = float(llm_score)
        items.append({
            "passage_id": ref_label,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "text": chunk.text or db_chunk.get("text", ""),
            "title": chunk.title or db_chunk.get("title", ""),
            "section_path": chunk.section_path or db_chunk.get("section_path", ""),
            "section_label": section_label,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "outline_source": str(metadata.get("outline_source", "")),
            "chunking_strategy": str(metadata.get("chunking_strategy", "")),
            "source_type": "book",
            "dense_score": float(getattr(chunk, "dense_score", 0.0) or 0.0),
            "fts_score": float(getattr(chunk, "fts_score", 0.0) or 0.0),
            "rrf_score": float(getattr(chunk, "rrf_score", 0.0) or 0.0),
            "score": score,
            "structure_status": structure.get("structure_status", "unknown_legacy"),
            "structure_confidence": structure.get("structure_confidence", "") or "",
            "dense_rank": int(metadata.get("dense_rank", 0) or 0),
            "keyword_rank": int(metadata.get("keyword_rank", 0) or 0),
            "fusion_rank": int(metadata.get("fusion_rank", 0) or 0),
            "retrieval_latency_ms": 0,
            "planner_query": "",
            "expanded_query": "",
            "warnings": "",
        })

    for idx, item in enumerate(tool_results.web_results, start=1):
        ref_label = str(item.get("ref_label") or f"W{idx}")
        items.append({
            "passage_id": ref_label,
            "chunk_id": f"web:{ref_label}",
            "document_id": "",
            "text": str(item.get("content") or item.get("text_snippet") or ""),
            "title": str(item.get("title") or ""),
            "section_path": str(item.get("site_name") or ""),
            "section_label": str(item.get("title") or item.get("site_name") or "web"),
            "page_start": -1,
            "page_end": -1,
            "outline_source": "web",
            "chunking_strategy": "web_search",
            "source_type": "web",
            "dense_score": 0.0,
            "fts_score": 0.0,
            "rrf_score": 0.0,
            "score": float(item.get("score") or 0.0),
            "structure_status": "web",
            "structure_confidence": "",
            "dense_rank": 0,
            "keyword_rank": 0,
            "fusion_rank": 0,
            "retrieval_latency_ms": 0,
            "planner_query": "",
            "expanded_query": "",
            "warnings": "",
        })

    return items
