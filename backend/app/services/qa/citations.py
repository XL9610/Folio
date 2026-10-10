"""Citation shaping for the QA path.

Owns the per-round `_finalize_chunk_citations` (renumber S# after dedup) and the
post-merge `finalize_merged_citations` (renumber to exactly what the LLM sees),
so every citation shape lives in one place for both ask and generate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..retrieval.base import RetrievedChunk
    from .tools import ToolResults


# Must match the truncation in AgentAnswerGenerator._build_context: it shows
# the first 10 book chunks and first 8 web results to the LLM.
MAX_BOOK_CITATIONS = 10
MAX_WEB_CITATIONS = 8


def _next_ref_label(results: ToolResults, source_type: str) -> str:
    prefix = "W" if source_type == "web" else "S"
    count = sum(
        1
        for c in results.chunk_citations
        if c.get("source_type", "book") == source_type
    )
    return f"{prefix}{count + 1}"


def _dedupe_chunks(results: ToolResults) -> None:
    seen: set[str] = set()
    unique: list[RetrievedChunk] = []
    for chunk in results.chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        unique.append(chunk)
    results.chunks = unique


def _finalize_chunk_citations(results: ToolResults) -> None:
    """Make final book citation refs match the final deduped chunk list."""
    _dedupe_chunks(results)
    existing_by_chunk: dict[str, dict] = {}
    web_citations: list[dict] = []
    final_chunk_ids = {chunk.chunk_id for chunk in results.chunks}

    for citation in results.chunk_citations:
        if citation.get("source_type", "book") != "book":
            web_citations.append(citation)
            continue
        chunk_id = citation.get("chunk_id", "")
        if not chunk_id or chunk_id not in final_chunk_ids or chunk_id in existing_by_chunk:
            continue
        existing_by_chunk[chunk_id] = dict(citation)

    book_citations: list[dict] = []
    for idx, chunk in enumerate(results.chunks, start=1):
        citation = existing_by_chunk.get(chunk.chunk_id, {})
        snippet = citation.get("text_snippet", "")
        if not snippet:
            snippet = chunk.text[:120].replace("\n", " ").strip()
            if len(chunk.text) > 120:
                snippet += "\u2026"
        citation.update({
            "source_type": "book",
            "ref_label": f"S{idx}",
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "text_snippet": snippet,
            "section_path": citation.get("section_path") or chunk.section_path or "",
            "title": citation.get("title") or chunk.title or "",
            "page_start": citation.get("page_start", chunk.page_start),
            "page_end": citation.get("page_end", chunk.page_end),
        })
        book_citations.append(citation)

    results.chunk_citations = book_citations + web_citations


def finalize_merged_citations(results: ToolResults) -> None:
    """Rebuild `chunk_citations` for the merged, multi-round tool results.

    `_finalize_chunk_citations` renumbers per-round, so after rounds are merged
    by `_merge_tool_results` the ref labels collide (each round restarts at S1).
    It also labels every retrieved chunk, even though `_build_context` only shows
    the first `MAX_BOOK_CITATIONS` to the LLM — the rest are dangling labels the
    model could never have cited. This renumbers to exactly what reached the LLM.
    """
    # Reuse the same dedupe the per-round finalize applies.
    seen: set[str] = set()
    unique: list[RetrievedChunk] = []
    for chunk in results.chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        unique.append(chunk)
    results.chunks = unique

    book_citations: list[dict] = []
    for idx, chunk in enumerate(results.chunks[:MAX_BOOK_CITATIONS], start=1):
        snippet = chunk.text[:120].replace("\n", " ").strip()
        if len(chunk.text) > 120:
            snippet += "\u2026"
        book_citations.append({
            "source_type": "book",
            "ref_label": f"S{idx}",
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "text_snippet": snippet,
            "section_path": chunk.section_path or "",
            "title": chunk.title or "",
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
        })

    web_citations: list[dict] = []
    for idx, item in enumerate(results.web_results[:MAX_WEB_CITATIONS], start=1):
        snippet = str(item.get("text_snippet") or "")
        if not snippet:
            content = str(item.get("content") or "")
            snippet = content[:160].replace("\n", " ").strip()
        web_citations.append({
            "source_type": "web",
            "ref_label": f"W{idx}",
            "chunk_id": f"web:W{idx}",
            "document_id": "",
            "text_snippet": snippet,
            "section_path": item.get("site_name", ""),
            "page_start": -1,
            "page_end": -1,
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "site_name": item.get("site_name", ""),
            "published_date": item.get("published_date", ""),
            "retrieved_at": item.get("retrieved_at", ""),
        })

    results.chunk_citations = book_citations + web_citations
