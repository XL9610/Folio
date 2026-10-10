"""FastAPI router for PDF upload and chunking."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Query, UploadFile, HTTPException

from ..core.config import config
from ..services.pdf_ingestion.chunker import PDFChunker
from ..services.pdf_ingestion.structure_metadata import build_structure_metadata
from ..services.llm import LLMConfig
from ..services.toc_llm_classifier.classifier import LLMTOCClassifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pdf", tags=["pdf"])

# Build LLM classifier if configured
_llm_config = LLMConfig()
_llm_classifier: LLMTOCClassifier | None = None
if _llm_config.is_configured:
    _llm_classifier = LLMTOCClassifier(_llm_config)

# Single chunker instance (stateless, safe to reuse)
_chunker = PDFChunker(toc_llm_classifier=_llm_classifier)

# Limit uploads to 50 MB
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@router.post("/chunk")
async def chunk_pdf(
    file: UploadFile = File(...),
    use_llm_toc_filter: bool = Query(config.chunking_llm_toc_filter, description="Use LLM to classify TOC entries"),
) -> dict[str, Any]:
    """Upload a PDF file and return chunked results.

    Returns the chunking result including:
    - strategy_used (toc or token_fallback)
    - chunks with full metadata (text, citations, metadata)
    - warnings (if any)
    - LLM TOC classification metadata (if enabled)
    """
    # --- Validation ---
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Check Content-Length header first (avoids reading oversized uploads into memory)
    from fastapi import Request
    # We can't easily get Request here without adding it as a dependency.
    # Instead, stream-read in chunks with a size limit.
    content = b""
    chunk_size = 1024 * 1024  # 1 MB read chunks
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        content += chunk
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
            )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # --- Sanitize filename for use as document_id ---
    safe_filename = Path(file.filename).name  # strip any path components
    # Remove null bytes and limit length
    safe_filename = safe_filename.replace("\x00", "")[:255]

    # --- Save to a temporary file ---
    suffix = Path(safe_filename).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(content)

        # --- Run the chunking pipeline (non-blocking, in threadpool) ---
        try:
            result = await asyncio.to_thread(
                _chunker.chunk, tmp_path, document_id=safe_filename,
                use_llm_toc_filter=use_llm_toc_filter,
            )
        except Exception as exc:
            logger.error("PDF chunking failed for %s: %s", safe_filename, exc, exc_info=True)
            raise HTTPException(
                status_code=422,
                detail="Failed to process PDF. The file may be corrupted or in an unsupported format.",
            ) from exc

    finally:
        # Clean up the temporary file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Validate output: reject scanned/textless PDFs that produce no usable chunks
    if result.strategy_used == "empty":
        raise HTTPException(status_code=422, detail="Could not read PDF. Is it corrupted?")
    if len(result.chunks) == 0:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF. Is it a scanned image?")
    total_text = "".join(p.text for p in result.pages).strip()
    if not total_text:
        raise HTTPException(status_code=422, detail="PDF has pages but no extractable text. Is it a scanned image?")

    response = {
        "document_id": result.document_id,
        "filename": safe_filename,
        "page_count": len(result.pages),
        "strategy_used": result.strategy_used,
        "total_chunks": len(result.chunks),
        "chunks": [c.to_dict() for c in result.chunks],
        "warnings": result.warnings,
        "toc_entries_total": result.toc_entries_total,
        "toc_entries_used": result.toc_entries_used,
        "front_matter_skipped": result.front_matter_skipped,
        "skipped_titles": result.skipped_titles,
        "first_used_title": result.first_used_title,
        "toc_filter_strategy": result.toc_filter_strategy,
        "llm_toc_filter_used": result.llm_toc_filter_used,
        "llm_first_content_title": result.llm_first_content_title,
        "llm_confidence": result.llm_confidence,
        "llm_reasoning_summary": result.llm_reasoning_summary,
        "first_chunk_title": result.first_chunk_title,
        "chunking_mode": result.chunking_mode,
        **build_structure_metadata(result).to_dict(),
        # New boundary metadata
        "body_start_title": result.body_start_title,
        "body_end_title": result.body_end_title,
        "front_matter_skipped_count": result.front_matter_skipped_count,
        "back_matter_skipped_count": result.back_matter_skipped_count,
        "wrapper_skipped_count": result.wrapper_skipped_count,
        "back_matter_skipped_titles": result.back_matter_skipped_titles,
        "wrapper_skipped_titles": result.wrapper_skipped_titles,
    }
    return response
