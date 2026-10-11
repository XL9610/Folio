"""FastAPI router for document ingestion.

POST /api/documents/ingest
  Upload any supported document, persist chunks to SQLite (FTS5 + sqlite-vec).
  Returns a summary — NOT the full chunk text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from ..core.config import config
from ..repositories.chunk_repository import get_document_sections
from ..repositories.document_repository import delete_document, get_document, list_documents
from ..repositories.notebook_repository import get_notebook, get_or_create_default
from ..services.extraction.artifact_store import ExtractionArtifactStore
from ..services.extraction.registry import infer_source_type
from ..services.index.sqlite_vec_store import SqliteVecStore
from ..services.ingestion.service import IngestionResult, SourceInput, ingest_source
from ..repositories.ingestion_job_repository import create_job
from ..services.pdf_ingestion.structure_metadata import document_structure_fields

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


def _run_ingestion_job(
    job_id: str,
    source: SourceInput,
    tmp_path: str,
    use_llm_toc_filter: bool,
    force_reingest: bool,
) -> None:
    """Run ingest_source in a background thread and update the job row."""
    from ..repositories.ingestion_job_repository import update_job

    def report(stage: str, pct: int) -> None:
        update_job(job_id, status="running", stage=stage, progress_pct=pct)

    try:
        report("extracting", 5)
        result = ingest_source(
            source=source,
            use_llm_toc_filter=use_llm_toc_filter,
            force_reingest=force_reingest,
            progress_callback=report,
        )
        update_job(
            job_id,
            status="completed",
            stage=result.status,  # completed | fts_only | indexed_partial
            progress_pct=100,
            document_id=result.document_id,
            warnings_json=json.dumps(result.warnings, ensure_ascii=False),
        )
    except Exception as exc:
        logger.error("Ingestion job %s failed: %s", job_id, exc, exc_info=True)
        update_job(job_id, status="failed", stage="failed", error_message=str(exc))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    notebook_id: str = Query("default", description="Notebook ID or 'default'"),
    use_llm_toc_filter: bool = Query(config.chunking_llm_toc_filter, description="Use LLM to classify TOC entries"),
    force_reingest: bool = Query(False, description="Re-ingest even if already exists"),
) -> dict:
    """Upload a document and ingest it into the study companion.

    POST /api/documents/ingest

    Supports PDF, EPUB, PPTX, DOCX, TXT, MD, and fallback formats.
    Steps:
      1. Validate & stream upload to temp file
      2. Infer source type from filename / content type
      3. Extract, chunk, persist, embed, and index
      4. Return ingestion summary

    The response does NOT include chunk text — use /api/search to query.
    """
    # --- Determine notebook ---
    if notebook_id == "default":
        nb = get_or_create_default()
        notebook_id = nb["id"]
    else:
        nb = get_notebook(notebook_id)
        if nb is None:
            raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")

    # --- Validation ---
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    # --- Stream upload to temp file ---
    safe_filename = sanitize_upload_filename(file.filename)
    suffix = Path(safe_filename).suffix
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    total_size = 0
    chunk_size = 1024 * 1024  # 1 MB
    job_dispatched = False

    try:
        with os.fdopen(fd, "wb") as tmp:
            while True:
                data = await file.read(chunk_size)
                if not data:
                    break
                total_size += len(data)
                if total_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Max {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
                    )
                tmp.write(data)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # --- Source type inference ---
        source_type = infer_source_type(filename=safe_filename)

        # --- PDF magic bytes check (PDF only) ---
        if source_type == "pdf":
            with open(tmp_path, "rb") as fh:
                header = fh.read(5)
            if not header.startswith(b"%PDF-"):
                raise HTTPException(status_code=400, detail="Not a valid PDF file (missing %PDF- header).")

        # --- Build source input ---
        source = build_source_input_for_upload(
            tmp_path=tmp_path,
            filename=safe_filename,
            notebook_id=notebook_id,
        )

        # --- Dispatch a background ingestion job ---
        job_id = create_job(notebook_id=notebook_id)
        asyncio.create_task(asyncio.to_thread(
            _run_ingestion_job,
            job_id, source, tmp_path, use_llm_toc_filter, force_reingest,
        ))
        job_dispatched = True

    finally:
        if not job_dispatched:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return {
        "job_id": job_id,
        "status": "queued",
        "notebook_id": notebook_id,
        "filename": safe_filename,
    }


@router.get("")
async def list_documents_endpoint(
    notebook_id: str = Query("default", description="Notebook ID or 'default'"),
) -> list[dict]:
    """List all documents in a notebook.

    GET /api/documents?notebook_id=default
    """
    if notebook_id == "default":
        nb = get_or_create_default()
        notebook_id = nb["id"]
    else:
        nb = get_notebook(notebook_id)
        if nb is None:
            raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")
        notebook_id = nb["id"]

    docs = list_documents(notebook_id)
    return [
        {
            "document_id": d["id"],
            "filename": d["filename"],
            "display_title": d.get("display_title", ""),
            "content_hash": d.get("content_hash", ""),
            "page_count": d["page_count"],
            "chunk_count": d["chunk_count"],
            "status": d["status"],
            "dense_indexed": d["status"] == "completed",
            "created_at": d["created_at"],
            **document_structure_fields(d),
        }
        for d in docs
    ]


@router.get("/{document_id}")
async def get_document_detail(document_id: str) -> dict:
    """Get a document's metadata and section outline.

    GET /api/documents/{document_id}

    Returns document info plus a list of sections (distinct section_path entries
    with page ranges and chunk counts).
    """
    doc = get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    sections = get_document_sections(document_id)

    return {
        "document_id": doc["id"],
        "filename": doc["filename"],
        "display_title": doc.get("display_title", ""),
        "notebook_id": doc["notebook_id"],
        "page_count": doc["page_count"],
        "chunk_count": doc["chunk_count"],
        "status": doc["status"],
        "dense_indexed": doc["status"] == "completed",
        "created_at": doc["created_at"],
        **document_structure_fields(doc),
        "sections": [
            {
                "title": s["title"],
                "section_path": s["section_path"],
                "page_start": s["page_start"],
                "page_end": s["page_end"],
                "chunk_count": s["chunk_count"],
            }
            for s in sections
        ],
    }


@router.delete("/{document_id}")
async def delete_document_endpoint(document_id: str) -> dict:
    """Delete a document and all its chunks.

    DELETE /api/documents/{document_id}

    Deletes dense vectors FIRST (to avoid stale vector references),
    then removes the document and chunks from SQLite.
    """
    doc = get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    # ── 1. Delete dense vectors first ──
    try:
        SqliteVecStore().delete_by_document_id(document_id)
    except Exception as exc:
        logger.error(
            "Failed to delete dense vectors for document %s: %s",
            document_id, exc,
        )
        raise HTTPException(
            status_code=503,
            detail=f"Failed to clean up document vectors: {exc}",
        )

    # ── 2. Delete extraction artifacts from disk (fail closed) ──
    try:
        ExtractionArtifactStore().delete(document_id)
    except Exception as exc:
        logger.error(
            "Failed to delete extraction artifacts for document %s: %s",
            document_id, exc,
        )
        raise HTTPException(
            status_code=503,
            detail=f"Failed to clean up document files: {exc}",
        )

    # ── 3. Delete from SQLite ──
    deleted = delete_document(document_id)

    return {
        "document_id": document_id,
        "deleted": deleted,
        "filename": doc["filename"],
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def sanitize_upload_filename(filename: str) -> str:
    """Sanitize an upload filename for safe storage."""
    return Path(filename).name.replace("\x00", "")[:255]


def validate_upload_size(total_size: int, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    """Raise HTTPException if file exceeds max size."""
    if total_size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {max_bytes // (1024*1024)} MB.",
        )


def build_source_input_for_upload(
    tmp_path: str,
    filename: str,
    notebook_id: str,
    content_type: str = "",
) -> SourceInput:
    """Build a SourceInput from an uploaded file's metadata."""
    source_type = infer_source_type(filename=filename)
    return SourceInput(
        source_type=source_type,
        file_path=tmp_path,
        filename=filename,
        notebook_id=notebook_id,
        mime_type=content_type,
    )


def response_from_ingestion_result(result: IngestionResult) -> dict:
    """Build the HTTP response dict from an IngestionResult."""
    return {
        "document_id": result.document_id,
        "notebook_id": result.notebook_id,
        "filename": result.filename,
        "display_title": result.display_title or "",
        "source_type": result.source_type,
        "extraction_engine": result.extraction_engine,
        "page_count": result.page_count,
        "chunk_count": result.chunk_count,
        "status": result.status,
        "strategy_used": result.strategy_used,
        "chunking_mode": result.chunking_mode,
        "first_chunk_title": result.first_chunk_title,
        "front_matter_skipped": result.front_matter_skipped,
        "toc_entries_total": result.toc_entries_total,
        "toc_entries_used": result.toc_entries_used,
        "toc_filter_strategy": result.toc_filter_strategy,
        "llm_toc_filter_used": result.llm_toc_filter_used,
        "structure_status": result.structure_status,
        "embedded_toc_present": result.embedded_toc_present,
        "synthetic_outline_mode": result.synthetic_outline_mode,
        "structure_confidence": result.structure_confidence,
        "structure_user_note": result.structure_user_note,
        "structure_llm_instruction": result.structure_llm_instruction,
        "body_start_title": result.body_start_title,
        "body_end_title": result.body_end_title,
        "front_matter_skipped_count": result.front_matter_skipped_count,
        "back_matter_skipped_count": result.back_matter_skipped_count,
        "wrapper_skipped_count": result.wrapper_skipped_count,
        "warnings": result.warnings,
    }
