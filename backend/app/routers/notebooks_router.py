"""FastAPI router — CRUD for notebooks."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..repositories.document_repository import list_documents
from ..repositories.notebook_repository import (
    create_notebook,
    delete_notebook,
    get_notebook_with_stats,
    list_notebooks,
    touch_notebook,
    update_notebook,
)
from ..services.extraction.artifact_store import ExtractionArtifactStore
from ..services.index.sqlite_vec_store import SqliteVecStore
from ._common import resolve_notebook_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notebooks", tags=["notebooks"])


# ── Request / Response models ─────────────────────────────────────────────


class CreateNotebookRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Notebook name")
    description: str = Field(default="", max_length=200, description="Optional description")


class UpdateNotebookRequest(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=200)


class NotebookResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    source_count: int = 0
    status: str = "empty"
    last_updated: str = ""
    created_at: str = ""
    last_opened_at: str = ""


# ── Endpoints ──────────────────────────────────────────────────────────────


@router.get("", response_model=list[NotebookResponse])
async def list_all() -> list[NotebookResponse]:
    """List all notebooks with computed stats."""
    notebooks = list_notebooks()
    return [
        NotebookResponse(
            id=n["id"],
            name=n["name"],
            description=n.get("description", ""),
            source_count=n.get("source_count", 0),
            status=n.get("status", "empty"),
            last_updated=n.get("last_updated") or n.get("created_at", ""),
            created_at=n.get("created_at", ""),
            last_opened_at=n.get("last_opened_at") or "",
        )
        for n in notebooks
    ]


@router.post("", response_model=NotebookResponse, status_code=201)
async def create(request: CreateNotebookRequest) -> NotebookResponse:
    """Create a new notebook."""
    nb = create_notebook(name=request.name, description=request.description)
    return NotebookResponse(
        id=nb["id"],
        name=nb["name"],
        description=nb.get("description", ""),
        source_count=nb.get("source_count", 0),
        status=nb.get("status", "empty"),
        last_updated=nb.get("last_updated") or nb.get("created_at", ""),
        created_at=nb.get("created_at", ""),
    )


@router.get("/{notebook_id}", response_model=NotebookResponse)
async def get_one(notebook_id: str) -> NotebookResponse:
    """Get a single notebook with computed stats."""
    resolved_id = resolve_notebook_id(notebook_id)
    nb = get_notebook_with_stats(resolved_id)
    if nb is None:
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")
    return NotebookResponse(
        id=nb["id"],
        name=nb["name"],
        description=nb.get("description", ""),
        source_count=nb.get("source_count", 0),
        status=nb.get("status", "empty"),
        last_updated=nb.get("last_updated") or nb.get("created_at", ""),
        created_at=nb.get("created_at", ""),
    )


@router.put("/{notebook_id}", response_model=NotebookResponse)
async def update(notebook_id: str, request: UpdateNotebookRequest) -> NotebookResponse:
    """Update notebook name and/or description."""
    resolved_id = resolve_notebook_id(notebook_id)
    nb = update_notebook(resolved_id, name=request.name, description=request.description)
    if nb is None:
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")
    return NotebookResponse(
        id=nb["id"],
        name=nb["name"],
        description=nb.get("description", ""),
        source_count=nb.get("source_count", 0),
        status=nb.get("status", "empty"),
        last_updated=nb.get("last_updated") or nb.get("created_at", ""),
        created_at=nb.get("created_at", ""),
    )


@router.delete("/{notebook_id}", status_code=204, response_model=None)
async def delete(notebook_id: str) -> None:
    """Delete a notebook and all of its documents (cascading)."""
    resolved_id = resolve_notebook_id(notebook_id)

    # Delete dense vectors and extraction artifacts first so no stale
    # chunk_id references or on-disk full text remain.
    doc_ids = [row["id"] for row in list_documents(resolved_id)]
    artifact_store = ExtractionArtifactStore()
    for doc_id in doc_ids:
        try:
            SqliteVecStore().delete_by_document_id(doc_id)
        except Exception as exc:
            logger.error("Failed to delete dense vectors for document %s: %s", doc_id, exc)
            raise HTTPException(status_code=503, detail=f"Failed to clean up document vectors: {exc}") from exc
        try:
            artifact_store.delete(doc_id)
        except Exception as exc:
            # Fail closed: keep the DB records so the user can retry rather than
            # silently dropping the notebook while plaintext remains on disk.
            logger.error("Failed to delete extraction artifacts for document %s: %s", doc_id, exc)
            raise HTTPException(
                status_code=503,
                detail=f"Failed to clean up document files: {exc}",
            ) from exc

    deleted = delete_notebook(resolved_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")


@router.put("/{notebook_id}/opened")
async def mark_opened(notebook_id: str) -> dict:
    """Record that a notebook was opened (for recently-opened ordering)."""
    resolved_id = resolve_notebook_id(notebook_id)
    if not touch_notebook(resolved_id):
        raise HTTPException(status_code=404, detail=f"Notebook '{notebook_id}' not found.")
    return {"ok": True}
