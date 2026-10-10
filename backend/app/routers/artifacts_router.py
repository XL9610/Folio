"""FastAPI router — CRUD for saved artifacts."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..repositories.artifact_repository import (
    create_artifact,
    list_artifacts,
    get_artifact,
    delete_artifact,
)
from ._common import resolve_notebook_id

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


class ArtifactResponse(BaseModel):
    id: str
    kind: str
    title: str
    content_json: dict = {}
    source_document_ids: list[str] = []
    params_json: dict = {}
    created_at: str = ""


class ArtifactListResponse(BaseModel):
    notebook_id: str
    artifacts: list[ArtifactResponse]


class CreateArtifactRequest(BaseModel):
    notebook_id: str = "default"
    kind: str = Field(..., description="Artifact kind, e.g. summary, quiz, note")
    title: str = ""
    content_json: dict = Field(default_factory=dict)
    source_document_ids: list[str] = Field(default_factory=list)
    params_json: dict = Field(default_factory=dict)


@router.post("", response_model=ArtifactResponse)
async def create_artifact_endpoint(request: CreateArtifactRequest) -> ArtifactResponse:
    """Create a saved artifact from user-authored UI content, such as a saved chat turn."""
    notebook_id = resolve_notebook_id(request.notebook_id)
    title = request.title.strip() or "Untitled"
    artifact = create_artifact(
        notebook_id=notebook_id,
        kind=request.kind.strip() or "note",
        title=title,
        content=request.content_json,
        source_document_ids=request.source_document_ids,
        params=request.params_json,
    )
    return ArtifactResponse(
        id=artifact["id"],
        kind=artifact["kind"],
        title=artifact["title"],
        content_json=artifact.get("content_json", {}),
        source_document_ids=artifact.get("source_document_ids", []),
        params_json=artifact.get("params_json", {}),
        created_at=artifact.get("created_at", ""),
    )


@router.get("", response_model=ArtifactListResponse)
async def list_artifacts_endpoint(
    notebook_id: str = "default",
    kind: str | None = None,
) -> ArtifactListResponse:
    """List all saved artifacts in a notebook."""
    notebook_id = resolve_notebook_id(notebook_id)

    artifacts = list_artifacts(notebook_id, kind=kind)
    return ArtifactListResponse(
        notebook_id=notebook_id,
        artifacts=[
            ArtifactResponse(
                id=a["id"],
                kind=a["kind"],
                title=a["title"],
                content_json=a.get("content_json", {}),
                source_document_ids=a.get("source_document_ids", []),
                params_json=a.get("params_json", {}),
                created_at=a.get("created_at", ""),
            )
            for a in artifacts
        ],
    )


@router.get("/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact_endpoint(artifact_id: str) -> ArtifactResponse:
    """Get a single artifact by ID."""
    a = get_artifact(artifact_id)
    if a is None:
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found.")
    return ArtifactResponse(
        id=a["id"],
        kind=a["kind"],
        title=a["title"],
        content_json=a.get("content_json", {}),
        source_document_ids=a.get("source_document_ids", []),
        params_json=a.get("params_json", {}),
        created_at=a.get("created_at", ""),
    )


@router.delete("/{artifact_id}")
async def delete_artifact_endpoint(artifact_id: str) -> dict:
    """Delete an artifact by ID."""
    deleted = delete_artifact(artifact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found.")
    return {"status": "deleted", "artifact_id": artifact_id}
