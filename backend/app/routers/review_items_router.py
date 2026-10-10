"""FastAPI router - Review Basket items."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..repositories.artifact_repository import get_artifact
from ._common import resolve_notebook_id
from ..repositories.review_item_repository import (
    delete_review_item,
    list_review_items,
    upsert_review_item,
)

router = APIRouter(prefix="/api/review-items", tags=["review-items"])


class ReviewItemResponse(BaseModel):
    id: str
    notebook_id: str
    artifact_id: str
    item_type: str
    item_index: int
    prompt: str
    answer: str = ""
    choices: list[str] = Field(default_factory=list)
    explanation: str = ""
    source_document_id: str = ""
    source_chunk_id: str = ""
    section_path: str = ""
    page_start: int = -1
    page_end: int = -1
    is_starred: bool = False
    is_mistake: bool = False
    last_user_answer: str = ""
    created_at: str = ""
    updated_at: str = ""


class ReviewItemListResponse(BaseModel):
    notebook_id: str
    review_items: list[ReviewItemResponse]


class UpsertReviewItemRequest(BaseModel):
    notebook_id: str = "default"
    artifact_id: str = Field(..., min_length=1)
    item_type: str = Field(..., pattern="^(flashcard|quiz_question)$")
    item_index: int = Field(..., ge=0)
    prompt: str = Field(..., min_length=1)
    answer: str = ""
    choices: list[str] = Field(default_factory=list)
    explanation: str = ""
    source_document_id: str = ""
    source_chunk_id: str = ""
    section_path: str = ""
    page_start: int = -1
    page_end: int = -1
    is_starred: bool = False
    is_mistake: bool = False
    last_user_answer: str = ""


def _to_response(item: dict) -> ReviewItemResponse:
    return ReviewItemResponse(
        id=item["id"],
        notebook_id=item["notebook_id"],
        artifact_id=item["artifact_id"],
        item_type=item["item_type"],
        item_index=item["item_index"],
        prompt=item["prompt"],
        answer=item.get("answer", ""),
        choices=item.get("choices", []),
        explanation=item.get("explanation", ""),
        source_document_id=item.get("source_document_id", ""),
        source_chunk_id=item.get("source_chunk_id", ""),
        section_path=item.get("section_path", ""),
        page_start=item.get("page_start", -1),
        page_end=item.get("page_end", -1),
        is_starred=bool(item.get("is_starred")),
        is_mistake=bool(item.get("is_mistake")),
        last_user_answer=item.get("last_user_answer", ""),
        created_at=item.get("created_at", ""),
        updated_at=item.get("updated_at", ""),
    )


@router.get("", response_model=ReviewItemListResponse)
async def list_review_items_endpoint(
    notebook_id: str = "default",
    filter: str = Query("all", pattern="^(all|mistakes|starred|flashcards|quiz)$"),
) -> ReviewItemListResponse:
    resolved_notebook_id = resolve_notebook_id(notebook_id)
    items = list_review_items(resolved_notebook_id, filter_by=filter)  # type: ignore[arg-type]
    return ReviewItemListResponse(
        notebook_id=resolved_notebook_id,
        review_items=[_to_response(i) for i in items],
    )


@router.post("", response_model=ReviewItemResponse)
async def upsert_review_item_endpoint(request: UpsertReviewItemRequest) -> ReviewItemResponse:
    resolved_notebook_id = resolve_notebook_id(request.notebook_id)
    artifact = get_artifact(request.artifact_id)
    if artifact is None or artifact.get("notebook_id") != resolved_notebook_id:
        raise HTTPException(status_code=404, detail=f"Artifact '{request.artifact_id}' not found.")

    item = upsert_review_item(
        notebook_id=resolved_notebook_id,
        artifact_id=request.artifact_id,
        item_type=request.item_type,
        item_index=request.item_index,
        prompt=request.prompt.strip(),
        answer=request.answer.strip(),
        choices=request.choices,
        explanation=request.explanation.strip(),
        source_document_id=request.source_document_id.strip(),
        source_chunk_id=request.source_chunk_id.strip(),
        section_path=request.section_path.strip(),
        page_start=request.page_start,
        page_end=request.page_end,
        is_starred=request.is_starred,
        is_mistake=request.is_mistake,
        last_user_answer=request.last_user_answer.strip(),
    )
    return _to_response(item)


@router.delete("/{item_id}")
async def delete_review_item_endpoint(item_id: str) -> dict:
    deleted = delete_review_item(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Review item '{item_id}' not found.")
    return {"status": "deleted", "review_item_id": item_id}
