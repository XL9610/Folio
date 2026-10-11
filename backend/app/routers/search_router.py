"""FastAPI router for hybrid search.

POST /api/search
  Query across documents in a notebook using dense + keyword hybrid retrieval.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..repositories.notebook_repository import get_notebook, get_or_create_default
from ..services.retrieval.retriever import HybridRetriever
from ..services.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from ..services.index.sqlite_vec_store import SqliteVecStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


class SearchRequest(BaseModel):
    notebook_id: str = Field("default", description="Notebook ID or 'default'")
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(10, ge=1, le=50, description="Max results")
    document_ids: list[str] | None = Field(None, description="Optional filter: only search these documents")


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    title: str = ""
    section_path: str = ""
    page_start: int = -1
    page_end: int = -1
    dense_score: float = 0.0
    fts_score: float = 0.0
    rrf_score: float = 0.0
    citation: str = ""


class SearchResponse(BaseModel):
    notebook_id: str
    query: str
    results: list[SearchResultItem]
    search_mode: str = "hybrid"
    warnings: list[str] = []
    latency_ms: dict[str, float] = Field(default_factory=dict)


# Lazy-initialised retriever
_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        embedder = SentenceTransformerEmbedder()
        dense = SqliteVecStore()
        _retriever = HybridRetriever(embedder, dense)
    return _retriever


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Search across ingested documents using hybrid retrieval.

    POST /api/search
    Body:
      { "notebook_id": "default", "query": "what is gradient descent?", "top_k": 5 }

    Returns ranked chunks with citation metadata.
    """
    # --- Resolve notebook ---
    if request.notebook_id == "default":
        nb = get_or_create_default()
        notebook_id = nb["id"]
    else:
        nb = get_notebook(request.notebook_id)
        if nb is None:
            raise HTTPException(status_code=404, detail=f"Notebook '{request.notebook_id}' not found.")
        notebook_id = nb["id"]

    # --- Search ---
    retriever = _get_retriever()
    results, meta = retriever.search_with_meta(
        notebook_id=notebook_id,
        query=request.query,
        top_k=request.top_k,
        document_ids=request.document_ids,
    )

    search_mode = "hybrid"
    if not meta.get("dense_available", True):
        search_mode = "keyword_only"

    return SearchResponse(
        notebook_id=notebook_id,
        query=request.query,
        results=[
            SearchResultItem(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                text=r.text,
                title=r.title,
                section_path=r.section_path,
                page_start=r.page_start,
                page_end=r.page_end,
                dense_score=r.dense_score,
                fts_score=r.fts_score,
                rrf_score=r.rrf_score,
                citation=r.citation(),
            )
            for r in results
        ],
        search_mode=search_mode,
        warnings=meta.get("warnings", []),
        latency_ms={
            k: meta.get(k, 0.0)
            for k in ("dense_latency_ms", "keyword_latency_ms", "fusion_latency_ms", "total_latency_ms")
        },
    )
