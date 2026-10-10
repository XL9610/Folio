"""FastAPI router — POST /api/generate for summary generation."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..repositories.notebook_repository import get_notebook, get_or_create_default
from ..repositories.document_repository import get_document, list_documents
from ..repositories.artifact_repository import create_artifact
from ..services.generation.scope_resolver import ScopeResolver
from ..services.generation.summary_service import SummaryGenerator
from ..services.generation.flashcards_service import FlashcardsGenerator
from ..services.generation.quiz_service import QuizGenerator
from ..services.generation.mindmap_service import MindmapGenerator
from ..services.retrieval.retriever import HybridRetriever
from ..services.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from ..services.index.sqlite_vec_store import SqliteVecStore
from ..services.llm import LLMConfig
from ..services.llm.client import OpenAICompatibleClient, create_llm_client
from ..services.qa.types import CitationResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["generate"])

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class GenerateParams(BaseModel):
    scope: str = Field("", description="What to generate from: chapter, page range, topic, or empty for whole doc")
    length: str = Field("medium", description="short | medium | detailed (summary only)")
    style: str = Field("structured", description="prose | bullets | structured (summary only)")
    count: int = Field(20, ge=10, le=50, description="Number of flashcards (10-50)")
    quiz_count: int = Field(10, ge=5, le=30, description="Number of quiz questions (5-30)")
    quiz_difficulty: str = Field("medium", description="basic | medium | hard")
    quiz_requirements: str = Field("", description="User's special instructions for quiz generation")


class GenerateRequest(BaseModel):
    kind: str = Field("summary", description="Generation kind (summary, quiz, etc.)")
    notebook_id: str = Field("default")
    document_ids: list[str] = Field(..., min_length=1, max_length=1)
    params: GenerateParams = Field(default_factory=GenerateParams)


class GenerateResponse(BaseModel):
    artifact_id: str
    kind: str
    title: str
    markdown: str = ""
    cards: list[dict] = []
    questions: list[dict] = []
    stats: dict[str, Any] = {}
    sources: list[CitationResponse] = []
    source_document_ids: list[str] = []
    params: dict[str, Any] = {}
    tree: dict[str, Any] | None = None
    mode: str = ""


# ---------------------------------------------------------------------------
# Lazy services
# ---------------------------------------------------------------------------

_llm_client = None
_summary_generator: SummaryGenerator | None = None
_flashcards_generator: FlashcardsGenerator | None = None
_quiz_generator: QuizGenerator | None = None
_mindmap_generator: MindmapGenerator | None = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        llm_cfg = LLMConfig()
        if not llm_cfg.is_configured:
            raise RuntimeError("LLM not configured")
        api_format = getattr(llm_cfg, "api_format", "openai-compatible")
        _llm_client = OpenAICompatibleClient(llm_cfg) if not isinstance(api_format, str) else create_llm_client(llm_cfg)
    return _llm_client


def _get_summary_generator() -> SummaryGenerator:
    global _summary_generator
    if _summary_generator is None:
        retriever = HybridRetriever(
            embedder=SentenceTransformerEmbedder(),
            dense_store=SqliteVecStore(),
        )
        scope_resolver = ScopeResolver(retriever)
        _summary_generator = SummaryGenerator(_get_llm_client(), scope_resolver)
    return _summary_generator


def _get_flashcards_generator() -> FlashcardsGenerator:
    global _flashcards_generator
    if _flashcards_generator is None:
        retriever = HybridRetriever(
            embedder=SentenceTransformerEmbedder(),
            dense_store=SqliteVecStore(),
        )
        scope_resolver = ScopeResolver(retriever)
        _flashcards_generator = FlashcardsGenerator(_get_llm_client(), scope_resolver)
    return _flashcards_generator


def _get_quiz_generator() -> QuizGenerator:
    global _quiz_generator
    if _quiz_generator is None:
        retriever = HybridRetriever(
            embedder=SentenceTransformerEmbedder(),
            dense_store=SqliteVecStore(),
        )
        scope_resolver = ScopeResolver(retriever)
        _quiz_generator = QuizGenerator(_get_llm_client(), scope_resolver)
    return _quiz_generator


def _get_mindmap_generator() -> MindmapGenerator:
    global _mindmap_generator
    if _mindmap_generator is None:
        retriever = HybridRetriever(
            embedder=SentenceTransformerEmbedder(),
            dense_store=SqliteVecStore(),
        )
        scope_resolver = ScopeResolver(retriever)
        _mindmap_generator = MindmapGenerator(_get_llm_client(), scope_resolver)
    return _mindmap_generator


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate a study artifact (summary/flashcards/quiz/mindmap) from one document.

    Each generation is single-source (document_ids has max_length=1); the
    frontend fans out a separate request per selected source.
    """
    # Check LLM
    llm_cfg = LLMConfig()
    if not llm_cfg.is_configured:
        raise HTTPException(
            status_code=503,
            detail={"code": "llm_not_configured", "message": "Generate requires an LLM."},
        )

    if request.kind not in ("summary", "flashcards", "quiz", "mindmap"):
        raise HTTPException(
            status_code=501,
            detail={"code": "not_implemented", "message": f"Kind '{request.kind}' is not yet supported."},
        )

    # Resolve notebook
    if request.notebook_id == "default":
        nb = get_or_create_default()
        notebook_id = nb["id"]
    else:
        nb = get_notebook(request.notebook_id)
        if nb is None:
            raise HTTPException(status_code=404, detail=f"Notebook '{request.notebook_id}' not found.")
        notebook_id = nb["id"]

    # Validate document_ids belong to this notebook
    all_docs = list_documents(notebook_id)
    valid_ids = {d["id"] for d in all_docs}
    for did in request.document_ids:
        if did not in valid_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{did}' not found in notebook '{request.notebook_id}'.",
            )

    doc_id = request.document_ids[0]
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    doc_filename = doc.get("filename", "")
    scope = request.params.scope

    # ── Dispatch by kind ──
    if request.kind == "summary":
        length = request.params.length or "medium"
        style = request.params.style or "structured"
        valid_lengths = {"short", "medium", "detailed"}
        valid_styles = {"prose", "bullets", "structured"}
        if length not in valid_lengths:
            length = "medium"
        if style not in valid_styles:
            style = "structured"

        try:
            generator = _get_summary_generator()
            result = generator.generate(
                notebook_id=notebook_id,
                document_id=doc_id,
                scope=scope,
                length=length,
                style=style,
                doc_filename=doc_filename,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail={"code": "llm_provider_error", "message": str(exc)})

        content = {"markdown": result["markdown"], "stats": result["stats"]}
        response_markdown = result["markdown"]
        response_cards: list[dict] = []
        response_questions: list[dict] = []
        gen_params = {"scope": scope, "length": length, "style": style}

    elif request.kind == "flashcards":
        count = request.params.count or 20
        try:
            fg = _get_flashcards_generator()
            result = fg.generate(
                notebook_id=notebook_id,
                document_id=doc_id,
                scope=scope,
                count=count,
                doc_filename=doc_filename,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail={"code": "generation_failed", "message": str(exc)})

        content = {"cards": result["cards"], "stats": result["stats"], "message": result.get("message", "")}
        response_markdown = result.get("message", "")
        response_cards = result["cards"]
        response_questions = []
        gen_params = {"scope": scope, "count": count}

    elif request.kind == "quiz":
        quiz_count = request.params.quiz_count or 10
        quiz_difficulty = request.params.quiz_difficulty or "medium"
        valid_difficulties = {"basic", "medium", "hard"}
        if quiz_difficulty not in valid_difficulties:
            quiz_difficulty = "medium"
        try:
            qg = _get_quiz_generator()
            result = qg.generate(
                notebook_id=notebook_id,
                document_id=doc_id,
                scope=scope,
                count=quiz_count,
                difficulty=quiz_difficulty,
                doc_filename=doc_filename,
                requirements=request.params.quiz_requirements,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail={"code": "generation_failed", "message": str(exc)})

        content = {"questions": result["questions"], "stats": result["stats"], "message": result.get("message", "")}
        response_markdown = result.get("message", "")
        response_cards = []
        response_questions = result["questions"]
        gen_params = {"scope": scope, "count": quiz_count, "difficulty": quiz_difficulty, "requirements": request.params.quiz_requirements}

    else:  # mindmap — single source only
        mindmap_doc_id = request.document_ids[0]
        mindmap_doc = get_document(mindmap_doc_id)
        mindmap_filename = mindmap_doc.get("filename", "") if mindmap_doc else ""
        try:
            mg = _get_mindmap_generator()
            result = mg.generate(
                notebook_id=notebook_id,
                document_id=mindmap_doc_id,
                scope=scope,
                doc_filename=mindmap_filename,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail={"code": "generation_failed", "message": str(exc)})

        content = {
            "markdown": result.get("markdown", ""),
            "tree": result.get("tree"),
            "mode": result.get("mode"),
            "stats": result["stats"],
            "sources": result.get("sources", []),
        }
        response_markdown = result.get("markdown", "")
        response_cards = []
        response_questions = []
        gen_params = {"scope": scope}

    # Persist artifact
    artifact = create_artifact(
        notebook_id=notebook_id,
        kind=request.kind,
        title=result["title"],
        content=content,
        source_document_ids=request.document_ids,
        params=gen_params,
    )

    return GenerateResponse(
        artifact_id=artifact["id"],
        kind=request.kind,
        title=result["title"],
        markdown=response_markdown,
        cards=response_cards,
        questions=response_questions,
        stats=result["stats"],
        sources=[CitationResponse(**(c if isinstance(c, dict) else asdict(c))) for c in result.get("sources", [])],
        source_document_ids=request.document_ids,
        params=gen_params,
        tree=result.get("tree"),
        mode=result.get("mode", ""),
    )
