"""FastAPI router for app configuration — LLM setup, status, and test."""

from __future__ import annotations

import logging
import os
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.env_file import set_env
from ..services.llm import LLMConfig
from ..services.web_search import get_web_search_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_format": "openai-compatible",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-5",
        "api_format": "anthropic",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.5-flash",
        "api_format": "google",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_format": "openai-compatible",
    },
    "local": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3",
        "api_format": "openai-compatible",
    },
}


class LLMConfigRequest(BaseModel):
    provider: str = Field("", description="Preset provider key or 'custom'")
    api_format: str = Field("", description="openai-compatible | anthropic | google")
    api_key: str = Field("", description="LLM API key")
    base_url: str = Field("", description="OpenAI-compatible base URL")
    model: str = Field("", description="Model name")
    disabled: bool = Field(False, description="When true, disable the LLM and clear the API key")


class LLMStatusResponse(BaseModel):
    configured: bool
    provider: str = ""
    api_format: str = ""
    base_url: str = ""
    model: str = ""
    has_api_key: bool = False


class TestLLMResponse(BaseModel):
    ok: bool
    model: str = ""
    latency_ms: float = 0
    error: str = ""


class WebSearchConfigRequest(BaseModel):
    provider: str = Field("tavily", description="tavily | custom | disabled")
    api_key: str = Field("", description="Provider API key (shared by tavily/custom)")
    base_url: str = Field("", description="Custom provider base URL")


class WebSearchStatusResponse(BaseModel):
    configured: bool
    provider: str = ""
    base_url: str = ""


class TestWebSearchResponse(BaseModel):
    ok: bool
    provider: str = ""
    latency_ms: float = 0
    result_count: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_cached_llm_services() -> None:
    """Force future requests to use freshly saved provider settings."""
    try:
        from . import ask_router
        ask_router._agent = None
    except Exception:
        logger.debug("Could not reset ask agent cache", exc_info=True)

    try:
        from . import generate_router
        generate_router._llm_client = None
        generate_router._summary_generator = None
        generate_router._flashcards_generator = None
        generate_router._quiz_generator = None
        generate_router._mindmap_generator = None
    except Exception:
        logger.debug("Could not reset generation caches", exc_info=True)

    try:
        from . import pdf_router
        cfg = LLMConfig()
        pdf_router._llm_config = cfg
        pdf_router._llm_classifier = pdf_router.LLMTOCClassifier(cfg) if cfg.is_configured else None
        pdf_router._chunker = pdf_router.PDFChunker(toc_llm_classifier=pdf_router._llm_classifier)
    except Exception:
        logger.debug("Could not reset PDF chunker cache", exc_info=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/llm-status", response_model=LLMStatusResponse)
async def llm_status() -> LLMStatusResponse:
    """Return whether the LLM is currently configured."""
    cfg = LLMConfig()
    return LLMStatusResponse(
        configured=cfg.is_configured,
        provider=cfg.provider or "",
        api_format=cfg.api_format or "",
        base_url=cfg.base_url or "",
        model=cfg.model or "",
        has_api_key=bool(cfg.api_key),
    )


@router.get("/embedding-status")
async def embedding_status() -> dict:
    """Return the local embedding model's load status (non-blocking).

    Triggers a background warm-up on first call so the frontend can poll this
    to show a "preparing semantic search" state until the model is ready.
    """
    from ..services.embedding.sentence_transformer_embedder import get_status, warm_up

    status = get_status()
    if status["status"] == "not_loaded":
        warm_up()
    return status


@router.post("/llm")
async def save_llm_config(request: LLMConfigRequest) -> dict:
    """Save LLM configuration to environment (.env)."""
    if request.disabled:
        set_env("LLM_REMOTE_ENABLED", "false")
        set_env("LLM_API_KEY", "")
        _reset_cached_llm_services()
        return {"ok": True, "disabled": True, "configured": False}

    provider = request.provider.strip().lower()

    # Resolve preset
    base_url = request.base_url.strip()
    model = request.model.strip()
    api_format = request.api_format.strip().lower()

    if provider and provider in PRESETS:
        preset = PRESETS[provider]
        if not base_url:
            base_url = preset["base_url"]
        if not model:
            model = preset["model"]
        if not api_format:
            api_format = preset["api_format"]

    if not api_format:
        api_format = "openai-compatible"

    if api_format not in {"openai-compatible", "anthropic", "google"}:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM API format: {api_format}")

    api_key = request.api_key.strip()
    if provider == "local":
        # Local servers need no key; never persist a cloud key for them.
        api_key = ""
    elif not api_key:
        # Keep the previous key only when the endpoint is unchanged (same
        # provider, base URL, and API format); otherwise the old key would be
        # sent to a different endpoint.
        prev_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        prev_base = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
        prev_format = os.getenv("LLM_API_FORMAT", "").strip().lower()
        same_endpoint = (
            provider == prev_provider
            and base_url.strip().rstrip("/") == prev_base
            and api_format == prev_format
        )
        api_key = os.getenv("LLM_API_KEY", "") if same_endpoint else ""

    set_env("LLM_PROVIDER", provider)
    set_env("LLM_API_FORMAT", api_format)
    set_env("LLM_API_KEY", api_key)
    set_env("LLM_BASE_URL", base_url)
    set_env("LLM_MODEL", model)
    set_env("LLM_REMOTE_ENABLED", "true")
    _reset_cached_llm_services()

    configured = provider == "local" or bool(api_key)
    return {"ok": True, "provider": provider, "api_format": api_format, "model": model, "configured": configured}


@router.post("/test-llm", response_model=TestLLMResponse)
async def test_llm() -> TestLLMResponse:
    """Test the current LLM configuration with a minimal completion."""
    cfg = LLMConfig()
    if not cfg.is_configured:
        return TestLLMResponse(ok=False, error="LLM not configured. Set up an API provider in Settings.")

    try:
        from ..services.llm.client import create_llm_client

        client = create_llm_client(cfg)
        t0 = time.monotonic()
        response = client.complete_chat(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
        )
        elapsed = (time.monotonic() - t0) * 1000
        return TestLLMResponse(
            ok=True,
            model=cfg.model,
            latency_ms=round(elapsed, 1),
        )
    except Exception as exc:
        logger.warning("LLM test failed: %s", exc)
        return TestLLMResponse(ok=False, error=str(exc))


@router.get("/presets")
async def get_presets() -> dict:
    """Return available LLM provider presets (without keys)."""
    return {"presets": PRESETS}


@router.get("/web-search-status", response_model=WebSearchStatusResponse)
async def web_search_status() -> WebSearchStatusResponse:
    """Return whether internet research is configured."""
    provider = os.getenv("WEB_SEARCH_PROVIDER", "disabled").strip().lower()
    api_key = os.getenv("WEB_SEARCH_API_KEY", "") or os.getenv("TAVILY_API_KEY", "")
    base_url = os.getenv("WEB_SEARCH_BASE_URL", "")
    if provider == "tavily":
        configured = bool(api_key)
    elif provider == "custom":
        configured = bool(base_url)
    else:
        configured = False
        provider = ""
    return WebSearchStatusResponse(
        configured=configured,
        provider=provider,
        base_url=base_url,
    )


@router.post("/web-search")
async def save_web_search_config(request: WebSearchConfigRequest) -> dict:
    """Save web search provider configuration to environment (.env)."""
    provider = (request.provider or "disabled").strip().lower()
    if provider not in {"disabled", "tavily", "custom"}:
        raise HTTPException(status_code=400, detail=f"Unsupported web search provider: {provider}")

    api_key = request.api_key.strip()
    base_url = request.base_url.strip()

    if provider == "disabled":
        set_env("WEB_SEARCH_PROVIDER", "disabled")
    elif provider == "tavily":
        if not api_key:
            api_key = os.getenv("WEB_SEARCH_API_KEY", "") or os.getenv("TAVILY_API_KEY", "")
        set_env("WEB_SEARCH_PROVIDER", "tavily")
        set_env("WEB_SEARCH_API_KEY", api_key)
        set_env("TAVILY_API_KEY", api_key)  # keep the legacy key in sync
    else:  # custom
        if not base_url:
            raise HTTPException(status_code=400, detail="Base URL is required for a custom web search provider.")
        set_env("WEB_SEARCH_PROVIDER", "custom")
        set_env("WEB_SEARCH_BASE_URL", base_url)
        set_env("WEB_SEARCH_API_KEY", api_key)

    # Force the ask pipeline to pick up the new provider on the next request.
    try:
        from . import ask_router
        ask_router._agent = None
    except Exception:
        pass

    return {"ok": True, "provider": provider, "configured": provider in {"tavily", "custom"}}


@router.post("/test-web-search", response_model=TestWebSearchResponse)
async def test_web_search() -> TestWebSearchResponse:
    """Test the current web search provider with a minimal query."""
    provider = get_web_search_provider()
    if not provider.enabled:
        return TestWebSearchResponse(
            ok=False,
            provider=provider.name,
            error="Web search is not configured. Add a web search provider in Settings.",
        )

    try:
        t0 = time.monotonic()
        results = provider.search("test query", max_results=1)
        elapsed = (time.monotonic() - t0) * 1000
        return TestWebSearchResponse(
            ok=True,
            provider=provider.name,
            latency_ms=round(elapsed, 1),
            result_count=len(results),
        )
    except Exception as exc:
        logger.warning("Web search test failed: %s", exc)
        return TestWebSearchResponse(
            ok=False,
            provider=provider.name,
            error=str(exc),
        )
