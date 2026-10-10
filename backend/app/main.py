import logging
import os
import secrets
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import pdf_router, documents_router, search_router, ask_router, generate_router, artifacts_router, config_router, conversations_router, notebooks_router, review_items_router, jobs_router
from .db.session import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise database tables on startup and check component health."""
    init_db()

    # Recover jobs/documents left non-terminal by a previous run (e.g. the app
    # closed mid-ingestion); the existing failed-retry path then allows re-upload.
    try:
        from .repositories.ingestion_job_repository import mark_interrupted_jobs_failed
        recovered = mark_interrupted_jobs_failed()
        if recovered:
            logger.info("Marked %d interrupted ingestion job(s) as failed", recovered)
    except Exception as exc:
        logger.warning("Failed to recover interrupted ingestion jobs: %s", exc)

    # Log component health on startup
    try:
        from .services.system.health import SystemHealthService
        health_results = SystemHealthService.check_all(dense_timeout=3.0)
        for h in health_results:
            if not h.healthy:
                logger.warning("Component %s unhealthy: %s", h.component, h.message)
            else:
                logger.info("Component %s healthy (%s)", h.component, h.message)
    except Exception as exc:
        logger.warning("Health check failed during startup: %s", exc)

    yield


app = FastAPI(title="RAG Study Companion", version="0.2.1", lifespan=lifespan)

# Allow frontend dev servers and static hosting to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",   # VS Code Live Server
        "http://localhost:5500",
        "http://127.0.0.1:5173",   # Vite dev server
        "http://localhost:5173",
        "http://127.0.0.1:8000",   # local FastAPI docs
        "http://localhost:8000",
        "null",                    # Electron file:// loads (null origin)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local API token. The Electron main process generates one per launch and passes
# it here via FOLIO_API_TOKEN; every /api/* request must then carry it in the
# X-Auth-Token header, which stops a foreign web page (whose requests arrive
# with Origin: null) from driving the local backend and exfiltrating the LLM
# key. Unset in a bare `uvicorn` dev run, where enforcement is off.
API_TOKEN = os.getenv("FOLIO_API_TOKEN", "")

# Loopback hosts that may call /api/* in tokenless (browser/dev) mode. Anything
# else — including the opaque "null" origin a sandboxed page sends — is rejected.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _is_loopback_origin(origin: str) -> bool:
    """True if *origin* is an http(s) URL whose host is a loopback address."""
    if not origin:
        return False
    try:
        parsed = urlparse(origin)
        return parsed.scheme in {"http", "https"} and parsed.hostname in _LOOPBACK_HOSTS
    except Exception:
        return False


@app.middleware("http")
async def enforce_api_token(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.method != "OPTIONS":
        if API_TOKEN:
            # Electron/desktop mode: require the per-launch token.
            if not secrets.compare_digest(request.headers.get("x-auth-token", ""), API_TOKEN):
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        else:
            # Tokenless browser/dev mode: reject opaque (null) and foreign
            # origins so a web page cannot drive the local backend (CORS blocks
            # reading the response, not the request). No-Origin CLI/test calls
            # are still allowed.
            origin = request.headers.get("origin", "")
            if origin and not _is_loopback_origin(origin):
                return JSONResponse(status_code=403, content={"detail": "Forbidden origin"})
    return await call_next(request)


app.include_router(pdf_router.router)
app.include_router(documents_router.router)
app.include_router(search_router.router)
app.include_router(ask_router.router)
app.include_router(generate_router.router)
app.include_router(artifacts_router.router)
app.include_router(config_router.router)
app.include_router(conversations_router.router)
app.include_router(notebooks_router.router)
app.include_router(review_items_router.router)
app.include_router(jobs_router.router)


@app.get("/health")
async def health():
    """Return component-level health status."""
    try:
        from .services.system.health import SystemHealthService
        results = SystemHealthService.check_all(dense_timeout=2.0)
        all_healthy = all(h.healthy for h in results)
        return {
            "status": "ok" if all_healthy else "degraded",
            "components": [
                {
                    "component": h.component,
                    "healthy": h.healthy,
                    "message": h.message,
                    "latency_ms": h.latency_ms,
                }
                for h in results
            ],
        }
    except Exception as exc:
        logger.warning("Health check failed, returning legacy response: %s", exc)
        return {"status": "ok"}
