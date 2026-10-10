"""Integration tests for the FastAPI PDF chunking endpoint."""

from __future__ import annotations

import os
import tempfile

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixture: small test PDF
# ---------------------------------------------------------------------------

@pytest.fixture
def small_pdf_bytes() -> bytes:
    """Create a minimal PDF (1 page, with text, no TOC) and return its bytes."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Test document content. " * 50, fontsize=12)
    doc.set_toc([
        [1, "Chapter 1", 1],
    ])
    buf = doc.tobytes()
    doc.close()
    return buf


@pytest.fixture
def scanned_pdf_bytes() -> bytes:
    """Create a PDF with pages but NO extractable text (simulates scanned PDF)."""
    doc = fitz.open()
    doc.new_page(width=595, height=842)  # empty page, no text inserted
    buf = doc.tobytes()
    doc.close()
    return buf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# M6 - Retrieval Surface Guard tests
# ---------------------------------------------------------------------------


def test_search_result_schema_excludes_synthetic_fields():
    """SearchResultItem must not expose context_prefix, context_mode, or indexed_text."""
    from app.routers.search_router import SearchResultItem

    # Build with only required fields — synthetic fields are not in the schema
    item = SearchResultItem(
        chunk_id="c1",
        document_id="d1",
        text="canonical text",
    )
    data = item.model_dump()
    assert "context_prefix" not in data
    assert "context_mode" not in data
    assert "indexed_text" not in data

    # Even if somehow present in extra, Pydantic ignores by default (v2 ignores extras)
    # Confirm the canonical text field is the only text field
    assert data["text"] == "canonical text"


def test_ask_debug_retrieved_chunks_excludes_prefix():
    """RetrievedChunkDebugResponse must not expose synthetic fields."""
    from app.routers.ask_router import RetrievedChunkDebugResponse

    item = RetrievedChunkDebugResponse(
        passage_id="S1",
        chunk_id="c1",
        document_id="d1",
        text="canonical chunk text",
    )
    data = item.model_dump()
    assert "context_prefix" not in data
    assert "context_mode" not in data
    assert "indexed_text" not in data
    # text must not start with metadata prefix
    assert not data["text"].startswith("[")


def test_debug_retrieved_chunks_dict_excludes_synthetic_fields():
    """_debug_retrieved_chunks output dict must not contain synthetic fields."""
    from app.routers.ask_router import _debug_retrieved_chunks
    from app.services.qa.tools import ToolResults
    from app.services.retrieval.base import RetrievedChunk

    results = ToolResults()
    results.chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            text="canonical chunk text",
            title="Chapter 1",
            section_path="Chapter 1 > Section A",
            page_start=1,
            page_end=1,
        ),
    ]

    debug = _debug_retrieved_chunks(results)
    assert len(debug) >= 1
    for item in debug:
        assert "context_prefix" not in item
        assert "context_mode" not in item
        assert "indexed_text" not in item
        assert "text" in item
        import re
        assert not re.match(r"^\[[^]]+ > [^]]+\]", item.get("text", "")), (
            f"Metadata prefix leaked into debug retrieved chunk: {item.get('text', '')[:80]}"
        )


# ---------------------------------------------------------------------------


def test_health_endpoint():
    """Health check should return status and component details."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] in ("ok", "degraded")
    assert "components" in data
    assert isinstance(data["components"], list)
    component_names = {c["component"] for c in data["components"]}
    assert "sqlite" in component_names
    assert "dense" in component_names


def test_run_ingestion_job_writes_result_status_and_warnings(monkeypatch):
    """Job success writes result.status into stage and result.warnings into warnings_json."""
    from unittest.mock import MagicMock
    from app.routers import documents_router
    from app.services.ingestion.service import SourceInput

    result = MagicMock()
    result.document_id = "doc-1"
    result.status = "fts_only"
    result.warnings = ["dense unavailable"]

    update_calls = []

    def fake_update(job_id, **fields):
        update_calls.append(fields)

    monkeypatch.setattr(documents_router, "ingest_source", lambda *a, **kw: result)
    monkeypatch.setattr("app.repositories.ingestion_job_repository.update_job", fake_update)

    source = SourceInput(source_type="txt", file_path="x.txt", filename="x.txt", notebook_id="nb1")
    documents_router._run_ingestion_job("job-1", source, "tmp-does-not-exist", False, False)

    completed = [c for c in update_calls if c.get("status") == "completed"]
    assert completed
    assert completed[0].get("stage") == "fts_only"
    import json
    assert json.loads(completed[0].get("warnings_json", "[]")) == ["dense unavailable"]


def test_chunk_endpoint_rejects_non_pdf():
    """Uploading a non-PDF should return 400."""
    resp = client.post(
        "/api/pdf/chunk",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400


def test_chunk_endpoint_no_filename():
    """Upload with no filename should return 400 or 422."""
    resp = client.post("/api/pdf/chunk")
    assert resp.status_code in (400, 422)


def test_chunk_endpoint_pdf_upload(small_pdf_bytes):
    """Upload a valid PDF and verify chunking result structure."""
    resp = client.post(
        "/api/pdf/chunk",
        files={"file": ("test.pdf", small_pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 200, resp.text

    data = resp.json()
    assert data["filename"] == "test.pdf"
    assert data["strategy_used"] in ("toc", "token_fallback")
    assert data["total_chunks"] >= 1
    assert len(data["chunks"]) == data["total_chunks"]

    # Verify chunk structure
    for chunk in data["chunks"]:
        assert "text" in chunk
        assert "document_id" in chunk
        assert "chunk_index" in chunk
        assert "page_start" in chunk
        assert "page_end" in chunk
        assert "chunk_strategy" in chunk
        assert chunk["text"], "Chunk text should not be empty"


def test_scanned_pdf_returns_422(scanned_pdf_bytes):
    """A PDF with pages but no extractable text should return 422."""
    resp = client.post(
        "/api/pdf/chunk",
        files={"file": ("scanned.pdf", scanned_pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text}"
    assert "text" in resp.json()["detail"].lower()


def test_corrupted_pdf_returns_422():
    """Uploading garbage bytes claiming to be PDF should return 422."""
    resp = client.post(
        "/api/pdf/chunk",
        files={"file": ("corrupt.pdf", b"not a pdf at all", "application/pdf")},
    )
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text}"


def test_toc_filter_metadata_in_api_response(small_pdf_bytes):
    """API response should always include TOC filter metadata fields."""
    resp = client.post(
        "/api/pdf/chunk",
        files={"file": ("test.pdf", small_pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "toc_entries_total" in data
    assert "toc_entries_used" in data
    assert "front_matter_skipped" in data
    assert "structure_status" in data
    assert "embedded_toc_present" in data
    assert "synthetic_outline_mode" in data
    assert "structure_confidence" in data
    assert "structure_user_note" in data
    assert "structure_llm_instruction" in data
    assert isinstance(data["toc_entries_total"], int)
    assert isinstance(data["front_matter_skipped"], bool)
    assert data["structure_status"] == "embedded_toc"
