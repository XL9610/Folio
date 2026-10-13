"""Integration tests for the hierarchical overview query path (Phase 3).

Covers the deterministic whole-source route: flag gating, scope gating,
complete/stale/missing hierarchy handling with a single 6/4 fallback,
summary-context separation from citable raw passages, citation validity,
notebook/document boundary enforcement, trace exposure without leaking
source text, and the ingestion seam (warning on failure, never fatal).
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.core.config import config
from app.db import session as db_session
from app.db.schema import create_tables
from app.repositories.chunk_repository import get_chunks_by_document, insert_chunks
from app.repositories.document_repository import create_source_document
from app.repositories.notebook_repository import create_notebook
from app.services.generation.hierarchical_summary_builder import (
    HierarchicalSummaryBuilder,
)
from app.services.qa.agent_generator import AgentAnswerGenerator
from app.services.qa.agent_planner import PlanResult, ToolCall
from app.services.qa.hierarchical_context import HierarchicalContextAdapter
from app.services.qa.tools import QAToolExecutor, ToolResults

from tests.test_hierarchical_builder import FakeLLM


@pytest.fixture
def test_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    original_conn = getattr(db_session._local, 'conn', None)
    db_session._local.conn = conn
    create_tables(conn)
    yield conn
    db_session._local.conn = original_conn
    conn.close()


def _seed_book(nb, filename="book.pdf", n_chapters=3, per_chapter=5):
    doc = create_source_document(
        nb["id"], filename, page_count=20, content_hash=f"hash-{filename}",
        structure_status="embedded_toc",
    )
    chunks = []
    for i in range(n_chapters * per_chapter):
        chapter = f"Chapter {i // per_chapter + 1}"
        chunks.append({
            "chunk_index": i,
            "text": f"Body text {i} about {chapter.lower()} material. " * 8,
            "title": chapter,
            "section_path": f"{chapter} > Section {i % per_chapter}",
            "page_start": i + 1,
            "page_end": i + 1,
            "chunk_strategy": "test",
        })
    insert_chunks(doc["id"], chunks)
    return doc


def _build_hierarchy(nb_id, doc_id, fake=None):
    builder = HierarchicalSummaryBuilder(llm_client=fake or FakeLLM())
    return builder.build_document(nb_id, doc_id)


def _executor(nb_id, doc_ids, scope):
    plan = PlanResult(task="overview", retrieval_scope=scope, tool_calls=[])
    return QAToolExecutor(retriever=object()).execute(
        plan=plan, notebook_id=nb_id, document_ids=doc_ids,
    )


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr(config, "hierarchical_overview_enabled", True)


# ---------------------------------------------------------------------------
# Route gating
# ---------------------------------------------------------------------------


def test_fact_scope_never_enters_hierarchical_path(test_db, flag_on):
    nb = create_notebook("NB")
    doc = _seed_book(nb)
    _build_hierarchy(nb["id"], doc["id"])
    results = _executor(nb["id"], [doc["id"]], "fact")
    assert results.summary_context == []
    assert not any(t["name"] == "hierarchical_overview" for t in results.trace)


def test_section_scope_never_enters_hierarchical_path(test_db, flag_on):
    nb = create_notebook("NB")
    doc = _seed_book(nb)
    _build_hierarchy(nb["id"], doc["id"])
    results = _executor(nb["id"], [doc["id"]], "section")
    assert results.summary_context == []
    assert not any(t["name"] == "hierarchical_overview" for t in results.trace)


def test_flag_off_no_route(test_db):
    nb = create_notebook("NB")
    doc = _seed_book(nb)
    _build_hierarchy(nb["id"], doc["id"])
    results = _executor(nb["id"], [doc["id"]], "whole_source")
    assert results.summary_context == []
    assert not any(t["name"] == "hierarchical_overview" for t in results.trace)


def test_missing_hierarchy_falls_back_once(test_db, flag_on):
    nb = create_notebook("NB")
    doc = _seed_book(nb)
    results = _executor(nb["id"], [doc["id"]], "whole_source")
    items = [t for t in results.trace if t["name"] == "hierarchical_overview"]
    assert len(items) == 1
    assert items[0]["status"] == "fallback"
    assert items[0]["args"]["reason"] == "missing_or_stale_index"
    # The existing 6/4 route results are untouched (no tool calls here, so
    # the fallback must not crash and must not populate summary context).
    assert results.summary_context == []


def test_stale_hierarchy_falls_back(test_db, flag_on):
    nb = create_notebook("NB")
    doc = _seed_book(nb)
    _build_hierarchy(nb["id"], doc["id"])
    # A new chunk changes the signature -> the cached hierarchy is stale.
    insert_chunks(doc["id"], [{
        "chunk_index": 99, "text": "new body text " * 8,
        "title": "Chapter 4", "section_path": "Chapter 4",
        "page_start": 99, "page_end": 99, "chunk_strategy": "test",
    }])
    results = _executor(nb["id"], [doc["id"]], "whole_source")
    items = [t for t in results.trace if t["name"] == "hierarchical_overview"]
    assert len(items) == 1
    assert items[0]["status"] == "fallback"
    assert items[0]["args"]["reason"] == "missing_or_stale_index"


def test_whole_source_uses_hierarchy_when_complete(test_db, flag_on):
    nb = create_notebook("NB")
    doc = _seed_book(nb)
    _build_hierarchy(nb["id"], doc["id"])
    results = _executor(nb["id"], [doc["id"]], "whole_source")
    items = [t for t in results.trace if t["name"] == "hierarchical_overview"]
    assert len(items) == 1
    assert items[0]["status"] == "ok"
    args = items[0]["args"]
    assert args["documents"] == 1
    assert args["section_nodes"] == 3
    assert args["cache"] == "hit"
    # FakeLLM cites 2 provenance chunks per section: 3 sections x 2 = 6.
    assert args["raw_evidence_count"] == 6

    # Citable evidence is hydrated canonical raw chunks only.
    chunk_ids = {r["id"] for r in get_chunks_by_document(doc["id"])}
    assert results.chunks
    assert all(c.chunk_id in chunk_ids for c in results.chunks)
    assert all(c["chunk_id"] in chunk_ids for c in results.chunk_citations)
    # Summary context is separate, without citation labels.
    assert results.summary_context
    assert all("ref_label" not in card for card in results.summary_context)


def test_evidence_round_robin_across_sections(test_db):
    nb = create_notebook("NB")
    doc = _seed_book(nb, per_chapter=5)
    _build_hierarchy(nb["id"], doc["id"])
    ctx = HierarchicalContextAdapter(nb["id"], [doc["id"]]).prepare()
    assert ctx.status == "ready"
    # First three evidence chunks must come from the three different
    # sections (one pass each), not three from Chapter 1.
    by_id = {r["id"]: r for r in get_chunks_by_document(doc["id"])}
    first_three = [by_id[cid]["section_path"].split(" >")[0]
                   for cid in ctx.evidence_chunk_ids[:3]]
    assert first_three == ["Chapter 1", "Chapter 2", "Chapter 3"]


def test_multi_doc_fair_allocation(test_db, flag_on):
    nb = create_notebook("NB")
    doc_a = _seed_book(nb, "a.pdf")
    doc_b = _seed_book(nb, "b.pdf")
    _build_hierarchy(nb["id"], doc_a["id"])
    _build_hierarchy(nb["id"], doc_b["id"])
    results = _executor(nb["id"], [doc_a["id"], doc_b["id"]], "whole_source")
    assert any(t["name"] == "hierarchical_overview" and t["status"] == "ok"
               for t in results.trace)
    per_doc: dict[str, int] = {}
    for c in results.chunks:
        per_doc[c.document_id] = per_doc.get(c.document_id, 0) + 1
    assert per_doc.get(doc_a["id"], 0) == 5
    assert per_doc.get(doc_b["id"], 0) == 5
    # Every document overview card is in the summary map.
    ids = {card["document_id"] for card in results.summary_context
           if card.get("major_topics")}
    assert ids == {doc_a["id"], doc_b["id"]}


def test_notebook_boundary_enforced(test_db, flag_on):
    nb = create_notebook("NB A")
    nb2 = create_notebook("NB B")
    doc = _seed_book(nb)
    _build_hierarchy(nb["id"], doc["id"])
    # Same document id presented under the wrong notebook must not route.
    results = _executor(nb2["id"], [doc["id"]], "whole_source")
    items = [t for t in results.trace if t["name"] == "hierarchical_overview"]
    assert items and items[0]["status"] == "fallback"
    assert results.summary_context == []


def test_selected_document_boundary_enforced(test_db, flag_on):
    nb = create_notebook("NB")
    doc = _seed_book(nb)
    other = _seed_book(nb, "other.pdf")
    _build_hierarchy(nb["id"], doc["id"])
    _build_hierarchy(nb["id"], other["id"])
    # Only doc requested -> evidence and cards must never include other.
    results = _executor(nb["id"], [doc["id"]], "whole_source")
    assert all(c.document_id == doc["id"] for c in results.chunks)
    assert all(card["document_id"] == doc["id"]
               for card in results.summary_context)


def test_trace_does_not_leak_source_text(test_db, flag_on):
    nb = create_notebook("NB")
    doc = _seed_book(nb)
    _build_hierarchy(nb["id"], doc["id"])
    results = _executor(nb["id"], [doc["id"]], "whole_source")
    trace_json = json.dumps(results.trace)
    assert "Body text" not in trace_json  # no raw chunk text in any trace


def test_generator_separates_summary_map_from_passages(test_db, flag_on):
    """Phase 3.1: the summary map is never rendered into the prompt; the
    hydrated raw passages are the only evidence shown."""
    nb = create_notebook("NB")
    doc = _seed_book(nb)
    _build_hierarchy(nb["id"], doc["id"])
    results = _executor(nb["id"], [doc["id"]], "whole_source")
    assert results.summary_context  # internal collection retained
    context = AgentAnswerGenerator(llm_client=None)._build_context(
        results, {"notebook_id": nb["id"], "document_ids": [doc["id"]]},
    )
    assert "[Summary map]" not in context
    assert "[Book passages]" in context
    assert all(card["document_id"] == doc["id"]
               for card in results.summary_context)


# ---------------------------------------------------------------------------
# Ingestion seam
# ---------------------------------------------------------------------------


def test_ingestion_hierarchy_failure_warns_but_never_fails(
    test_db, monkeypatch, tmp_path, flag_on,
):
    from app.services.ingestion import service
    from app.services.extraction.models import ExtractedDocument
    from app.services.pdf_ingestion.models import Chunk, ChunkingResult, DocumentPage

    nb = create_notebook("NB")

    class _FakeChunker:
        def chunk(self, *_args, **_kwargs) -> ChunkingResult:
            return ChunkingResult(
                document_id="book.pdf",
                pages=[DocumentPage(page_number=1, text="page text")],
                chunks=[Chunk(
                    text="body chunk text " * 10,
                    document_id="book.pdf", chunk_index=0,
                    title="Chapter 1", page_start=1, page_end=1,
                )],
                strategy_used="toc", chunking_mode="hierarchical_toc",
                toc_entries_total=1, toc_entries_used=1,
                first_chunk_title="Chapter 1",
                body_start_title="Chapter 1", body_end_title="Chapter 1",
                front_matter_skipped_count=0,
            )

    class _FakeDense:
        def health_check(self, timeout=None):
            return False, "no vec", 0.0

    monkeypatch.setattr(
        "app.services.generation.hierarchical_summary_builder.HierarchicalSummaryBuilder",
        lambda: _ExplodingBuilder(),
    )

    class _ExplodingBuilder:
        def build_document(self, notebook_id, document_id, force=False):
            raise RuntimeError("fake summary failure")

    pdf = tmp_path / "book.pdf"
    pdf.write_text("x", encoding="utf-8")
    result = service.ingest_source(
        source=service.SourceInput(
            source_type="pdf", file_path=str(pdf),
            filename="book.pdf", notebook_id=nb["id"],
        ),
        chunker_selector=service.ChunkerSelector(pdf_chunker=_FakeChunker()),
        embedder=object(),
        dense_store=_FakeDense(),
        extracted_document=ExtractedDocument(
            title="book", full_text="body chunk text",
            extraction_engine="pymupdf_textbook", metadata={"source_type": "pdf"},
        ),
    )
    # Ingestion itself succeeded; the hierarchy failure is only a warning.
    assert result.status == "fts_only"
    assert any("Hierarchical summaries failed" in w for w in result.warnings)


def test_ingestion_hierarchy_skipped_when_llm_disabled(
    test_db, monkeypatch, tmp_path, flag_on,
):
    from app.services.ingestion import service

    class _DisabledConfig:
        is_configured = False

    monkeypatch.setattr("app.services.llm.LLMConfig", _DisabledConfig)
    called: list = []

    class _SpyBuilder:
        def build_document(self, notebook_id, document_id, force=False):
            called.append(document_id)
            return None

    monkeypatch.setattr(
        "app.services.generation.hierarchical_summary_builder.HierarchicalSummaryBuilder",
        lambda: _SpyBuilder(),
    )
    service._build_hierarchy_for_document("nb", "doc", None) is None
    assert called == []  # never touched a builder, never sent content anywhere
