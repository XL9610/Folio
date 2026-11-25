"""Tests for Phase 4 routes D (leaf expansion) and E (simple raw baseline).

Covers data flow, budgets, section quotas, evidence-only reader, graceful
degradation without summary cards, planner augmentation, and the
no-extra-query-time-LLM invariant (a non-callable retriever would crash
if either route touched it).
"""

from __future__ import annotations

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
from app.services.qa.agent_planner import (
    ToolCall,
    _augment_overview_tool_calls,
)
from app.services.qa.tools import QAToolExecutor, ToolResults
from app.services.retrieval.base import RetrievedChunk

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


def _seed(n_chapters, per_chapter, test_db, nb, build_hierarchy=True,
          content_hash="h"):
    doc = create_source_document(
        nb["id"], "book.pdf", page_count=20, content_hash=content_hash,
        structure_status="embedded_toc",
    )
    chunks = []
    for i in range(n_chapters * per_chapter):
        chapter = f"Chapter {i // per_chapter + 1}"
        chunks.append({
            "chunk_index": i,
            "text": f"Body {i} in {chapter} about topic. " * 8,
            "title": chapter,
            "section_path": chapter,
            "page_start": i + 1,
            "page_end": i + 1,
            "chunk_strategy": "test",
        })
    insert_chunks(doc["id"], chunks)
    if build_hierarchy:
        HierarchicalSummaryBuilder(llm_client=FakeLLM()).build_document(
            nb["id"], doc["id"],
        )
    return doc


def _as_relevance(chunk_rows):
    return [RetrievedChunk(
        chunk_id=r["id"], document_id=r["document_id"], text=r["text"],
        title=r.get("title", ""), section_path=r.get("section_path", ""),
        page_start=r.get("page_start", -1), page_end=r.get("page_end", -1),
    ) for r in chunk_rows]


def _executor():
    return QAToolExecutor(retriever=object())


# ---------------------------------------------------------------------------
# E — expanded raw
# ---------------------------------------------------------------------------


def test_expanded_raw_sets_double_budget_and_evidence_only(test_db, monkeypatch):
    monkeypatch.setattr(config, "overview_expanded_raw", True)
    nb = create_notebook("NB")
    doc = _seed(3, 2, test_db, nb)
    results = ToolResults(chunks=_as_relevance(get_chunks_by_document(doc["id"])))
    from app.services.qa.agent_planner import PlanResult
    _executor()._apply_expanded_raw(
        results, PlanResult(task="overview", retrieval_scope="whole_source"),
        nb["id"], [doc["id"]],
    )
    assert results.evidence_budget == 20
    assert results.evidence_only is True
    assert any(t["name"] == "expanded_raw" and t["status"] == "ok"
               for t in results.trace)


def test_expanded_raw_ignored_for_fact_scope(test_db, monkeypatch):
    monkeypatch.setattr(config, "overview_expanded_raw", True)
    from app.services.qa.agent_planner import PlanResult
    results = ToolResults()
    _executor()._apply_expanded_raw(
        results, PlanResult(task="x", retrieval_scope="fact"), "nb", ["d"],
    )
    assert results.evidence_budget == 10
    assert results.evidence_only is False


def test_expanded_raw_planner_widens_retrieve_and_skips_coverage(monkeypatch):
    monkeypatch.setattr(config, "overview_expanded_raw", True)
    routed = _augment_overview_tool_calls(
        "overview of the whole book", [], ["d1"], planner_scope="whole_source",
    )
    names = [t.name for t in routed]
    assert "get_source_chunks" not in names
    retrieve = [t for t in routed if t.name == "retrieve_chunks"][0]
    assert retrieve.args["top_k"] == 20


# ---------------------------------------------------------------------------
# D — leaf expansion
# ---------------------------------------------------------------------------


class _FakeRetriever:
    """Returns candidate chunks in the requested order (within-candidate
    ranking is the caller's job; D only asks for the candidate set)."""

    def __init__(self, rows_by_id):
        self._rows = {r["id"]: r for r in rows_by_id}
        self.last_chunk_ids = None

    def search(self, notebook_id, query, top_k=10, document_ids=None,
               options=None, chunk_ids=None):
        self.last_chunk_ids = chunk_ids
        out = []
        for cid in (chunk_ids or []):
            r = self._rows.get(cid)
            if r:
                out.append(RetrievedChunk(
                    chunk_id=cid, document_id=r["document_id"], text=r["text"],
                    title=r.get("title", ""), section_path=r.get("section_path", ""),
                    page_start=r.get("page_start", -1),
                    page_end=r.get("page_end", -1),
                ))
        return out[:top_k]


def _executor_with(retriever):
    return QAToolExecutor(retriever=retriever)


def _plan(scope="whole_source"):
    from app.services.qa.agent_planner import PlanResult
    return PlanResult(task="overview", retrieval_scope=scope)


def test_leaf_expansion_selects_sections_and_excludes_others(test_db, monkeypatch):
    monkeypatch.setattr(config, "overview_leaf_expansion", True)
    nb = create_notebook("NB")
    doc = _seed(5, 2, test_db, nb)  # Chapter 1..5, 2 chunks each
    rows = get_chunks_by_document(doc["id"])
    results = ToolResults()  # no global relevance at all
    _executor_with(_FakeRetriever(rows))._apply_leaf_expansion(
        results, _plan(), nb["id"], [doc["id"]],
        user_query="give an overview of the whole work",
    )
    # Ordinal selection (cards carry no query-distinguishing terms) picks the
    # first 4 of 5 sections; the 5th must be excluded entirely.
    included = {c.section_path for c in results.chunks}
    assert included == {"Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4"}
    assert results.evidence_only is True
    assert any(t["name"] == "leaf_expansion" for t in results.trace)


def test_leaf_expansion_searches_within_candidates_not_global(test_db, monkeypatch):
    """Fix 1: a candidate chunk absent from global results still reaches the
    reader because D re-runs retrieval over the candidate set, not the
    global top-K."""
    monkeypatch.setattr(config, "overview_leaf_expansion", True)
    nb = create_notebook("NB")
    doc = _seed(4, 5, test_db, nb)  # 4 chapters x 5 = 20 candidates
    rows = get_chunks_by_document(doc["id"])
    retriever = _FakeRetriever(rows)
    results = ToolResults()  # global relevance empty -> old filter would yield []
    _executor_with(retriever)._apply_leaf_expansion(
        results, _plan(), nb["id"], [doc["id"]], user_query="the whole work",
    )
    assert retriever.last_chunk_ids  # D passed the candidate set to search
    assert len(results.chunks) == 10
    assert all(c.chunk_id in {r["id"] for r in rows} for c in results.chunks)


def test_leaf_expansion_candidate_cap_is_cross_section(test_db, monkeypatch):
    """Fix 2: a 61-chunk first section must not starve the other selected
    sections out of the 60-candidate cap."""
    monkeypatch.setattr(config, "overview_leaf_expansion", True)
    nb = create_notebook("NB")
    doc = create_source_document(
        nb["id"], "lopsided.pdf", content_hash="h", structure_status="embedded_toc",
    )
    chunks = []
    for i in range(61):
        chunks.append({"chunk_index": i, "text": f"Chapter 1 body {i} " * 8,
                       "title": "Chapter 1", "section_path": "Chapter 1",
                       "page_start": i + 1, "page_end": i + 1, "chunk_strategy": "test"})
    for cno in (2, 3, 4):
        for j in range(2):
            chunks.append({"chunk_index": 100 + cno + j,
                           "text": f"Chapter {cno} body " * 8,
                           "title": f"Chapter {cno}",
                           "section_path": f"Chapter {cno}",
                           "page_start": 200 + cno, "page_end": 200 + cno,
                           "chunk_strategy": "test"})
    insert_chunks(doc["id"], chunks)
    HierarchicalSummaryBuilder(llm_client=FakeLLM()).build_document(
        nb["id"], doc["id"],
    )
    rows = get_chunks_by_document(doc["id"])
    results = ToolResults()
    _executor_with(_FakeRetriever(rows))._apply_leaf_expansion(
        results, _plan(), nb["id"], [doc["id"]], user_query="the whole work",
    )
    sections = {c.section_path for c in results.chunks}
    # Cross-section allocation keeps later sections in the candidate pool.
    assert {"Chapter 2", "Chapter 3", "Chapter 4"} & sections


def test_leaf_expansion_degrades_to_ordinal_without_cards(test_db, monkeypatch):
    monkeypatch.setattr(config, "overview_leaf_expansion", True)
    nb = create_notebook("NB")
    doc = _seed(5, 2, test_db, nb, build_hierarchy=False)  # no summary nodes
    rows = get_chunks_by_document(doc["id"])
    results = ToolResults()
    _executor_with(_FakeRetriever(rows))._apply_leaf_expansion(
        results, _plan(), nb["id"], [doc["id"]],
        user_query="give an overview of the whole work",
    )
    # Ordinal fallback selects the first 4 sections; Chapter 5 excluded.
    included = {c.section_path for c in results.chunks}
    assert "Chapter 1" in included
    assert "Chapter 5" not in included
    assert results.evidence_only is True


def test_leaf_expansion_candidate_cap_is_per_document(test_db, monkeypatch):
    """Two documents each get their own 60-candidate cap (120 total), not a
    shared 60 (which would give ~30 each)."""
    monkeypatch.setattr(config, "overview_leaf_expansion", True)
    nb = create_notebook("NB")
    doc_a = _seed(4, 20, test_db, nb, build_hierarchy=False, content_hash="ha")
    doc_b = _seed(4, 20, test_db, nb, build_hierarchy=False, content_hash="hb")
    rows = (get_chunks_by_document(doc_a["id"])
            + get_chunks_by_document(doc_b["id"]))
    retriever = _FakeRetriever(rows)
    results = ToolResults()
    _executor_with(retriever)._apply_leaf_expansion(
        results, _plan(), nb["id"], [doc_a["id"], doc_b["id"]],
        user_query="give an overview of the whole work",
    )
    ids = set(retriever.last_chunk_ids or [])
    assert len(ids) == 120  # 60 per document
    a_ids = {r["id"] for r in get_chunks_by_document(doc_a["id"])}
    b_ids = {r["id"] for r in get_chunks_by_document(doc_b["id"])}
    assert len(ids & a_ids) == 60
    assert len(ids & b_ids) == 60


def test_leaf_expansion_planner_keeps_retrieve_and_skips_coverage(monkeypatch):
    monkeypatch.setattr(config, "overview_leaf_expansion", True)
    routed = _augment_overview_tool_calls(
        "overview of the whole book", [], ["d1"], planner_scope="whole_source",
    )
    names = [t.name for t in routed]
    assert "get_source_chunks" not in names
    assert any(t.name == "retrieve_chunks" for t in routed)


# ---------------------------------------------------------------------------
# Fix 4 — the D/E gate inherits the chapter/section protections
# ---------------------------------------------------------------------------


def _chapter_plan(scope, tool_names=()):
    from app.services.qa.agent_planner import PlanResult, ToolCall
    return PlanResult(
        task="x", retrieval_scope=scope,
        tool_calls=[ToolCall(name=n, args={}) for n in tool_names],
    )


def test_gate_blocks_numbered_chapter_reference_for_fact_scope(test_db, monkeypatch):
    monkeypatch.setattr(config, "overview_leaf_expansion", True)
    monkeypatch.setattr(config, "overview_expanded_raw", True)
    nb = create_notebook("NB")
    doc = _seed(3, 2, test_db, nb)
    rows = get_chunks_by_document(doc["id"])
    # fact scope + "Summarize Chapter 3" must NOT enter D/E.
    results = ToolResults()
    _executor_with(_FakeRetriever(rows))._apply_leaf_expansion(
        results, _chapter_plan("fact"), nb["id"], [doc["id"]],
        user_query="Summarize Chapter 3",
    )
    _executor_with(_FakeRetriever(rows))._apply_expanded_raw(
        results, _chapter_plan("fact"), nb["id"], [doc["id"]],
        user_query="Summarize Chapter 3",
    )
    assert results.evidence_only is False
    assert results.evidence_budget == 10
    assert not any(t["name"] in ("leaf_expansion", "expanded_raw")
                   for t in results.trace)


def test_gate_blocks_section_tool_plans(test_db, monkeypatch):
    monkeypatch.setattr(config, "overview_leaf_expansion", True)
    nb = create_notebook("NB")
    doc = _seed(3, 2, test_db, nb)
    rows = get_chunks_by_document(doc["id"])
    # unknown scope but a find_sections plan = scoped, not overview.
    results = ToolResults()
    _executor_with(_FakeRetriever(rows))._apply_leaf_expansion(
        results, _chapter_plan("unknown", ["find_sections"]),
        nb["id"], [doc["id"]], user_query="what does the methods section say",
    )
    assert not any(t["name"] == "leaf_expansion" for t in results.trace)


# ---------------------------------------------------------------------------
# Evidence-only reader
# ---------------------------------------------------------------------------


def test_evidence_only_reader_shows_only_passages(test_db):
    from app.services.qa.agent_planner import PlanResult
    tr = ToolResults(
        chunks=[RetrievedChunk(
            chunk_id="c1", document_id="d1", text="a passage",
            title="Chapter 1", section_path="Chapter 1", page_start=1,
            page_end=1,
        )],
        metadata=[{"filename": "book.pdf", "page_count": 10, "chunk_count": 5,
                   "structure_status": "embedded_toc", "structure_user_note": "",
                   "structure_llm_instruction": ""}],
        outline=[{"filename": "book.pdf", "sections": [
            {"section_path": "Chapter 1", "title": "Chapter 1",
             "page_start": 1, "page_end": 2, "chunk_count": 5},
        ], "structure_status": "embedded_toc", "structure_llm_instruction": ""}],
        evidence_only=True,
    )
    context = AgentAnswerGenerator(llm_client=None)._build_context(tr, {})
    assert "[Book passages]" in context
    assert "a passage" in context
    assert "[Document" not in context
    assert "[Outline]" not in context
    assert "File:" not in context
    # Without evidence-only, structure would appear.
    tr.evidence_only = False
    context = AgentAnswerGenerator(llm_client=None)._build_context(tr, {})
    assert "[Document" in context and "[Outline]" in context
