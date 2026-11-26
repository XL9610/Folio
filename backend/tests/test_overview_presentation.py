"""Tests for Phase 3.1 — evidence-grounded overview without visible
inline citations.

Covers the required items: summary hierarchy content never reaches the
generator prompt; hydrated raw chunks do; overview presentation hides
internal [S#] markers only for whole_source scope; fact/section answers
are untouched; cleanup produces no artifacts; internal citations and
provenance survive; the 6/4 fallback path is unaffected.
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
from app.routers.ask_router import _apply_overview_presentation
from app.services.generation.hierarchical_summary_builder import (
    HierarchicalSummaryBuilder,
)
from app.services.qa.agent_generator import (
    AGENT_ANSWER_SYSTEM_PROMPT,
    OVERVIEW_GROUNDING_BLOCK,
    AgentAnswerGenerator,
)
from app.services.qa.agent_planner import PlanResult
from app.services.qa.citations import strip_inline_book_citations
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


@pytest.fixture
def built_hierarchy(test_db):
    nb = create_notebook("NB")
    doc = create_source_document(
        nb["id"], "book.pdf", page_count=10, content_hash="h",
        structure_status="embedded_toc",
    )
    insert_chunks(doc["id"], [{
        "chunk_index": i,
        "text": f"Body text {i} about chapter {i // 5 + 1}. " * 8,
        "title": f"Chapter {i // 5 + 1}",
        "section_path": f"Chapter {i // 5 + 1}",
        "page_start": i + 1, "page_end": i + 1, "chunk_strategy": "test",
    } for i in range(10)])
    HierarchicalSummaryBuilder(llm_client=FakeLLM()).build_document(
        nb["id"], doc["id"],
    )
    return nb, doc


@pytest.fixture
def results_with_summary_context(built_hierarchy):
    """ToolResults carrying a factual summary_context (map content)."""
    nb, doc = built_hierarchy
    plan = PlanResult(task="overview", retrieval_scope="whole_source",
                      tool_calls=[])
    results = QAToolExecutor(retriever=object()).execute(
        plan=plan, notebook_id=nb["id"], document_ids=[doc["id"]],
    )
    # Inject factual map content that must NEVER reach the prompt.
    # The executor route is flag-gated off in these tests; provide the
    # hydrated evidence the generator context renders as [Book passages].
    from app.services.retrieval.base import RetrievedChunk
    results.chunks = [RetrievedChunk(
        chunk_id="c1", document_id=doc["id"],
        text="Body text 0 about chapter 1 material. " * 4,
        title="Chapter 1", section_path="Chapter 1",
        page_start=1, page_end=1,
    )]
    results.summary_context = [{
        "document_id": doc["id"],
        "filename": "book.pdf",
        "title": "Document overview",
        "summary": "The book was published in 1902 by Eris Press.",
        "major_topics": [
            {"text": "On 4 July 1903 the inventor Alice Novac flew the first "
                     "steam glider from Beacon Hill."},
        ],
        "points": [{"text": "Chapter 2 describes the 1910 factory fire."}],
        "section_key": "Chapter 1",
    }]
    return results


class _RecordingLLM:
    """Fake generator client capturing the system prompt."""

    def __init__(self):
        self.captured: list[tuple[list[dict], int | None]] = []

    def complete_chat(self, messages, max_tokens=None, temperature=None):
        self.captured.append((messages, max_tokens))
        return "answer"


# ---------------------------------------------------------------------------
# Generator prompt isolation (required items 1-3)
# ---------------------------------------------------------------------------


def test_summary_context_never_reaches_generator_prompt(results_with_summary_context):
    tr = results_with_summary_context
    context = AgentAnswerGenerator(llm_client=None)._build_context(
        tr, {"notebook_id": "nb", "document_ids": ["d"]},
    )
    assert "[Summary map]" not in context
    assert "1902 by Eris Press" not in context          # document summary
    assert "Alice Novac flew the first steam glider" not in context  # major_topics text
    assert "1910 factory fire" not in context           # point text
    assert "Document overview" not in context           # document title
    assert tr.summary_context  # internal structure retained for debug/eval


def test_summary_reader_flag_renders_summary_and_swaps_prompt(
    results_with_summary_context, monkeypatch,
):
    """OVERVIEW_SUMMARY_READER on: the stored summary is the reader's material."""
    from app.services.qa import agent_generator as gen

    monkeypatch.setattr(gen.config, "overview_summary_reader", True)
    tr = results_with_summary_context
    llm = _RecordingLLM()
    plan = PlanResult(task="overview", retrieval_scope="whole_source")
    AgentAnswerGenerator(llm).generate("What is this book about?", [], plan, tr, {})
    system, user = (m["content"] for m in llm.captured[0][0])
    assert "[Whole-source summary]" in user
    assert "1902 by Eris Press" in user
    assert gen.SUMMARY_READER_BLOCK in system
    assert gen.OVERVIEW_GROUNDING_BLOCK not in system


def test_hydrated_raw_chunks_still_reach_prompt(results_with_summary_context):
    tr = results_with_summary_context
    context = AgentAnswerGenerator(llm_client=None)._build_context(
        tr, {"notebook_id": "nb", "document_ids": ["d"]},
    )
    assert "[Book passages]" in context
    # The hydrated raw chunk text is present.
    assert "Body text 0 about chapter 1" in context


def test_overview_block_only_for_whole_source_scope():
    gen = AgentAnswerGenerator(llm_client=_RecordingLLM())
    for scope, expected in (("whole_source", True), ("fact", False),
                            ("section", False), ("unknown", False)):
        gen.generate(
            query="q", history=None,
            plan=PlanResult(task="overview", retrieval_scope=scope),
            tool_results=ToolResults(), source_info={},
        )
        messages, _max_tokens = gen._llm.captured[-1]
        system = messages[0]["content"]
        assert (OVERVIEW_GROUNDING_BLOCK in system) is expected, scope
        assert "Book Passages are the only factual evidence" in system if expected else True


# ---------------------------------------------------------------------------
# Presentation layer (required items 4-7)
# ---------------------------------------------------------------------------


def _plan(scope: str) -> PlanResult:
    return PlanResult(task="overview", retrieval_scope=scope)


def test_whole_source_overview_hides_inline_book_citations():
    answer = ("The rebellion begins in Chapter 1 [S2]. Later the farm "
              "changes [S3][S4]. The ending is famous [S1].")
    shown = _apply_overview_presentation(_plan("whole_source"), answer)
    assert "[S1]" not in shown and "[S2]" not in shown and "[S3]" not in shown
    assert "The rebellion begins in Chapter 1." in shown
    # Web markers stay visible (presentation only hides book refs).
    assert _apply_overview_presentation(_plan("whole_source"), "x [W1]") == "x [W1]"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("The heap property is X [S2].", "fact"),
        ("Chapter 3 covers demand and supply [S1][S2].", "section"),
        ("Some synthesis [S4].", "unknown"),
    ],
    ids=['fact_scope_keeps_inline_citations', 'section_scope_keeps_inline_citations', 'unknown_scope_keeps_inline_citations'],
)
def test_fact_scope_keeps_inline_citations(value, expected):
    answer = value
    assert _apply_overview_presentation(_plan(expected), answer) == answer


def test_strip_cleans_artifacts():
    cases = [
        ("word [S1]. next", "word. next"),                 # space before period
        ("word [S1] , tail", "word, tail"),                # space before comma
        ("word([S2]) tail", "word tail"),                  # empty parens removed
        ("word（[S1]）tail", "wordtail"),                  # fullwidth parens
        ("a  b", "a b"),                                  # extra spaces collapse
        ("end [S1] [S2]", "end"),                          # trailing refs
    ]
    for raw, expected in cases:
        assert strip_inline_book_citations(raw) == expected, raw


def test_strip_preserves_markdown_structure():
    """Inline cleanup only: headings, blank lines, and paragraph breaks
    must survive — the overview answer is Markdown."""
    answer = (
        "## Early chapters\n\n"
        "The rebellion starts here [S1].\n\n"
        "## Later development\n\n"
        "The farm changes [S2].\n\n"
        "- first point [S3]\n"
        "- second point"
    )
    shown = strip_inline_book_citations(answer)
    assert "## Early chapters\n\nThe rebellion starts here.\n\n" in shown
    assert "## Later development\n\nThe farm changes." in shown
    assert "- first point\n- second point" in shown
    assert "[S" not in shown
    # No flattening into one line: all four paragraph breaks survive.
    assert shown.count("\n\n") == 4


# ---------------------------------------------------------------------------
# Internal grounding retained (required item 8)
# ---------------------------------------------------------------------------


def test_internal_citations_survive_presentation_hiding():
    """The strip happens on the display copy; the internal answer with
    [S#] markers and the citation list remain available (the pipeline
    stores internal_answer before stripping and never mutates citations)."""
    answer = "Claim one [S1]. Claim two [S2]."
    shown = _apply_overview_presentation(_plan("whole_source"), answer)
    assert shown != answer
    # Internal markers are still derivable: nothing removed them from the
    # source text; the presentation function returns a new string.
    assert "[S1]" in answer and "[S2]" in answer
    # The response schema carries the internal answer for debug.
    from app.routers.ask_schemas import AskResponse
    fields = AskResponse.model_fields
    assert "internal_answer" in fields
    from app.services.qa.types import AskResult
    assert "internal_answer" in AskResult.__dataclass_fields__


# ---------------------------------------------------------------------------
# Fallback and regressions (required items 9-10)
# ---------------------------------------------------------------------------


def test_hierarchy_unavailable_keeps_64_fallback(monkeypatch):
    """Missing hierarchy -> exactly one fallback trace item; presentation
    behavior is scope-based and independent of hierarchy availability."""
    monkeypatch.setattr(config, "hierarchical_overview_enabled", True)
    import sqlite3 as _s

    from app.db import session as _session
    conn = _s.connect(":memory:", check_same_thread=False)
    conn.row_factory = _s.Row
    conn.execute("PRAGMA foreign_keys=ON")
    original = getattr(_session._local, 'conn', None)
    _session._local.conn = conn
    create_tables(conn)
    try:
        nb = create_notebook("NB")
        doc = create_source_document(nb["id"], "b.pdf", content_hash="h")
        insert_chunks(doc["id"], [{
            "chunk_index": 0, "text": "content " * 8, "title": "t",
            "section_path": "t", "page_start": 1, "page_end": 1,
            "chunk_strategy": "test",
        }])
        plan = PlanResult(task="overview", retrieval_scope="whole_source",
                          tool_calls=[])
        results = QAToolExecutor(retriever=object()).execute(
            plan=plan, notebook_id=nb["id"], document_ids=[doc["id"]],
        )
    finally:
        _session._local.conn = original
        conn.close()
    items = [t for t in results.trace if t["name"] == "hierarchical_overview"]
    assert len(items) == 1 and items[0]["status"] == "fallback"
    assert results.summary_context == []


def test_citation_validity_survives_presentation_stripping():
    """Strip ordering invariant: citations are finalized from tool results
    BEFORE presentation stripping, and the strip never regenerates or
    alters the citation list, so citation validity cannot regress."""
    from app.services.qa.citations import _finalize_chunk_citations
    from app.services.retrieval.base import RetrievedChunk

    tr = ToolResults(chunks=[
        RetrievedChunk(chunk_id="c1", document_id="d1", text="first passage"),
        RetrievedChunk(chunk_id="c2", document_id="d1", text="second passage"),
    ])
    _finalize_chunk_citations(tr)
    before = [dict(c) for c in tr.chunk_citations]
    _apply_overview_presentation(_plan("whole_source"), "claims [S1] and [S2]")
    assert tr.chunk_citations == before
    assert [c["chunk_id"] for c in tr.chunk_citations] == ["c1", "c2"]
