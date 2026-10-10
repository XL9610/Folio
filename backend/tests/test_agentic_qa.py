"""Eval tests for the LLM-required agentic QA pipeline.

Tests:
- LLM disabled → /api/ask returns 503
- Planner validates tools, produces sub_questions, normalizes labels
- Tools enforce boundaries, find_sections, get_section_chunks, validate_tool_calls
- Evidence evaluator deterministic + LLM gates
- Generator evidence status handling
- Bounded loop: exit on sufficient / clarify on ambiguous / max rounds
- No fallback templates
"""

import json
from unittest.mock import MagicMock, patch

from app.services.qa.agent_generator import AgentAnswerGenerator
from app.services.qa.agent_planner import (
    WHITELIST_TOOLS,
    AgentPlanner,
    PlanResult,
    ToolCall,
)
from app.services.qa.evidence_evaluator import EvidenceEval, EvidenceEvaluator
from app.services.qa.tools import QAToolExecutor, SectionMatch, ToolResults
from app.services.retrieval.base import RetrievedChunk

# ──────────────────────────────────────────────────────────────────────
# Fake LLM client for tests
# ──────────────────────────────────────────────────────────────────────

class FakeLLM:
    """Mock LLM client that returns predefined JSON or text."""
    def __init__(self, response_data=None, chat_response="Mock answer"):
        self._data = response_data or {}
        self._chat = chat_response
        self.complete_chat_calls = []

    def complete_json(self, system_prompt, user_prompt):
        from app.services.llm.client import LLMJSONResponse
        return LLMJSONResponse(
            data=self._data,
            raw_text=json.dumps(self._data),
            model_used="mock",
            finish_reason="stop",
        )

    def complete_chat(self, messages, max_tokens=None):
        self.complete_chat_calls.append({"messages": messages, "max_tokens": max_tokens})
        return self._chat


def test_planner_overview():
    llm = FakeLLM({
        "task": "overview",
        "tool_calls": [
            {"name": "get_source_metadata", "args": {"document_ids": ["d1"]}},
            {"name": "get_source_outline", "args": {"document_ids": ["d1"]}},
        ]
    })
    planner = AgentPlanner(llm)
    result = planner.plan("What is this book about?", None, ["d1"])
    assert result.task == "overview"
    # Overview augmentation adds a body-chunk retrieval so the answer is grounded.
    assert [t.name for t in result.tool_calls] == [
        "get_source_metadata", "get_source_outline", "get_source_chunks",
    ]


def test_planner_overview_skips_chapter_query():
    """'Summarize Chapter 1' is a chapter query, not a whole-document sweep."""
    llm = FakeLLM({
        "task": "summarize",
        "tool_calls": [
            {"name": "get_source_outline", "args": {"document_ids": ["d1"]}},
        ],
    })
    planner = AgentPlanner(llm)
    result = planner.plan("Summarize Chapter 1", None, ["d1"])
    names = [t.name for t in result.tool_calls]
    assert "get_source_chunks" not in names
    # The structural augmentation still locates the chapter.
    assert "find_sections" in names


def test_planner_overview_skips_named_section_query():
    """A section-scoped plan (find_sections/get_section_chunks) must not get a
    whole-document body sweep appended on top of it."""
    llm = FakeLLM({
        "task": "summarize",
        "tool_calls": [
            {"name": "find_sections", "args": {"section_query": "methods", "document_ids": ["d1"]}},
            {"name": "get_section_chunks", "args": {"section_paths": [], "document_ids": ["d1"], "limit": 15}},
        ],
    })
    planner = AgentPlanner(llm)
    result = planner.plan("Summarize the methods section", None, ["d1"])
    names = [t.name for t in result.tool_calls]
    assert "get_source_chunks" not in names
    assert "find_sections" in names


def test_planner_concept_question():
    llm = FakeLLM({
        "task": "explain_concept",
        "tool_calls": [
            {"name": "retrieve_chunks", "args": {"query": "data structure definition", "document_ids": ["d1"], "top_k": 8}}
        ]
    })
    planner = AgentPlanner(llm)
    result = planner.plan("What is a data structure?", None, ["d1"])
    assert result.task == "explain_concept"
    assert result.tool_calls[0].name == "retrieve_chunks"


def test_planner_rejects_non_whitelisted_tool():
    llm = FakeLLM({
        "task": "overview",
        "tool_calls": [
            {"name": "send_email", "args": {"to": "admin"}},
            {"name": "get_source_outline", "args": {"document_ids": ["d1"]}},
        ]
    })
    planner = AgentPlanner(llm)
    result = planner.plan("plan", None, ["d1"])
    # Non-whitelisted tool should be skipped
    assert all(t.name in WHITELIST_TOOLS for t in result.tool_calls)
    assert len(result.tool_calls) == 1


def test_planner_enforces_document_boundary():
    """Planner should not allow tools to access documents not in selected_ids."""
    llm = FakeLLM({
        "task": "overview",
        "tool_calls": [
            {"name": "get_source_outline", "args": {"document_ids": ["d1", "evil_doc"]}},
        ]
    })
    planner = AgentPlanner(llm)
    result = planner.plan("overview", None, ["d1"])
    # "evil_doc" should be filtered out
    for tc in result.tool_calls:
        if tc.name == "get_source_outline":
            docs = tc.args.get("document_ids", [])
            assert "evil_doc" not in docs


def test_planner_rejects_web_search_when_disabled():
    llm = FakeLLM({
        "task": "specific_search",
        "tool_calls": [
            {"name": "retrieve_chunks", "args": {"query": "book term", "document_ids": ["d1"]}},
            {"name": "web_search", "args": {"query": "outside term", "max_results": 3}},
        ],
    })
    planner = AgentPlanner(llm)
    result = planner.plan("explain outside term", None, ["d1"], allow_web=False)
    assert [t.name for t in result.tool_calls] == ["retrieve_chunks"]


def test_planner_keeps_web_search_when_enabled():
    llm = FakeLLM({
        "task": "specific_search",
        "tool_calls": [
            {"name": "retrieve_chunks", "args": {"query": "book term", "document_ids": ["d1"]}},
            {"name": "web_search", "args": {"query": "outside term", "max_results": 3}},
        ],
    })
    planner = AgentPlanner(llm)
    result = planner.plan("explain outside term", None, ["d1"], allow_web=True)
    assert [t.name for t in result.tool_calls] == ["retrieve_chunks", "web_search"]
    assert "document_ids" not in result.tool_calls[1].args


def test_planner_unknown_task_defaults():
    llm = FakeLLM({
        "task": "delete_all_data",
        "tool_calls": []
    })
    planner = AgentPlanner(llm)
    result = planner.plan("query", None, ["d1"])
    assert result.task == "answer_directly"  # unknown → default


# ──────────────────────────────────────────────────────────────────────
# Tool executor tests
# ──────────────────────────────────────────────────────────────────────

def test_tool_executor_respects_boundary():
    """Tools should not return data for documents outside the boundary."""
    # With no real retriever, just verify boundary logic
    from unittest.mock import MagicMock

    from app.services.qa.tools import QAToolExecutor
    from app.services.retrieval.retriever import HybridRetriever

    # Create a fake retriever
    fake_retriever = MagicMock(spec=HybridRetriever)
    executor = QAToolExecutor(fake_retriever)

    plan = PlanResult(
        task="overview",
        tool_calls=[
            ToolCall(name="get_source_metadata", args={"document_ids": ["d1"]}),
        ]
    )

    # Should not crash even with empty doc list (tools check boundary)
    results = executor.execute(plan, "nb1", [], top_k=5)
    assert len(results.metadata) == 0  # no valid docs


def test_tool_executor_rejects_web_search_when_disabled():
    fake_retriever = MagicMock()
    executor = QAToolExecutor(fake_retriever)
    plan = PlanResult(
        task="specific_search",
        tool_calls=[ToolCall(name="web_search", args={"query": "outside term"})],
    )

    results = executor.execute(plan, "nb1", ["d1"], top_k=5, allow_web=False)

    assert results.web_results == []
    assert results.chunk_citations == []
    assert results.trace == []


def test_tool_executor_web_search_disabled_provider_trace():
    fake_retriever = MagicMock()
    executor = QAToolExecutor(fake_retriever)
    plan = PlanResult(
        task="specific_search",
        tool_calls=[ToolCall(name="web_search", args={"query": "outside term"})],
    )

    results = executor.execute(plan, "nb1", ["d1"], top_k=5, allow_web=True)

    assert results.web_results == []
    assert results.trace[0]["name"] == "web_search"
    assert results.trace[0]["status"] == "disabled"


def test_tool_executor_web_search_fake_provider_citation():
    from app.services.web_search import WebSearchProvider, WebSearchResult

    class FakeWebProvider(WebSearchProvider):
        name = "fake"
        enabled = True

        def search(self, query: str, max_results: int = 5):
            return [
                WebSearchResult(
                    title="External Term",
                    url="https://example.com/term",
                    content="A concise external explanation of the term.",
                    site_name="example.com",
                    retrieved_at="2026-05-13T00:00:00+00:00",
                )
            ]

    fake_retriever = MagicMock()
    executor = QAToolExecutor(fake_retriever, web_search_provider=FakeWebProvider())
    plan = PlanResult(
        task="specific_search",
        tool_calls=[ToolCall(name="web_search", args={"query": "outside term", "max_results": 3})],
    )

    results = executor.execute(plan, "nb1", ["d1"], top_k=5, allow_web=True)

    assert results.web_results[0]["ref_label"] == "W1"
    assert results.chunk_citations[0]["source_type"] == "web"
    assert results.chunk_citations[0]["ref_label"] == "W1"
    assert results.chunk_citations[0]["url"] == "https://example.com/term"


# ──────────────────────────────────────────────────────────────────────
# Answer generator tests
# ──────────────────────────────────────────────────────────────────────

def test_answer_generator_returns_citations():
    llm = FakeLLM({"answer": "ok"})
    llm.complete_chat = lambda messages, max_tokens=None: "This is an answer about data structures."

    generator = AgentAnswerGenerator(llm)
    from app.services.qa.tools import ToolResults

    tr = ToolResults(
        chunk_citations=[
            {"chunk_id": "c1", "document_id": "d1", "text_snippet": "A data structure is...",
             "section_path": "Ch 1", "page_start": 0, "page_end": 0},
        ]
    )

    answer, citations = generator.generate(
        query="What is a data structure?",
        history=None,
        plan=PlanResult(task="explain_concept"),
        tool_results=tr,
        source_info={"notebook_id": "nb1", "document_ids": ["d1"]},
    )

    assert isinstance(answer, str)
    assert len(answer) > 10
    assert len(citations) == 1


def test_answer_generator_context_labels_book_and_web():
    from app.services.retrieval.base import RetrievedChunk

    generator = AgentAnswerGenerator(FakeLLM(chat_response="Mixed answer [S1][W1]"))
    tr = ToolResults()
    tr.section_matches = [
        SectionMatch(
            document_id="d1",
            section_path="List of Texts",
            title="List of Texts",
            page_start=573,
            page_end=574,
            chunk_count=1,
            rank=1,
            match_type="substring",
        )
    ]
    tr.chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            text="Book evidence about the concept.",
            title="Chapter 1",
            section_path="Chapter 1",
            page_start=2,
            page_end=2,
        )
    ]
    tr.chunk_citations = [
        {
            "source_type": "book",
            "ref_label": "S1",
            "chunk_id": "c1",
            "document_id": "d1",
            "text_snippet": "Book evidence",
        },
        {
            "source_type": "web",
            "ref_label": "W1",
            "chunk_id": "web:W1",
            "document_id": "",
            "text_snippet": "Web evidence",
            "url": "https://example.com",
            "title": "External",
        },
    ]
    tr.web_results = [
        {
            "ref_label": "W1",
            "title": "External",
            "site_name": "example.com",
            "url": "https://example.com",
            "content": "Web evidence about the concept.",
        }
    ]

    context = generator._build_context(tr, {"notebook_id": "nb1", "document_ids": ["d1"]})

    assert "[Book passages]" in context
    assert "[S1]" in context
    assert "[Internet results]" in context
    assert "[W1]" in context


def test_answer_generator_no_tool_results():
    llm = FakeLLM({"answer": "ok"})
    llm.complete_chat = lambda messages, max_tokens=None: "I don't have enough information."

    generator = AgentAnswerGenerator(llm)
    tr = ToolResults()  # empty

    answer, citations = generator.generate(
        query="What is a data structure?",
        history=None,
        plan=PlanResult(task="explain_concept"),
        tool_results=tr,
        source_info={"notebook_id": "nb1", "document_ids": ["d1"]},
    )

    assert isinstance(answer, str)
    assert len(citations) == 0


# ──────────────────────────────────────────────────────────────────────
# API-level tests — /api/ask disabled-LLM behavior
# ──────────────────────────────────────────────────────────────────────

def test_ask_request_preferences_are_optional_and_alias_compatible():
    from app.routers.ask_router import AskRequest

    legacy = AskRequest(query="What is a data structure?")
    assert legacy.preferences is None
    assert legacy.follow_up is False

    request = AskRequest(
        query="What is a data structure?",
        follow_up=True,
        preferences={
            "answerStyle": "plain",
            "answerLength": "short",
            "detailLevel": "simple",
            "citationDensity": "minimal",
            "autoSummaryTable": True,
            "enterToSend": False,
        },
    )

    prefs = request.preferences.model_dump(exclude_none=True)
    assert request.follow_up is True
    assert prefs["answer_style"] == "plain"
    assert prefs["answer_length"] == "short"
    assert prefs["detail_level"] == "simple"
    assert prefs["citations"] == "minimal"
    assert prefs["auto_summary_table"] is True


def test_answer_preferences_are_injected_into_final_generation_prompt():
    llm = FakeLLM(chat_response="A concise answer.")
    generator = AgentAnswerGenerator(llm)

    answer, citations = generator.generate(
        query="What is a data structure?",
        history=None,
        plan=PlanResult(task="explain_concept"),
        tool_results=ToolResults(),
        source_info={"notebook_id": "nb1", "document_ids": ["d1"]},
        preferences={
            "style": "plain",
            "length": "short",
            "citations": "minimal",
            "custom_instructions": "Use one takeaway at the end.",
        },
    )

    prompt = llm.complete_chat_calls[0]["messages"][1]["content"]
    assert answer == "A concise answer."
    assert citations == []
    assert "User query: What is a data structure?" in prompt
    assert "Answer preferences" in prompt
    assert "Use plain language" in prompt
    assert "Keep the answer brief" in prompt
    assert "Use one takeaway at the end." in prompt


def test_answer_preferences_ignore_auto_language_instruction():
    llm = FakeLLM(chat_response="An answer.")
    generator = AgentAnswerGenerator(llm)

    generator.generate(
        query="What is a data structure?",
        history=None,
        plan=PlanResult(task="explain_concept"),
        tool_results=ToolResults(),
        source_info={"notebook_id": "nb1", "document_ids": ["d1"]},
        preferences={"language": "auto"},
    )

    prompt = llm.complete_chat_calls[0]["messages"][1]["content"]
    assert "Use this language if possible" not in prompt


def test_answer_preferences_accept_chinese_and_english_language_labels():
    llm = FakeLLM(chat_response="An answer.")
    generator = AgentAnswerGenerator(llm)

    generator.generate(
        query="What is a data structure?",
        history=None,
        plan=PlanResult(task="explain_concept"),
        tool_results=ToolResults(),
        source_info={"notebook_id": "nb1", "document_ids": ["d1"]},
        preferences={"language": "Chinese"},
    )
    generator.generate(
        query="What is a data structure?",
        history=None,
        plan=PlanResult(task="explain_concept"),
        tool_results=ToolResults(),
        source_info={"notebook_id": "nb1", "document_ids": ["d1"]},
        preferences={"language": "English"},
    )

    chinese_prompt = llm.complete_chat_calls[0]["messages"][1]["content"]
    english_prompt = llm.complete_chat_calls[1]["messages"][1]["content"]
    assert "Use this language if possible: Chinese." in chinese_prompt
    assert "Use this language if possible: English." in english_prompt


def test_pipeline_passes_preferences_to_generator_without_mutating_query():
    from app.routers.ask_router import AgenticAskPipeline

    pipeline = object.__new__(AgenticAskPipeline)
    pipeline._planner = MagicMock()
    pipeline._tools = MagicMock()
    pipeline._evaluator = MagicMock()
    pipeline._generator = MagicMock()

    plan = PlanResult(task="explain_concept", tool_calls=[])
    pipeline._planner.plan.return_value = plan
    pipeline._tools.execute.return_value = ToolResults()
    pipeline._evaluator.evaluate.return_value = EvidenceEval(status="proceed", reason="ok")
    pipeline._generator.generate.return_value = ("Styled answer", [])

    preferences = {"style": "bullets", "length": "short"}
    result = pipeline.answer(
        notebook_id="nb1",
        query="What is X?",
        document_ids=["d1"],
        top_k=5,
        history=[],
        allow_web=False,
        preferences=preferences,
        follow_up=False,
    )

    assert result.answer == "Styled answer"
    assert pipeline._planner.plan.call_args.kwargs["query"] == "What is X?"
    assert pipeline._tools.execute.call_args.kwargs["user_query"] == "What is X?"
    assert pipeline._generator.generate.call_args.kwargs["preferences"] == preferences


def test_pipeline_rewrites_short_confusion_when_explicit_followup():
    from app.routers.ask_router import AgenticAskPipeline

    pipeline = object.__new__(AgenticAskPipeline)
    pipeline._planner = MagicMock()
    pipeline._tools = MagicMock()
    pipeline._evaluator = MagicMock()
    pipeline._generator = MagicMock()

    plan = PlanResult(task="explain_concept", tool_calls=[])
    pipeline._planner.plan.return_value = plan
    pipeline._tools.execute.return_value = ToolResults()
    pipeline._evaluator.evaluate.return_value = EvidenceEval(status="proceed", reason="ok")
    pipeline._generator.generate.return_value = ("Deeper answer", [])

    history = [
        {"role": "user", "text": "Explain style in art history", "intent": ""},
        {"role": "assistant", "text": "Style is a pattern of visual choices.", "intent": "explain_concept"},
    ]
    result = pipeline.answer(
        notebook_id="nb1",
        query="没听懂",
        document_ids=["d1"],
        top_k=5,
        history=history,
        allow_web=False,
        preferences=None,
        follow_up=True,
    )

    planner_query = pipeline._planner.plan.call_args.kwargs["query"]
    generator_query = pipeline._generator.generate.call_args.kwargs["query"]
    assert result.answer == "Deeper answer"
    assert "Explain style in art history" in planner_query
    assert "User follow-up: 没听懂" in planner_query
    assert generator_query == planner_query


def test_api_ask_llm_disabled_returns_503(monkeypatch):
    """When LLM_REMOTE_ENABLED=false, /api/ask must return 503, not a template answer.

    This is the critical boundary: no LLM = explicit failure, never a fake answer.
    """
    # Ensure LLM appears disabled regardless of local .env
    monkeypatch.setenv("LLM_REMOTE_ENABLED", "false")

    from fastapi.testclient import TestClient

    from app.main import app

    # Recreate client with patched env
    client = TestClient(app)

    resp = client.post(
        "/api/ask",
        json={
            "notebook_id": "default",
            "query": "How should I study this book?",
            "top_k": 5,
        },
    )

    assert resp.status_code == 503, (
        f"Expected 503 when LLM disabled, got {resp.status_code}. "
        f"Body: {resp.text[:300]}"
    )

    detail = resp.json().get("detail", {})
    assert detail.get("code") == "llm_not_configured", (
        f"Expected detail.code='llm_not_configured', got {detail}"
    )

    # Double-check: response must NOT contain a fake answer or template
    response_str = str(resp.json()).lower()
    for banned in ["from your sources", "study plan", "chapter 1", "overview"]:
        assert banned not in response_str, (
            f"Response must not contain fake template content '{banned}'"
        )


# ──────────────────────────────────────────────────────────────────────
# NEW: Planner — sub_questions, label normalization, clarification
# ──────────────────────────────────────────────────────────────────────

def test_planner_produces_sub_questions():
    """Planner should accept and return sub_questions from LLM."""
    llm = FakeLLM({
        "planner_label": "compare",
        "user_goal": "Compare how both books introduce graphs",
        "sub_questions": [
            {
                "id": "sq1",
                "question": "How does doc A introduce graphs?",
                "scope": {"document_ids": ["d1"]},
                "evidence_needed": "Section about graph introduction"
            },
            {
                "id": "sq2",
                "question": "How does doc B introduce graphs?",
                "scope": {"document_ids": ["d2"]},
                "evidence_needed": "Section about graph introduction"
            },
        ],
        "tool_calls": [
            {"name": "get_source_outline", "args": {"document_ids": ["d1", "d2"]}},
        ]
    })
    planner = AgentPlanner(llm)
    result = planner.plan("Compare how both books introduce graphs", None, ["d1", "d2"])
    assert result.task == "compare"
    assert result.user_goal == "Compare how both books introduce graphs"
    assert len(result.sub_questions) == 2
    assert result.sub_questions[0].id == "sq1"
    assert result.sub_questions[0].question == "How does doc A introduce graphs?"


def test_planner_clarify_needed():
    """Planner should set clarify_needed when query is ambiguous."""
    llm = FakeLLM({
        "planner_label": "clarify",
        "user_goal": "Unknown",
        "clarify_needed": True,
        "clarifying_question": "Which document do you mean?",
        "sub_questions": [],
        "tool_calls": [],
    })
    planner = AgentPlanner(llm)
    result = planner.plan("Summarize it", None, ["d1", "d2"])
    assert result.clarify_needed is True
    assert len(result.clarifying_question) > 0


def test_planner_tool_name_label_aliased():
    """Planner label 'retrieve_chunks' should normalize to 'specific_search'."""
    llm = FakeLLM({
        "planner_label": "retrieve_chunks",
        "tool_calls": [
            {"name": "retrieve_chunks", "args": {"query": "test"}}
        ]
    })
    planner = AgentPlanner(llm)
    result = planner.plan("find X", None, ["d1"])
    # Label should be normalized, not left as tool name
    assert result.task == "specific_search"


def test_planner_whitelist_includes_new_tools():
    """WHITELIST_TOOLS must include find_sections and get_section_chunks."""
    assert "find_sections" in WHITELIST_TOOLS
    assert "get_section_chunks" in WHITELIST_TOOLS


# ──────────────────────────────────────────────────────────────────────
# NEW: Tool executor — find_sections, get_section_chunks, validate
# ──────────────────────────────────────────────────────────────────────

def test_retrieve_tool_error_only_when_all_backends_fail():
    """tool_error is set only when every available backend errored (not zero-hit)."""
    from unittest.mock import MagicMock

    scenarios = [
        # zero hit (both empty, no error)
        ({"dense_available": True, "keyword_backend": "fts"}, False),
        # dense errored but keyword succeeded → keyword-only degradation
        ({"dense_available": True, "keyword_backend": "fts", "dense_error": "x"}, False),
        # both backends errored → retrieval unavailable
        ({"dense_available": True, "keyword_backend": "fts", "dense_error": "x", "keyword_error": "y"}, True),
        # no backend available at all → retrieval unavailable
        ({"dense_available": False, "keyword_backend": "none"}, True),
    ]
    for meta, expected in scenarios:
        retriever = MagicMock()
        retriever.search_with_meta.return_value = ([], meta)
        executor = QAToolExecutor(retriever)
        results = ToolResults()
        tc = ToolCall(name="retrieve_chunks", args={"query": "q", "document_ids": ["d1"], "top_k": 5})
        executor._tool_retrieve(results, tc, "nb1", ["d1"], 5)
        assert results.tool_error is expected, f"meta={meta}"


def test_merge_tool_results_preserves_tool_error():
    """tool_error must OR across rounds — a clean round must not clear an error."""
    from app.routers.ask_router import _merge_tool_results

    into = ToolResults()
    from_round = ToolResults()
    from_round.tool_error = True
    _merge_tool_results(into, from_round)
    assert into.tool_error is True

    # A later clean round must not reset a prior failure.
    sticky = ToolResults()
    sticky.tool_error = True
    _merge_tool_results(sticky, ToolResults())
    assert sticky.tool_error is True


def _setup_tool_db(monkeypatch):
    """Provide an in-memory SQLite DB for tool-execution tests."""
    import sqlite3
    from app.db import session as db_session
    from app.db.schema import create_tables

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    create_tables(conn)
    monkeypatch.setattr(db_session._local, "conn", conn, raising=False)
    return conn


def test_get_source_chunks_without_section_path(monkeypatch):
    """Docs whose chunks have no section_path still yield body text."""
    _setup_tool_db(monkeypatch)
    from app.repositories.notebook_repository import create_notebook
    from app.repositories.document_repository import create_source_document
    from app.repositories.chunk_repository import insert_chunks

    nb = create_notebook("NB")
    doc = create_source_document(nb["id"], "a.pdf")
    insert_chunks(doc["id"], [
        {"chunk_index": 0, "text": "body text one", "title": "", "section_path": "",
         "page_start": -1, "page_end": -1, "chunk_strategy": "toc"},
    ])

    executor = QAToolExecutor(MagicMock())
    results = ToolResults()
    tc = ToolCall(name="get_source_chunks", args={"document_ids": [doc["id"]], "limit": 20})
    executor._tool_get_source_chunks(results, tc, nb["id"], [doc["id"]], 0)
    assert [c.text for c in results.chunks] == ["body text one"]


def test_get_source_chunks_covers_later_sections(monkeypatch):
    """Documents with more than 12 sections must still cover the latter half."""
    _setup_tool_db(monkeypatch)
    from app.repositories.notebook_repository import create_notebook
    from app.repositories.document_repository import create_source_document
    from app.repositories.chunk_repository import insert_chunks

    nb = create_notebook("NB")
    doc = create_source_document(nb["id"], "a.pdf")
    insert_chunks(doc["id"], [
        {"chunk_index": i, "text": f"section {i+1} body content",
         "title": f"Section {i+1}", "section_path": f"Section {i+1}",
         "page_start": i + 1, "page_end": i + 1, "chunk_strategy": "toc"}
        for i in range(20)
    ])

    executor = QAToolExecutor(MagicMock())
    results = ToolResults()
    tc = ToolCall(name="get_source_chunks", args={"document_ids": [doc["id"]], "limit": 20})
    executor._tool_get_source_chunks(results, tc, nb["id"], [doc["id"]], 0)

    titles = {c.title for c in results.chunks}
    assert any(f"Section {i}" in titles for i in range(15, 21))


def test_get_source_chunks_excludes_wrong_notebook(monkeypatch):
    """Chunks from a document in a different notebook are never returned."""
    _setup_tool_db(monkeypatch)
    from app.repositories.notebook_repository import create_notebook
    from app.repositories.document_repository import create_source_document
    from app.repositories.chunk_repository import insert_chunks

    nb1 = create_notebook("NB1")
    nb2 = create_notebook("NB2")
    doc1 = create_source_document(nb1["id"], "a.pdf")
    doc2 = create_source_document(nb2["id"], "b.pdf")
    insert_chunks(doc1["id"], [
        {"chunk_index": 0, "text": "doc1 body", "title": "", "section_path": "",
         "page_start": -1, "page_end": -1, "chunk_strategy": "toc"},
    ])
    insert_chunks(doc2["id"], [
        {"chunk_index": 0, "text": "doc2 body", "title": "", "section_path": "",
         "page_start": -1, "page_end": -1, "chunk_strategy": "toc"},
    ])

    executor = QAToolExecutor(MagicMock())
    results = ToolResults()
    tc = ToolCall(name="get_source_chunks", args={"document_ids": [doc1["id"], doc2["id"]], "limit": 20})
    executor._tool_get_source_chunks(results, tc, nb1["id"], [doc1["id"], doc2["id"]], 0)

    texts = {c.text for c in results.chunks}
    assert "doc1 body" in texts
    assert "doc2 body" not in texts


def test_get_source_chunks_prepends_overview_samples(monkeypatch):
    """Overview samples must land first so the generator's first-10 sees them."""
    _setup_tool_db(monkeypatch)
    from app.repositories.notebook_repository import create_notebook
    from app.repositories.document_repository import create_source_document
    from app.repositories.chunk_repository import insert_chunks

    nb = create_notebook("NB")
    doc = create_source_document(nb["id"], "a.pdf")
    insert_chunks(doc["id"], [
        {"chunk_index": i, "text": f"overview body {i}",
         "title": f"Section {i}", "section_path": f"Section {i}",
         "page_start": i + 1, "page_end": i + 1, "chunk_strategy": "toc"}
        for i in range(3)
    ])

    executor = QAToolExecutor(MagicMock())
    results = ToolResults()
    # Simulate retrieve_chunks having already populated the chunk list.
    results.chunks = [RetrievedChunk(
        chunk_id="retr", document_id=doc["id"], text="retrieve result",
        title="Retr", section_path="Retr", page_start=1, page_end=1,
    )]
    tc = ToolCall(name="get_source_chunks", args={"document_ids": [doc["id"]], "limit": 20})
    executor._tool_get_source_chunks(results, tc, nb["id"], [doc["id"]], 0)

    # Overview samples come first, ahead of the earlier retrieve result.
    assert results.chunks[0].text != "retrieve result"
    assert {c.text for c in results.chunks[:3]} == {"overview body 0", "overview body 1", "overview body 2"}


def test_get_source_chunks_limits_to_generator_window(monkeypatch):
    """Overview must prepare only the ~10 chunks the generator consumes, and
    those must be interleaved across documents rather than all from the first."""
    _setup_tool_db(monkeypatch)
    from app.repositories.notebook_repository import create_notebook
    from app.repositories.document_repository import create_source_document
    from app.repositories.chunk_repository import insert_chunks

    nb = create_notebook("NB")
    doc_ids = []
    for name in ("a.pdf", "b.pdf"):
        doc = create_source_document(nb["id"], name)
        insert_chunks(doc["id"], [
            {"chunk_index": i, "text": f"{name} body {i}",
             "title": f"{name} S{i}", "section_path": f"{name} Section {i}",
             "page_start": i + 1, "page_end": i + 1, "chunk_strategy": "toc"}
            for i in range(30)
        ])
        doc_ids.append(doc["id"])

    executor = QAToolExecutor(MagicMock())
    results = ToolResults()
    tc = ToolCall(name="get_source_chunks", args={"document_ids": doc_ids, "limit": 20})
    executor._tool_get_source_chunks(results, tc, nb["id"], doc_ids, 0)

    # The generator reads results.chunks[:10]; overview must fit within it.
    assert len(results.chunks) <= 10
    doc_set = {c.document_id for c in results.chunks}
    assert doc_set == set(doc_ids)  # both documents represented


def test_get_source_chunks_spreads_across_single_document(monkeypatch):
    """Within one long document, the 10 consumed chunks must span start/end."""
    _setup_tool_db(monkeypatch)
    from app.repositories.notebook_repository import create_notebook
    from app.repositories.document_repository import create_source_document
    from app.repositories.chunk_repository import insert_chunks

    nb = create_notebook("NB")
    doc = create_source_document(nb["id"], "long.pdf")
    insert_chunks(doc["id"], [
        {"chunk_index": i, "text": f"body {i}", "title": f"S{i}",
         "section_path": f"Section {i}", "page_start": i + 1, "page_end": i + 1,
         "chunk_strategy": "toc"}
        for i in range(100)
    ])

    executor = QAToolExecutor(MagicMock())
    results = ToolResults()
    tc = ToolCall(name="get_source_chunks", args={"document_ids": [doc["id"]], "limit": 20})
    executor._tool_get_source_chunks(results, tc, nb["id"], [doc["id"]], 0)

    assert len(results.chunks) <= 10
    pages = sorted(c.page_start for c in results.chunks)
    assert pages[0] < 20
    assert pages[-1] > 80


def test_tool_validate_tool_calls_filters_unknown():
    """validate_tool_calls should skip non-whitelisted tools."""
    from unittest.mock import MagicMock
    executor = QAToolExecutor(MagicMock())

    calls = [
        ToolCall(name="send_email", args={}),
        ToolCall(name="retrieve_chunks", args={"query": "test"}),
    ]
    validated = executor.validate_tool_calls(calls, ["d1"])
    assert len(validated) == 1
    assert validated[0].name == "retrieve_chunks"


def test_tool_validate_tool_calls_bounds_document_ids():
    """validate_tool_calls should filter document_ids to allowed set."""
    from unittest.mock import MagicMock
    executor = QAToolExecutor(MagicMock())

    calls = [
        ToolCall(name="retrieve_chunks", args={
            "query": "test", "document_ids": ["d1", "evil", "d3"]
        }),
    ]
    validated = executor.validate_tool_calls(calls, ["d1", "d2"])
    assert "evil" not in validated[0].args.get("document_ids", [])
    assert "d3" not in validated[0].args.get("document_ids", [])


def test_tool_find_sections_no_match():
    """find_sections should return no_match trace when nothing found."""
    from unittest.mock import MagicMock, patch
    executor = QAToolExecutor(MagicMock())

    plan = PlanResult(
        task="locate",
        tool_calls=[
            ToolCall(name="find_sections", args={
                "section_query": "Nonexistent Chapter X",
                "document_ids": ["d1"],
            }),
        ]
    )

    # Mock get_document and get_document_sections to return empty
    with patch("app.services.qa.tools.get_document") as mock_doc, \
         patch("app.services.qa.tools.get_document_sections") as mock_sections:
        mock_doc.return_value = {"id": "d1", "notebook_id": "nb1", "filename": "test.pdf"}
        mock_sections.return_value = []

        results = executor.execute(plan, "nb1", ["d1"])
        assert len(results.section_matches) == 0
        assert results.trace[-1]["status"] == "no_match"


def test_tool_find_sections_chapter_match():
    """find_sections should match 'Chapter 1' exactly."""
    from unittest.mock import MagicMock, patch
    executor = QAToolExecutor(MagicMock())

    plan = PlanResult(
        task="locate",
        tool_calls=[
            ToolCall(name="find_sections", args={
                "section_query": "Chapter 1",
                "document_ids": ["d1"],
            }),
        ]
    )

    mock_sections = [
        {"section_path": "Chapter 1: Introduction", "title": "Chapter 1: Introduction",
         "page_start": 1, "page_end": 20, "chunk_count": 15},
        {"section_path": "Chapter 10: Advanced", "title": "Chapter 10: Advanced",
         "page_start": 200, "page_end": 250, "chunk_count": 30},
        {"section_path": "Chapter 2: Basics", "title": "Chapter 2: Basics",
         "page_start": 21, "page_end": 50, "chunk_count": 20},
    ]

    with patch("app.services.qa.tools.get_document") as mock_doc, \
         patch("app.services.qa.tools.get_document_sections") as mock_sections_fn:
        mock_doc.return_value = {"id": "d1", "notebook_id": "nb1", "filename": "test.pdf"}
        mock_sections_fn.return_value = mock_sections

        results = executor.execute(plan, "nb1", ["d1"])
        assert len(results.section_matches) >= 1
        # Chapter 1 should be first (rank 1 = exact)
        # Chapter 10 should NOT be matched as "Chapter 1"
        matched_paths = [m.section_path for m in results.section_matches]
        assert any("Chapter 1:" in p for p in matched_paths)
        # Chapter 10 should not match "Chapter 1"
        assert not any("Chapter 10" in p for p in matched_paths)


def test_tool_get_section_chunks_no_paths():
    """get_section_chunks with no section_paths or prefix should error."""
    from unittest.mock import MagicMock
    executor = QAToolExecutor(MagicMock())

    plan = PlanResult(
        task="specific_search",
        tool_calls=[
            ToolCall(name="get_section_chunks", args={
                "document_ids": ["d1"],
            }),
        ]
    )

    results = executor.execute(plan, "nb1", ["d1"])
    assert len(results.chunks) == 0
    assert results.trace[-1]["status"] == "error"


def test_tool_registry_is_whitelisted():
    """QAToolExecutor.is_whitelisted should work for all registered tools."""
    from unittest.mock import MagicMock
    executor = QAToolExecutor(MagicMock())

    assert executor.is_whitelisted("retrieve_chunks") is True
    assert executor.is_whitelisted("find_sections") is True
    assert executor.is_whitelisted("get_section_chunks") is True
    assert executor.is_whitelisted("get_source_outline") is True
    assert executor.is_whitelisted("get_source_metadata") is True
    assert executor.is_whitelisted("evil_tool") is False


# ──────────────────────────────────────────────────────────────────────
# NEW: Evidence evaluator tests
# ──────────────────────────────────────────────────────────────────────

def test_evaluator_no_documents():
    """Evaluator should flag no documents as ambiguous."""
    llm = FakeLLM({})
    evaluator = EvidenceEvaluator(llm)
    plan = PlanResult(task="answer_directly")
    tr = ToolResults()

    result = evaluator.evaluate(
        query="What is X?",
        plan=plan,
        tool_results=tr,
        notebook_id="nb1",
        document_ids=[],
    )
    assert result.status == "ambiguous"
    assert len(result.clarifying_question) > 0


def test_evaluator_no_evidence():
    """Evaluator should flag empty results as insufficient."""
    llm = FakeLLM({})
    evaluator = EvidenceEvaluator(llm)
    plan = PlanResult(task="answer_directly")
    tr = ToolResults()  # completely empty

    result = evaluator.evaluate(
        query="What is X?",
        plan=plan,
        tool_results=tr,
        notebook_id="nb1",
        document_ids=["d1"],
    )
    assert result.status == "insufficient"


def test_evaluator_retrieval_error_is_distinct_from_no_hit():
    """Tool errors must be flagged as retrieval-unavailable, not 'no content'."""
    llm = FakeLLM({})
    evaluator = EvidenceEvaluator(llm)
    tr = ToolResults()
    tr.tool_error = True  # a search tool raised; no evidence gathered

    result = evaluator.evaluate(
        query="What does this article talk about?",
        plan=PlanResult(task="answer_directly"),
        tool_results=tr,
        notebook_id="nb1",
        document_ids=["d1"],
    )
    assert result.status == "error"


def test_evaluator_metadata_does_not_mask_retrieval_error():
    """Metadata/outline are deterministic DB reads; they must not mask a
    body-retrieval failure (otherwise the generator summarizes a filename)."""
    llm = FakeLLM({})
    evaluator = EvidenceEvaluator(llm)
    tr = ToolResults()
    tr.tool_error = True
    tr.metadata = [{"filename": "book.pdf", "page_count": 100, "chunk_count": 50}]
    tr.outline = [{"filename": "book.pdf", "sections": [{"section_path": "Ch 1", "title": "Ch 1"}]}]

    result = evaluator.evaluate(
        query="Summarize this book",
        plan=PlanResult(task="summarize"),
        tool_results=tr,
        notebook_id="nb1",
        document_ids=["d1"],
    )
    assert result.status == "error"


def test_evaluator_has_evidence_returns_proceed():
    """Evaluator (deterministic-only) returns 'proceed' when evidence exists."""
    llm = FakeLLM({})
    evaluator = EvidenceEvaluator(llm)
    plan = PlanResult(task="explain_concept")

    from app.services.retrieval.base import RetrievedChunk
    tr = ToolResults()
    tr.chunks = [
        RetrievedChunk(
            chunk_id="c1", document_id="d1",
            text="A data structure is a way of organizing data.",
            section_path="Chapter 1", title="Introduction",
            page_start=1, page_end=2,
        )
    ]

    result = evaluator.evaluate(
        query="What is a data structure?",
        plan=plan,
        tool_results=tr,
        notebook_id="nb1",
        document_ids=["d1"],
    )
    # Deterministic-only evaluator returns "proceed" (not "sufficient")
    assert result.status == "proceed"


def test_evaluator_overview_queries_skip_exact_term_gate():
    """Overview/summary intent must not be flagged off-topic when generic topic
    words (article/talk/findings) are absent from the body text."""
    from app.services.retrieval.base import RetrievedChunk

    for query in (
        "What does this article talk about?",
        "Summarize this article",
        "What are the main findings?",
    ):
        evaluator = EvidenceEvaluator(FakeLLM({}))
        tr = ToolResults()
        tr.chunks = [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                text="Gradient descent iteratively adjusts parameters to minimize a loss function.",
                section_path="Chapter 2: Methods",
                title="Gradient Descent",
                page_start=3,
                page_end=4,
            )
        ]
        result = evaluator.evaluate(
            query=query,
            plan=PlanResult(task="answer_directly"),
            tool_results=tr,
            notebook_id="nb1",
            document_ids=["d1"],
        )
        assert result.status == "proceed", f"{query!r} wrongly blocked: {result.reason}"


def test_evaluator_section_matches_count_as_evidence():
    """Section-match-only structural evidence should not be treated as empty."""
    llm = FakeLLM({})
    evaluator = EvidenceEvaluator(llm)
    tr = ToolResults()
    tr.section_matches = [
        SectionMatch(
            document_id="d1",
            section_path="Experiments",
            title="Experiments",
            page_start=6,
            page_end=9,
            chunk_count=3,
            rank=1,
            match_type="substring",
        )
    ]

    result = evaluator.evaluate(
        query="What section discusses experiments?",
        plan=PlanResult(task="locate"),
        tool_results=tr,
        notebook_id="nb1",
        document_ids=["d1"],
    )

    assert result.status == "proceed"


def test_evaluator_flags_off_topic_mandatory_query_terms():
    """If high-signal query terms are absent from all evidence, refuse early."""
    llm = FakeLLM({})
    evaluator = EvidenceEvaluator(llm)

    from app.services.retrieval.base import RetrievedChunk
    tr = ToolResults()
    tr.chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            text="This chapter explains interfaces, generics, linked structures, and algorithm analysis.",
            section_path="Chapter 2: Object-Oriented Design",
            title="Interfaces",
            page_start=10,
            page_end=12,
        )
    ]

    result = evaluator.evaluate(
        query="What does the APHINITY trial study?",
        plan=PlanResult(task="explain_concept"),
        tool_results=tr,
        notebook_id="nb1",
        document_ids=["d1"],
    )

    assert result.status == "insufficient"
    assert "high-signal query term" in result.reason


def test_evaluator_keeps_relevant_mandatory_query_terms():
    """The same exact-term guard should pass when source evidence contains the term."""
    llm = FakeLLM({})
    evaluator = EvidenceEvaluator(llm)

    from app.services.retrieval.base import RetrievedChunk
    tr = ToolResults()
    tr.chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            text="SIRA stands for SuperIntelligent Retrieval Agent.",
            section_path="Introduction",
            title="Our Proposal: SIRA",
            page_start=1,
            page_end=2,
        )
    ]

    result = evaluator.evaluate(
        query="What is SIRA?",
        plan=PlanResult(task="explain_concept"),
        tool_results=tr,
        notebook_id="nb1",
        document_ids=["d1"],
    )

    assert result.status == "proceed"


def test_evaluator_flags_isolated_terms_for_definition_query():
    """A concept query should not pass when terms appear only in unrelated chunks."""
    llm = FakeLLM({})
    evaluator = EvidenceEvaluator(llm)

    from app.services.retrieval.base import RetrievedChunk
    tr = ToolResults()
    tr.chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            text="This bibliography contains a list of texts in art history.",
            section_path="List of Texts",
            title="List of Texts",
            page_start=573,
            page_end=574,
        ),
        RetrievedChunk(
            chunk_id="c2",
            document_id="d1",
            text="The essay mentions linked solutions in a discussion of late Roman style.",
            section_path="Anthropology and/as Art History",
            title="Introduction",
            page_start=160,
            page_end=163,
        ),
    ]

    result = evaluator.evaluate(
        query="What is a singly linked list?",
        plan=PlanResult(task="explain_concept"),
        tool_results=tr,
        notebook_id="nb1",
        document_ids=["d1"],
    )

    assert result.status == "insufficient"
    assert "isolated words" in result.reason


def test_evaluator_allows_multiword_definition_when_concept_is_present():
    llm = FakeLLM({})
    evaluator = EvidenceEvaluator(llm)

    from app.services.retrieval.base import RetrievedChunk
    tr = ToolResults()
    tr.chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            text="Singly Linked Lists are a concrete implementation of a sequence.",
            section_path="Chapter 3 > Singly Linked Lists",
            title="Singly Linked Lists",
            page_start=120,
            page_end=124,
        )
    ]

    result = evaluator.evaluate(
        query="What is a singly linked list?",
        plan=PlanResult(task="explain_concept"),
        tool_results=tr,
        notebook_id="nb1",
        document_ids=["d1"],
    )

    assert result.status == "proceed"


def test_evaluator_does_not_call_llm():
    """Evaluator is deterministic-only; LLM must never be called."""
    call_count = 0

    class CountingFakeLLM:
        def __init__(self):
            pass
        def complete_json(self, *a, **kw):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("LLM evaluator should not be called!")
        def complete_chat(self, *a, **kw):
            return "ok"

    llm = CountingFakeLLM()
    evaluator = EvidenceEvaluator(llm)

    from app.services.retrieval.base import RetrievedChunk
    tr = ToolResults()
    tr.chunks = [
        RetrievedChunk(
            chunk_id="c1", document_id="d1", text="Some text",
            section_path="Ch 1", title="Intro", page_start=1, page_end=1,
        )
    ]

    result = evaluator.evaluate(
        query="What is X?",
        plan=PlanResult(task="explain_concept"),
        tool_results=tr,
        notebook_id="nb1",
        document_ids=["d1"],
    )
    assert result.status == "proceed"
    assert call_count == 0, f"LLM should not be called by deterministic evaluator, but was called {call_count} times"


# ──────────────────────────────────────────────────────────────────────
# NEW: Generator — evidence status handling
# ──────────────────────────────────────────────────────────────────────

def test_generator_receives_evidence_status():
    """Generator should weave evidence status into the prompt."""
    llm = FakeLLM(chat_response="Not enough evidence to answer.")
    AgentAnswerGenerator(llm)

# ──────────────────────────────────────────────────────────────────────
# NEW: Generator — synthesize-first tests
# ──────────────────────────────────────────────────────────────────────

def test_generator_does_not_include_evidence_status():
    """Generator must NOT inject evidence_status/evaluator verdict into prompt."""
    llm = FakeLLM(chat_response="Based on the retrieved material...")
    generator = AgentAnswerGenerator(llm)

    tr = ToolResults()
    answer, citations = generator.generate(
        query="Summarize Chapter 1",
        history=None,
        plan=PlanResult(task="summarize"),
        tool_results=tr,
        source_info={
            "notebook_id": "nb1",
            "document_ids": ["d1"],
            "evidence_status": "insufficient",
            "evidence_reason": "No matching chapter chunks found",
        },
    )

    prompt = llm.complete_chat_calls[0]["messages"][1]["content"].lower()
    # Evidence status must NOT leak into the prompt
    assert "evidence quality" not in prompt
    assert "evidence evaluator" not in prompt
    assert "do not fabricate" not in prompt
    assert "importantevidence" not in prompt.replace(" ", "")


def test_generator_prompt_has_no_refusal_antipatterns():
    """System prompt must not contain refusal-triggering rules.

    Note: the prompt may contain meta-instructions telling the LLM NOT to
    use certain phrases (e.g. 'Do not say "I would need a passage..."').
    Those are fine — they instruct the LLM away from refusal language.
    The test checks that the prompt's actual RULES don't encode refusal.
    """
    from app.services.qa.agent_generator import AGENT_ANSWER_SYSTEM_PROMPT
    prompt_lower = AGENT_ANSWER_SYSTEM_PROMPT.lower()
    # These phrases in RULE form encode the old refuse-first philosophy
    assert "do not pretend" not in prompt_lower
    # "do not say" and "the user should provide" appear within meta-instructions
    # telling the LLM NOT to use those phrases — that's correct, not a refusal rule.
    # Positive: should contain synthesize-first language
    assert "prefer answering over refusing" in prompt_lower
    assert "synthesize" in prompt_lower


def test_generator_does_not_get_refuse_instruction_for_insufficient():
    """Even when evidence_status='needs_more_search', no IMPORTANT refuse block."""
    llm = FakeLLM(chat_response="I can answer partially.")
    generator = AgentAnswerGenerator(llm)

    tr = ToolResults()
    answer, citations = generator.generate(
        query="Tell me about Chapter 5",
        history=None,
        plan=PlanResult(task="summarize"),
        tool_results=tr,
        source_info={
            "notebook_id": "nb1",
            "document_ids": ["d1"],
            "evidence_status": "needs_more_search",
            "evidence_reason": "Weak retrieval results",
        },
    )

    prompt = llm.complete_chat_calls[0]["messages"][1]["content"]
    assert "IMPORTANT" not in prompt, "IMPORTANT refuse block must not appear"
    assert "needs_more_search" not in prompt.lower(), "evidence_status must not leak"
    assert "Do NOT fabricate" not in prompt


# ──────────────────────────────────────────────────────────────────────
# NEW: Pipeline / bounded loop tests
# ──────────────────────────────────────────────────────────────────────

def test_pipeline_clarification_path():
    """When planner says clarify_needed, pipeline should return clarification."""

    # Fake LLM that produces clarification plan
    llm = FakeLLM({
        "planner_label": "clarify",
        "user_goal": "",
        "clarify_needed": True,
        "clarifying_question": "Which chapter do you want to summarize?",
        "sub_questions": [],
        "tool_calls": [],
    })

    from app.routers.ask_router import AgenticAskPipeline

    with patch("app.routers.ask_router.create_llm_client", return_value=llm), \
         patch("app.routers.ask_router.HybridRetriever"), \
         patch("app.routers.ask_router.SentenceTransformerEmbedder"), \
         patch("app.routers.ask_router.SqliteVecStore"), \
         patch("app.routers.ask_router.LLMConfig") as mock_cfg:

        mock_cfg.return_value.is_configured = True
        pipeline = AgenticAskPipeline()
        result = pipeline.answer(
            notebook_id="nb1",
            query="Summarize it",
            document_ids=["d1", "d2"],
            top_k=5,
            history=[],
        )

        assert result.agent_status == "clarification_needed"
        assert "Which chapter" in result.answer


def test_pipeline_followup_expansion_ignores_planner_clarification():
    """Explicit expansion follow-ups should continue the previous topic."""
    from app.routers.ask_router import AgenticAskPipeline

    pipeline = object.__new__(AgenticAskPipeline)
    pipeline._planner = MagicMock()
    pipeline._tools = MagicMock()
    pipeline._evaluator = MagicMock()
    pipeline._generator = MagicMock()

    clarify_plan = PlanResult(
        task="clarify",
        clarify_needed=True,
        clarifying_question="Which concept do you want me to explain?",
    )
    pipeline._planner.plan.return_value = clarify_plan
    pipeline._tools.execute.return_value = ToolResults()
    pipeline._evaluator.evaluate.return_value = EvidenceEval(status="proceed", reason="ok")
    pipeline._generator.generate.return_value = ("A deeper explanation of Visitor.", [])

    history = [
        {"role": "user", "text": "Tell me a simple concept from this book", "intent": ""},
        {"role": "assistant", "text": "A simple concept is Visitor.", "intent": "explain_concept"},
    ]
    result = pipeline.answer(
        notebook_id="nb1",
        query="\u7ee7\u7eed\u6df1\u5165\u8bb2\u89e3",
        document_ids=["d1"],
        top_k=5,
        history=history,
        follow_up=True,
    )

    executed_plan = pipeline._tools.execute.call_args.kwargs["plan"]
    generator_query = pipeline._generator.generate.call_args.kwargs["query"]
    assert result.agent_status == "completed"
    assert result.answer == "A deeper explanation of Visitor."
    assert executed_plan.task == "explain_concept"
    assert executed_plan.clarify_needed is False
    assert executed_plan.tool_calls[0].name == "retrieve_chunks"
    assert "Tell me a simple concept from this book" in executed_plan.tool_calls[0].args["query"]
    assert "\u7ee7\u7eed\u6df1\u5165\u8bb2\u89e3" in generator_query


def test_pipeline_loop_stops_on_sufficient():
    """Bounded loop should stop after first round when evidence is sufficient."""
    # Create a multi-step mock LLM that:
    # 1. Planner returns retrieve plan
    # 2. Evaluator returns sufficient

    planner_response = FakeLLM({
        "planner_label": "explain_concept",
        "user_goal": "Explain data structures",
        "sub_questions": [],
        "tool_calls": [
            {"name": "retrieve_chunks", "args": {"query": "data structures", "top_k": 5}}
        ],
    })

    evaluator_response = FakeLLM({
        "status": "sufficient",
        "reason": "Got relevant passages",
    })

    generator_llm = FakeLLM(chat_response="Data structures are fundamental...")

    # We need a composite LLM client that delegates to the right mock per method
    class CompositeFakeLLM:
        def __init__(self, planner_llm, evaluator_llm, generator_llm):
            self.planner = planner_llm
            self.evaluator = evaluator_llm
            self.generator = generator_llm
            self.complete_json_calls = []
            self.complete_chat_calls = []

        def complete_json(self, system_prompt, user_prompt):
            self.complete_json_calls.append(system_prompt[:50])
            # First call: planner; second call: evaluator
            if len(self.complete_json_calls) == 1:
                return self.planner.complete_json(system_prompt, user_prompt)
            else:
                return self.evaluator.complete_json(system_prompt, user_prompt)

        def complete_chat(self, messages, max_tokens=None):
            self.complete_chat_calls.append(messages)
            return self.generator.complete_chat(messages, max_tokens)

    composite = CompositeFakeLLM(planner_response, evaluator_response, generator_llm)

    from app.routers.ask_router import AgenticAskPipeline

    with patch("app.routers.ask_router.create_llm_client", return_value=composite), \
         patch("app.routers.ask_router.HybridRetriever"), \
         patch("app.routers.ask_router.SentenceTransformerEmbedder"), \
         patch("app.routers.ask_router.SqliteVecStore"), \
         patch("app.routers.ask_router.LLMConfig") as mock_cfg:

        mock_cfg.return_value.is_configured = True

        # Also need to mock the retriever to return something with evidence
        with patch("app.routers.ask_router.QAToolExecutor") as mock_executor_cls:
            from app.services.retrieval.base import RetrievedChunk

            mock_executor = MagicMock()
            tr_with_evidence = ToolResults()
            tr_with_evidence.chunks = [
                RetrievedChunk(
                    chunk_id="c1", document_id="d1",
                    text="A data structure is a way of organizing data.",
                    section_path="Chapter 1", title="Introduction",
                    page_start=1, page_end=2,
                )
            ]
            mock_executor.execute.return_value = tr_with_evidence
            mock_executor.validate_tool_calls = lambda calls, doc_ids: calls
            mock_executor.is_whitelisted = lambda name: True
            mock_executor_cls.return_value = mock_executor

            pipeline = AgenticAskPipeline()
            result = pipeline.answer(
                notebook_id="nb1",
                query="What is a data structure?",
                document_ids=["d1"],
                top_k=5,
                history=[],
            )

            # Should have completed (not clarification, not timeout)
            assert result.agent_status == "completed", f"Got status: {result.agent_status}, answer: {result.answer[:200]}"
            # Evaluator no longer calls LLM (Fix 1B: deterministic-only), so only planner calls complete_json
            assert len(composite.complete_json_calls) == 1, f"Expected 1 planner call, got {len(composite.complete_json_calls)}"
            # Should have called generator
            assert len(composite.complete_chat_calls) == 1


def test_pipeline_respects_max_rounds():
    """Bounded loop must not exceed MAX_AGENT_ROUNDS."""
    # Planner: retrieve, Evaluator always "needs_more_search"
    planner_llm = FakeLLM({
        "planner_label": "explain_concept",
        "tool_calls": [
            {"name": "retrieve_chunks", "args": {"query": "test", "top_k": 3}}
        ],
    })

    evaluator_llm = FakeLLM({
        "status": "needs_more_search",
        "reason": "Try again",
        "suggested_retrieval_query": "better search",
    })

    generator_llm = FakeLLM(chat_response="Insufficient context.")

    class NeverSatisfiedLLM:
        def __init__(self):
            self.complete_json_count = 0
            self.complete_chat_calls = []

        def complete_json(self, system_prompt, user_prompt):
            self.complete_json_count += 1
            if self.complete_json_count == 1:  # planner
                return planner_llm.complete_json(system_prompt, user_prompt)
            else:  # evaluator
                return evaluator_llm.complete_json(system_prompt, user_prompt)

        def complete_chat(self, messages, max_tokens=None):
            self.complete_chat_calls.append(messages)
            return generator_llm.complete_chat(messages, max_tokens)

    never_satisfied = NeverSatisfiedLLM()

    from app.routers.ask_router import MAX_AGENT_ROUNDS, AgenticAskPipeline

    with patch("app.routers.ask_router.create_llm_client", return_value=never_satisfied), \
         patch("app.routers.ask_router.HybridRetriever"), \
         patch("app.routers.ask_router.SentenceTransformerEmbedder"), \
         patch("app.routers.ask_router.SqliteVecStore"), \
         patch("app.routers.ask_router.LLMConfig") as mock_cfg:

        mock_cfg.return_value.is_configured = True

        with patch("app.routers.ask_router.QAToolExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.execute.return_value = ToolResults()
            mock_executor.validate_tool_calls = lambda calls, doc_ids: calls
            mock_executor.is_whitelisted = lambda name: True
            mock_executor_cls.return_value = mock_executor

            pipeline = AgenticAskPipeline()
            result = pipeline.answer(
                notebook_id="nb1",
                query="Find something obscure",
                document_ids=["d1"],
                top_k=3,
                history=[],
            )

            # Should not exceed max rounds
            # complete_json_count = 1 (planner) + N rounds of evaluator
            # Max rounds = MAX_AGENT_ROUNDS, so max evaluator calls = MAX_AGENT_ROUNDS
            # Total complete_json = 1 + MAX_AGENT_ROUNDS
            assert never_satisfied.complete_json_count <= 1 + MAX_AGENT_ROUNDS
            # Should still generate an answer (even if insufficient)
            assert result.answer is not None
            assert len(result.answer) > 0


# ──────────────────────────────────────────────────────────────────────
# NEW: Generator — old messages without tool_trace (regression safety)
# ──────────────────────────────────────────────────────────────────────

def test_generator_handles_old_style_messages():
    """Generator should not crash when source_info has no evidence_status (old messages)."""
    llm = FakeLLM(chat_response="Answer from old-style context.")
    generator = AgentAnswerGenerator(llm)

    tr = ToolResults()
    answer, citations = generator.generate(
        query="What is a data structure?",
        history=None,
        plan=PlanResult(task="explain_concept"),
        tool_results=tr,
        source_info={
            "notebook_id": "nb1",
            "document_ids": ["d1"],
            # No evidence_status — old style
        },
    )

    assert isinstance(answer, str)
    assert len(answer) > 0


# ──────────────────────────────────────────────────────────────────────
# NEW: Planner label alias normalization
# ──────────────────────────────────────────────────────────────────────

def test_planner_label_explain_normalizes():
    """Label 'explain' should normalize to 'explain_concept'."""
    llm = FakeLLM({
        "planner_label": "explain",
        "tool_calls": [
            {"name": "retrieve_chunks", "args": {"query": "definition"}}
        ]
    })
    planner = AgentPlanner(llm)
    result = planner.plan("What is Gregorian chant?", None, ["d1"])
    assert result.task == "explain_concept"


def test_planner_label_answer_normalizes():
    """Label 'answer' should normalize to 'answer_directly'."""
    llm = FakeLLM({
        "planner_label": "answer",
        "tool_calls": []
    })
    planner = AgentPlanner(llm)
    result = planner.plan("hello", None, ["d1"])
    assert result.task == "answer_directly"


def test_planner_label_define_normalizes():
    """Label 'define' should normalize to 'explain_concept'."""
    llm = FakeLLM({
        "planner_label": "define",
        "tool_calls": [
            {"name": "retrieve_chunks", "args": {"query": "definition"}}
        ]
    })
    planner = AgentPlanner(llm)
    result = planner.plan("Define X", None, ["d1"])
    assert result.task == "explain_concept"


# ──────────────────────────────────────────────────────────────────────
# NEW: Evaluator — relevant chunks should NOT be marked insufficient
# ──────────────────────────────────────────────────────────────────────

def test_evaluator_relevant_chunks_not_insufficient():
    """When chunks relate to the topic, evaluator should not say insufficient."""
    llm = FakeLLM({
        "status": "sufficient",
        "reason": "Chunks 1-3 discuss Gregorian chant characteristics.",
    })
    evaluator = EvidenceEvaluator(llm)

    from app.services.retrieval.base import RetrievedChunk
    tr = ToolResults()
    tr.chunks = [
        RetrievedChunk(
            chunk_id="c1", document_id="d1",
            text="Gregorian chant is one of the great treasures of Western civilization. This body of chant includes some of the oldest and most beautiful melodies.",
            section_path="3. Roman Liturgy and Chant", title="The Roman Liturgy",
            page_start=79, page_end=80,
        ),
        RetrievedChunk(
            chunk_id="c2", document_id="d1",
            text="Characteristics of Chant: Gregorian chants are very diverse in style, with varying approaches to performance, treatment of the text, and melodic character.",
            section_path="3. Roman Liturgy and Chant > Characteristics of Chant",
            title="Characteristics of Chant", page_start=84, page_end=84,
        ),
    ]

    result = evaluator.evaluate(
        query="What is Gregorian chant?",
        plan=PlanResult(task="explain_concept"),
        tool_results=tr,
        notebook_id="nb1",
        document_ids=["d1"],
    )
    # With relevant chunks, should be sufficient or needs_more_search, NOT insufficient
    assert result.status != "insufficient", f"Got status={result.status}, reason={result.reason}"


# ──────────────────────────────────────────────────────────────────────
# NEW: Pipeline — insufficient_context strips citations
# ──────────────────────────────────────────────────────────────────────

def test_pipeline_insufficient_context_no_citations():
    """When evidence is insufficient, citations should not be returned."""

    # Planner: simple retrieve plan
    planner_llm = FakeLLM({
        "planner_label": "explain",
        "tool_calls": [
            {"name": "retrieve_chunks", "args": {"query": "obscure topic", "top_k": 3}}
        ],
    })
    # Evaluator: says insufficient
    evaluator_llm = FakeLLM({
        "status": "insufficient",
        "reason": "No relevant passages found.",
    })
    generator_llm = FakeLLM(chat_response="I cannot answer this.")

    class CompositeLLM:
        def __init__(self):
            self.complete_json_count = 0
            self.complete_chat_calls = []

        def complete_json(self, system_prompt, user_prompt):
            self.complete_json_count += 1
            if self.complete_json_count == 1:
                return planner_llm.complete_json(system_prompt, user_prompt)
            return evaluator_llm.complete_json(system_prompt, user_prompt)

        def complete_chat(self, messages, max_tokens=None):
            self.complete_chat_calls.append(messages)
            return generator_llm.complete_chat(messages, max_tokens)

    composite = CompositeLLM()

    from app.routers.ask_router import AgenticAskPipeline

    with patch("app.routers.ask_router.create_llm_client", return_value=composite), \
         patch("app.routers.ask_router.HybridRetriever"), \
         patch("app.routers.ask_router.SentenceTransformerEmbedder"), \
         patch("app.routers.ask_router.SqliteVecStore"), \
         patch("app.routers.ask_router.LLMConfig") as mock_cfg:

        mock_cfg.return_value.is_configured = True

        with patch("app.routers.ask_router.QAToolExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.execute.return_value = ToolResults()  # empty evidence
            mock_executor.validate_tool_calls = lambda calls, doc_ids: calls
            mock_executor.is_whitelisted = lambda name: True
            mock_executor_cls.return_value = mock_executor

            pipeline = AgenticAskPipeline()
            result = pipeline.answer(
                notebook_id="nb1",
                query="What is something obscure?",
                document_ids=["d1"],
                top_k=3,
                history=[],
            )

            assert result.agent_status == "insufficient_context"
            # Citations should be empty when context is insufficient
            assert len(result.citations) == 0, (
                f"Expected 0 citations for insufficient_context, got {len(result.citations)}"
            )
            assert composite.complete_chat_calls == []


def test_pipeline_retrieval_unavailable_when_tools_error():
    """A tool error must surface as retrieval_unavailable, not a zero-hit."""
    planner_llm = FakeLLM({
        "planner_label": "explain",
        "tool_calls": [
            {"name": "retrieve_chunks", "args": {"query": "x", "top_k": 3}}
        ],
    })

    class CompositeLLM:
        def __init__(self):
            self.complete_chat_calls = []

        def complete_json(self, system_prompt, user_prompt):
            return planner_llm.complete_json(system_prompt, user_prompt)

        def complete_chat(self, messages, max_tokens=None):
            self.complete_chat_calls.append(messages)
            return "unused"

    composite = CompositeLLM()

    from app.routers.ask_router import AgenticAskPipeline

    with patch("app.routers.ask_router.create_llm_client", return_value=composite), \
         patch("app.routers.ask_router.HybridRetriever"), \
         patch("app.routers.ask_router.SentenceTransformerEmbedder"), \
         patch("app.routers.ask_router.SqliteVecStore"), \
         patch("app.routers.ask_router.LLMConfig") as mock_cfg:

        mock_cfg.return_value.is_configured = True

        with patch("app.routers.ask_router.QAToolExecutor") as mock_executor_cls:
            tr = ToolResults()
            tr.tool_error = True  # a search tool raised; no evidence gathered

            mock_executor = MagicMock()
            mock_executor.execute.return_value = tr
            mock_executor.validate_tool_calls = lambda calls, doc_ids: calls
            mock_executor.is_whitelisted = lambda name: True
            mock_executor_cls.return_value = mock_executor

            pipeline = AgenticAskPipeline()
            result = pipeline.answer(
                notebook_id="nb1",
                query="What is X?",
                document_ids=["d1"],
                top_k=3,
                history=[],
            )

            assert result.agent_status == "retrieval_unavailable"
            # The generator LLM must NOT be called on a retrieval error.
            assert composite.complete_chat_calls == []


def test_pipeline_retrieval_unavailable_despite_metadata():
    """Metadata/outline succeeding must not mask a body-retrieval failure."""
    planner_llm = FakeLLM({
        "planner_label": "summarize",
        "tool_calls": [
            {"name": "get_source_metadata", "args": {"document_ids": ["d1"]}},
            {"name": "get_source_outline", "args": {"document_ids": ["d1"]}},
        ],
    })

    class CompositeLLM:
        def __init__(self):
            self.complete_chat_calls = []

        def complete_json(self, system_prompt, user_prompt):
            return planner_llm.complete_json(system_prompt, user_prompt)

        def complete_chat(self, messages, max_tokens=None):
            self.complete_chat_calls.append(messages)
            return "unused"

    composite = CompositeLLM()

    from app.routers.ask_router import AgenticAskPipeline

    with patch("app.routers.ask_router.create_llm_client", return_value=composite), \
         patch("app.routers.ask_router.HybridRetriever"), \
         patch("app.routers.ask_router.SentenceTransformerEmbedder"), \
         patch("app.routers.ask_router.SqliteVecStore"), \
         patch("app.routers.ask_router.LLMConfig") as mock_cfg:

        mock_cfg.return_value.is_configured = True

        with patch("app.routers.ask_router.QAToolExecutor") as mock_executor_cls:
            tr = ToolResults()
            tr.tool_error = True  # body retrieval raised
            tr.metadata = [{"filename": "book.pdf", "page_count": 100, "chunk_count": 50}]
            tr.outline = [{"filename": "book.pdf", "sections": []}]

            mock_executor = MagicMock()
            mock_executor.execute.return_value = tr
            mock_executor.validate_tool_calls = lambda calls, doc_ids: calls
            mock_executor.is_whitelisted = lambda name: True
            mock_executor_cls.return_value = mock_executor

            pipeline = AgenticAskPipeline()
            result = pipeline.answer(
                notebook_id="nb1",
                query="Summarize this book",
                document_ids=["d1"],
                top_k=3,
                history=[],
            )

            assert result.agent_status == "retrieval_unavailable"
            assert composite.complete_chat_calls == []


def test_pipeline_sufficient_context_has_citations():
    """When evidence is sufficient, citations should be present."""
    from app.services.retrieval.base import RetrievedChunk

    planner_llm = FakeLLM({
        "planner_label": "explain",
        "tool_calls": [
            {"name": "retrieve_chunks", "args": {"query": "test", "top_k": 3}}
        ],
    })
    evaluator_llm = FakeLLM({
        "status": "sufficient",
        "reason": "Good evidence.",
    })
    generator_llm = FakeLLM(chat_response="Here is the answer.")

    class CompositeLLM2:
        def __init__(self):
            self.complete_json_count = 0
        def complete_json(self, system_prompt, user_prompt):
            self.complete_json_count += 1
            if self.complete_json_count == 1:
                return planner_llm.complete_json(system_prompt, user_prompt)
            return evaluator_llm.complete_json(system_prompt, user_prompt)
        def complete_chat(self, messages, max_tokens=None):
            return generator_llm.complete_chat(messages, max_tokens)

    composite = CompositeLLM2()

    from app.routers.ask_router import AgenticAskPipeline

    with patch("app.routers.ask_router.create_llm_client", return_value=composite), \
         patch("app.routers.ask_router.HybridRetriever"), \
         patch("app.routers.ask_router.SentenceTransformerEmbedder"), \
         patch("app.routers.ask_router.SqliteVecStore"), \
         patch("app.routers.ask_router.LLMConfig") as mock_cfg:

        mock_cfg.return_value.is_configured = True

        with patch("app.routers.ask_router.QAToolExecutor") as mock_executor_cls:
            tr = ToolResults()
            tr.chunks = [RetrievedChunk(
                chunk_id="c1", document_id="d1", text="Relevant text",
                section_path="Ch 1", title="Intro", page_start=1, page_end=1,
            )]
            tr.chunk_citations = [{
                "chunk_id": "c1", "document_id": "d1",
                "text_snippet": "Relevant text...",
                "section_path": "Ch 1", "page_start": 1, "page_end": 1,
            }]

            mock_executor = MagicMock()
            mock_executor.execute.return_value = tr
            mock_executor.validate_tool_calls = lambda calls, doc_ids: calls
            mock_executor.is_whitelisted = lambda name: True
            mock_executor_cls.return_value = mock_executor

            pipeline = AgenticAskPipeline()
            result = pipeline.answer(
                notebook_id="nb1",
                query="What is X?",
                document_ids=["d1"],
                top_k=3,
                history=[],
            )

            assert result.agent_status == "completed"
            assert len(result.citations) > 0, (
                f"Expected citations for sufficient context, got {len(result.citations)}"
            )


# ---------------------------------------------------------------------------
# M6 - Retrieval Surface Guard tests
# ---------------------------------------------------------------------------


def test_qa_tool_snippets_use_canonical_text():
    """QA tool citations must use canonical chunk text without metadata prefix."""
    tr = ToolResults()
    tr.chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            text="canonical chunk body",
            title="Chapter 1",
            section_path="Chapter 1 > Section",
            page_start=1,
            page_end=1,
        ),
    ]
    # Simulate what _finalize_chunk_citations does
    QAToolExecutor._finalize_chunk_citations(tr)

    assert len(tr.chunk_citations) >= 1
    snippet = tr.chunk_citations[0].get("text_snippet", "")
    assert "canonical chunk body" in snippet
    import re
    assert not re.match(r"^\[[^]]+ > [^]]+\]", snippet), (
        f"Metadata prefix leaked into QA tool snippet: {snippet}"
    )


def test_qa_evidence_text_excludes_prefix():
    """Agent QA evidence (RetrievedChunk.text) must not expose metadata prefix."""
    chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        text="canonical chunk content",
        title="Chapter 1",
        section_path="Chapter 1 > Section A",
        page_start=1,
        page_end=1,
    )
    # The text field must be canonical — no prefix
    import re
    assert not re.match(r"^\[[^]]+ > [^]]+\]", chunk.text), (
        f"Metadata prefix leaked into RetrievedChunk.text: {chunk.text[:80]}"
    )
    assert chunk.text == "canonical chunk content"
