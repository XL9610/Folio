import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.routers.ask_router import (
    _try_answer_structural_query,
    _with_source_framing_questions,
    _with_unresolved_section_disclaimer,
)
from app.services.qa.agent_generator import AgentAnswerGenerator
from app.services.qa.agent_planner import AgentPlanner, PlanResult, ToolCall
from app.services.qa.tools import (
    QAToolExecutor,
    SectionMatch,
    ToolResults,
    extract_numbered_section_label,
    extract_numbered_section_prefix,
)
from app.services.retrieval.base import RetrievedChunk

from .test_agentic_qa import FakeLLM


MUSIC_SECTIONS = [
    {
        "section_path": "PART FOUR: THE NINETEENTH CENTURY",
        "title": "PART FOUR: THE NINETEENTH CENTURY",
        "page_start": 280,
        "page_end": 299,
        "chunk_count": 2,
    },
    {
        "section_path": "PART FOUR: THE NINETEENTH CENTURY > 19. Romanticism",
        "title": "19. Romanticism",
        "page_start": 281,
        "page_end": 299,
        "chunk_count": 8,
    },
    {
        "section_path": "PART FOUR: THE NINETEENTH CENTURY > 20. Music in the Nineteenth Century",
        "title": "20. Music in the Nineteenth Century",
        "page_start": 300,
        "page_end": 340,
        "chunk_count": 14,
    },
    {
        "section_path": "PART FOUR: THE NINETEENTH CENTURY > 20. Music in the Nineteenth Century > Opera and Song",
        "title": "Opera and Song",
        "page_start": 305,
        "page_end": 315,
        "chunk_count": 4,
    },
    {
        "section_path": "PART FOUR: THE NINETEENTH CENTURY > 21. Late Romanticism",
        "title": "21. Late Romanticism",
        "page_start": 341,
        "page_end": 360,
        "chunk_count": 7,
    },
]


def test_extract_numbered_section_handles_common_outline_shapes():
    assert extract_numbered_section_label(
        "PART FOUR > 20. Music in the Nineteenth Century",
        "",
    ) == (20, "20. Music in the Nineteenth Century")
    assert extract_numbered_section_prefix(
        "PART FOUR > 20. Music in the Nineteenth Century > Opera",
        "",
    ) == (
        20,
        "PART FOUR > 20. Music in the Nineteenth Century",
        "20. Music in the Nineteenth Century",
    )
    assert extract_numbered_section_label(
        "Chapter 1 Art as History",
        "",
    ) == (1, "Chapter 1 Art as History")
    assert extract_numbered_section_label("Chapter 10", "") == (10, "Chapter 10")
    assert extract_numbered_section_label("PART FOUR > 20.1 A Subsection", "") == (None, "")


def test_find_sections_matches_nested_bare_numbered_chapter():
    executor = QAToolExecutor(MagicMock())
    plan = PlanResult(
        task="locate",
        tool_calls=[
            ToolCall(
                name="find_sections",
                args={"section_query": "Chapter 20", "document_ids": ["d1"]},
            )
        ],
    )

    with patch("app.services.qa.tools.get_document") as mock_doc, patch(
        "app.services.qa.tools.get_document_sections"
    ) as mock_sections:
        mock_doc.return_value = {"id": "d1", "notebook_id": "nb1", "filename": "music.pdf"}
        mock_sections.return_value = MUSIC_SECTIONS

        results = executor.execute(plan, "nb1", ["d1"])

    assert results.section_matches
    assert results.section_matches[0].section_path.endswith("20. Music in the Nineteenth Century")
    assert all(
        extract_numbered_section_label(m.section_path, m.title)[0] == 20
        for m in results.section_matches
    )
    assert not any("21. Late Romanticism" in m.section_path for m in results.section_matches)


def test_get_section_chunks_uses_chapter_prefix_for_nested_matches():
    executor = QAToolExecutor(MagicMock())
    plan = PlanResult(
        task="summarize",
        tool_calls=[
            ToolCall(
                name="find_sections",
                args={"section_query": "Chapter 20", "document_ids": ["d1"]},
            ),
            ToolCall(
                name="get_section_chunks",
                args={"section_paths": [], "document_ids": ["d1"], "limit": 20},
            ),
        ],
    )

    with patch("app.services.qa.tools.get_document") as mock_doc, patch(
        "app.services.qa.tools.get_document_sections"
    ) as mock_sections, patch(
        "app.services.qa.tools.get_chunks_by_section_prefix"
    ) as mock_by_prefix:
        mock_doc.return_value = {"id": "d1", "notebook_id": "nb1", "filename": "music.pdf"}
        mock_sections.return_value = MUSIC_SECTIONS
        mock_by_prefix.return_value = []

        executor.execute(plan, "nb1", ["d1"])

    mock_by_prefix.assert_called()
    assert mock_by_prefix.call_args.args[1] == (
        "PART FOUR: THE NINETEENTH CENTURY > 20. Music in the Nineteenth Century"
    )


def test_tool_results_append_retrieval_after_section_chunks_and_renumber_citations():
    executor = QAToolExecutor(
        MagicMock(
            search_with_meta=MagicMock(
                return_value=([
                    RetrievedChunk(
                        chunk_id="c2",
                        document_id="d1",
                        text="Retrieved semantic passage about experiments.",
                        section_path="Experiments > Results",
                        title="Results",
                        page_start=7,
                        page_end=8,
                    )
                ], {})
            )
        )
    )
    plan = PlanResult(
        task="specific_search",
        tool_calls=[
            ToolCall(
                name="get_section_chunks",
                args={
                    "section_paths": ["Experiments"],
                    "document_ids": ["d1"],
                    "limit": 10,
                },
            ),
            ToolCall(
                name="retrieve_chunks",
                args={
                    "query": "experiments results",
                    "document_ids": ["d1"],
                    "top_k": 5,
                },
            ),
        ],
    )

    with patch("app.services.qa.tools.get_document") as mock_doc, patch(
        "app.services.qa.tools.get_chunks_for_section_paths"
    ) as mock_by_paths:
        mock_doc.return_value = {
            "id": "d1",
            "notebook_id": "nb1",
            "filename": "paper.pdf",
        }
        mock_by_paths.return_value = [
            {
                "id": "c1",
                "document_id": "d1",
                "text": "Section passage about experimental setup.",
                "title": "Experiments",
                "section_path": "Experiments",
                "page_start": 6,
                "page_end": 6,
            }
        ]

        results = executor.execute(plan, "nb1", ["d1"], top_k=5)

    assert [chunk.chunk_id for chunk in results.chunks] == ["c1", "c2"]
    assert [
        citation["ref_label"]
        for citation in results.chunk_citations
        if citation.get("source_type", "book") == "book"
    ] == ["S1", "S2"]
    assert {
        citation["chunk_id"]
        for citation in results.chunk_citations
        if citation.get("source_type", "book") == "book"
    } == {"c1", "c2"}


def test_get_section_chunks_falls_back_to_retrieval_for_unmatched_named_section():
    retriever = MagicMock()
    retriever.search.return_value = [
        RetrievedChunk(
            chunk_id="preface-c1",
            document_id="d1",
            text="This preface says the book provides an introduction to immigration law and practice.",
            section_path="Preface",
            title="Preface",
            page_start=3,
            page_end=3,
        )
    ]
    executor = QAToolExecutor(retriever)
    results = ToolResults(target_section="preface")
    tool_call = ToolCall(
        name="get_section_chunks",
        args={"section_paths": [], "document_ids": ["d1"], "limit": 10},
    )

    executor._tool_get_section_chunks(results, tool_call, "nb1", ["d1"], top_k=5)
    executor._finalize_chunk_citations(results)

    assert [chunk.chunk_id for chunk in results.chunks] == ["preface-c1"]
    assert results.chunk_citations[0]["ref_label"] == "S1"
    assert results.trace[-1]["status"] == "fallback_retrieve"


def test_planner_augments_chapter_title_and_content_queries():
    planner = AgentPlanner(FakeLLM({"planner_label": "answer", "tool_calls": []}))

    title_plan = planner.plan("What's the title of chapter 20?", None, ["d1"])
    title_tools = [tc.name for tc in title_plan.tool_calls]
    assert "find_sections" in title_tools
    assert "get_source_outline" in title_tools
    assert "get_section_chunks" not in title_tools

    content_plan = planner.plan("What did chapter 20 discuss?", None, ["d1"])
    content_tools = [tc.name for tc in content_plan.tool_calls]
    assert "find_sections" in content_tools
    assert "get_source_outline" in content_tools
    assert "get_section_chunks" in content_tools
    assert "retrieve_chunks" in content_tools


def test_generator_outline_context_lists_nested_numbered_headings():
    generator = AgentAnswerGenerator(FakeLLM())

    context = generator._build_outline_context(MUSIC_SECTIONS, "music.pdf")

    assert "Detected numbered chapter/section headings (complete)" in context
    assert "20. Music in the Nineteenth Century" in context
    assert "path: PART FOUR: THE NINETEENTH CENTURY > 20. Music in the Nineteenth Century" in context
    assert "Opera and Song" in context


def test_structural_title_and_page_answers_are_direct():
    tr = ToolResults()
    tr.section_matches = [
        SectionMatch(
            document_id="d1",
            section_path="PART FOUR: THE NINETEENTH CENTURY > 20. Music in the Nineteenth Century",
            title="20. Music in the Nineteenth Century",
            page_start=300,
            page_end=340,
            chunk_count=14,
            rank=1,
            match_type="exact_chapter",
        )
    ]

    title = _try_answer_structural_query("What's the title of Chapter 20?", tr)
    page = _try_answer_structural_query("Where does Chapter 20 start?", tr)

    assert title is not None
    assert "20. Music in the Nineteenth Century" in title["answer"]
    assert page is not None
    assert "starts at p. 300" in page["answer"]


def test_section_discussion_answer_names_common_parent_section():
    tr = ToolResults()
    tr.section_matches = [
        SectionMatch(
            document_id="d1",
            section_path="Experiments > Experimental Setup",
            title="Experimental Setup",
            page_start=6,
            page_end=6,
            chunk_count=1,
            rank=1,
            match_type="substring",
        ),
        SectionMatch(
            document_id="d1",
            section_path="Experiments > Main Results",
            title="Main Results",
            page_start=7,
            page_end=8,
            chunk_count=2,
            rank=2,
            match_type="substring",
        ),
    ]

    answer = _try_answer_structural_query("What section discusses experiments?", tr)

    assert answer is not None
    assert "Experiments" in answer["answer"]


def test_unreliable_outline_chapter_list_is_honest_no_answer():
    tr = ToolResults()
    tr.outline = [{
        "document_id": "d1",
        "structure_status": "no_reliable_outline",
        "structure_confidence": "low",
        "sections": [{"section_path": "TOC", "title": "TOC"}],
    }]

    answer = _try_answer_structural_query("What chapters are in this source?", tr)

    assert answer is not None
    assert "not enough information" in answer["answer"]
    assert "cannot identify" in answer["answer"]
    assert "should not infer chapter titles" in answer["answer"]


def test_source_framing_questions_preserve_direct_concept_wording():
    tr = ToolResults()
    tr.chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            text=(
                ("Context before the explicit heading. " * 80)
                +
                "The Economic Problem\n"
                "There are three questions that any economy must face:\n"
                "What goods and services should be produced?\n"
                "How should these goods and services be produced?\n"
                "Who should get the goods and services that have been produced?"
            ),
            title="The Economy and Economic Systems",
            section_path="Chapter 1 > The Economy and Economic Systems",
            page_start=17,
            page_end=18,
        )
    ]
    tr.chunk_citations = [{"chunk_id": "c1", "ref_label": "S1", "source_type": "book"}]

    answer = _with_source_framing_questions(
        "What is the economic problem?",
        tr,
        "The economic problem is about scarcity and choice.",
    )

    assert "goods and services" in answer
    assert "[S1]" in answer


def test_page_window_outline_does_not_match_chapter_number():
    executor = QAToolExecutor(MagicMock())
    plan = PlanResult(
        task="locate",
        tool_calls=[
            ToolCall(
                name="find_sections",
                args={"section_query": "Chapter 20", "document_ids": ["d1"]},
            )
        ],
    )

    with patch("app.services.qa.tools.get_document") as mock_doc, patch(
        "app.services.qa.tools.get_document_sections"
    ) as mock_sections:
        mock_doc.return_value = {
            "id": "d1",
            "notebook_id": "nb1",
            "filename": "no-toc.pdf",
            "structure_status": "page_window_outline",
            "embedded_toc_present": 0,
            "synthetic_outline_mode": "page_windows",
        }
        mock_sections.return_value = [
            {
                "section_path": "Pages 20-27",
                "title": "Pages 20-27",
                "page_start": 20,
                "page_end": 27,
                "chunk_count": 4,
            }
        ]

        results = executor.execute(plan, "nb1", ["d1"])

    assert not results.section_matches
    assert results.trace[-1]["status"] == "no_match"


def test_structural_title_no_match_is_honest_for_page_windows():
    tr = ToolResults()
    tr.outline = [
        {
            "document_id": "d1",
            "filename": "no-toc.pdf",
            "structure_status": "page_window_outline",
            "structure_llm_instruction": "Page windows are not chapters.",
            "sections": [
                {
                    "section_path": "Pages 20-27",
                    "title": "Pages 20-27",
                    "page_start": 20,
                    "page_end": 27,
                    "chunk_count": 4,
                }
            ],
        }
    ]

    answer = _try_answer_structural_query("What's the title of Chapter 20?", tr)

    assert answer is not None
    assert answer["type"] == "section_not_found"
    assert "cannot identify a reliable" in answer["answer"]
    assert "titled **Chapter 20**" not in answer["answer"]


def test_content_chapter_no_match_gets_disclaimer():
    tr = ToolResults()
    tr.outline = [{"structure_status": "page_window_outline"}]
    tr.chunks = [
        MagicMock(chunk_id="c1", text="retrieved passage", document_id="d1"),
    ]

    answer = _with_unresolved_section_disclaimer(
        "What did chapter 20 discuss?",
        tr,
        "The retrieved passages discuss musical style.",
    )

    assert answer.startswith("Note: I could not identify a reliable **Chapter 20** boundary")
    assert "semantically retrieved passages" in answer


def test_qa_question_set_scores_planning_and_structural_answers():
    cases = json.loads(
        (Path(__file__).parent / "fixtures" / "qa_question_set.json").read_text(
            encoding="utf-8"
        )
    )
    llm_by_category = {
        "structure.outline": {"planner_label": "overview", "tool_calls": [
            {"name": "get_source_outline", "args": {"document_ids": ["d1"]}},
        ]},
        "structure.metadata": {"planner_label": "overview", "tool_calls": [
            {"name": "get_source_metadata", "args": {"document_ids": ["d1"]}},
        ]},
        "content.concept": {"planner_label": "explain_concept", "tool_calls": [
            {"name": "retrieve_chunks", "args": {"query": "sonata form definition", "document_ids": ["d1"]}},
        ]},
        "content.compare": {"planner_label": "compare", "tool_calls": []},
    }
    structural_results = ToolResults()
    structural_results.section_matches = [
        SectionMatch(
            document_id="d1",
            section_path="PART FOUR: THE NINETEENTH CENTURY > 20. Music in the Nineteenth Century",
            title="20. Music in the Nineteenth Century",
            page_start=300,
            page_end=340,
            chunk_count=14,
            rank=1,
            match_type="exact_chapter",
        )
    ]

    passed = 0
    total = 0
    for case in cases:
        seed = llm_by_category.get(case["category"], {"planner_label": "answer", "tool_calls": []})
        plan = AgentPlanner(FakeLLM(seed)).plan(case["query"], None, ["d1"])
        planned = [tc.name for tc in plan.tool_calls]

        for expected in case["expected_tools"]:
            total += 1
            passed += int(expected in planned)
        for forbidden in case["forbidden_tools"]:
            total += 1
            passed += int(forbidden not in planned)

        if case.get("expected_answer_contains"):
            answer = _try_answer_structural_query(case["query"], structural_results)
            total += len(case["expected_answer_contains"])
            if answer:
                passed += sum(
                    1 for needle in case["expected_answer_contains"]
                    if needle in answer["answer"]
                )

    score = passed / total
    assert score >= 0.95
