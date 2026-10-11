"""AgentPlanner — LLM-based tool planning for agentic QA.

Given the user query, conversation history, and available tools,
the planner returns a JSON plan specifying:
  - planner_label: observability-only label (NOT used for control flow)
  - user_goal: what the user ultimately wants
  - sub_questions: decomposed information needs
  - tool_calls: which tools to invoke

Does NOT answer the user question — just plans the tool strategy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .query_intent import is_overview_query

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool whitelist — single source of truth for allowed tools
# ---------------------------------------------------------------------------

WHITELIST_TOOLS = {
    "get_source_outline",
    "get_source_metadata",
    "retrieve_chunks",
    "find_sections",
    "get_section_chunks",
    "web_search",
    "get_source_chunks",
}


# ---------------------------------------------------------------------------
# Known task labels — observability only, NOT used for control flow
# ---------------------------------------------------------------------------

KNOWN_LABELS = {
    "study_help",
    "overview",
    "explain_concept",
    "compare",
    "specific_search",
    "answer_directly",
    "ask_clarification",
    "summarize",
    "locate",
    "clarify",
}


# Map common label variants to canonical forms
_NORMALIZE_TASK_ALIASES = {
    # Tool-name-as-task mistakes
    "retrieve_chunks": "specific_search",
    "get_source_outline": "overview",
    "get_source_metadata": "overview",
    "find_sections": "locate",
    "get_section_chunks": "specific_search",
    "web_search": "specific_search",
    # Common LLM output variants
    "explain": "explain_concept",
    "answer": "answer_directly",
    "search": "specific_search",
    "define": "explain_concept",
    "summary": "summarize",
}


_SECTION_REF_RE = re.compile(
    r"\b(?:chapter|ch|section|sec)\s*#?\s*(\d{1,3})\b",
    re.IGNORECASE,
)
_SECTION_CONTENT_RE = re.compile(
    r"\b(discuss|discussed|about|summari[sz]e|summary|explain|cover|covered|"
    r"content|main idea|theme|themes|say|describe|overview|compare|contrast)\b",
    re.IGNORECASE,
)
_SECTION_STRUCTURAL_RE = re.compile(
    r"\b(title|titled|name|named|heading|called|page|pages|start|starts|"
    r"begin|begins|where|located|location)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are a study companion planner. Your job is to decide which tools to call to answer the user.

Available tools:
- get_source_outline: Get the table of contents / section structure of selected documents.
- get_source_metadata: Get document metadata (page count, chunk count, filename).
- retrieve_chunks: Search for relevant text chunks across selected documents. Args: query (string), document_ids (list), top_k (int), original_query (string, optional — preserves the user's original question for query expansion).
- find_sections: Locate sections matching a natural-language hint like "Chapter 1", "graphs", "dynamic programming". Args: section_query (string), document_ids (list).
- get_section_chunks: Retrieve chunks from specific sections. Args: section_paths (list of strings) OR section_prefix (string), document_ids (list), limit (int).
- web_search: Search the public internet for external clarification. Args: query (string), max_results (int). This tool is only available when Internet research is enabled.

Rules:
0. The CURRENT user query is authoritative. Use conversation history only for pronoun/reference resolution. Do NOT plan tools for a previous topic if it differs from the current query.
1. Do NOT answer the question. Only plan tools.
2. Only use tools from the available list above.
3. Break the user's question into sub-questions when helpful (e.g. compare questions -> one sub-question per document).
4. If the question mentions a chapter/section (e.g. "Summarize Chapter 1", "What does Chapter 3 say"), use find_sections first, then get_section_chunks.
5. If the question is about source structure or scope, use get_source_outline.
6. For concept-definition questions ("what is X", "define X", "explain X", "meaning of X"):
   - Use retrieve_chunks with a query that combines the concept with terms like "definition", "introduction", "characteristics", "overview".
   - If the selected source has relevant chapter headings visible in prior context, also plan find_sections to locate the best section.
7. For other concept or content questions, use retrieve_chunks with a well-phrased search query.
8. If the question is ambiguous (e.g. "Summarize Chapter 1" with multiple sources), mark clarify_needed: true and include a clarifying question.
9. If no sources are selected, mark clarify_needed: true.
10. If the question is vague ("what should I do", "help me"), still try to use available tools but note it may need clarification.
11. Use web_search only when Internet research is enabled AND the selected sources may not explain a term, need outside context, or the user asks for current/external information. Always prefer source tools first for claims about uploaded books.
12. Query contract for retrieve_chunks: Your query argument is the planner_search_query — preserve concept words, citations, chapter markers, and cross-document keywords from the user's original question. Pass the user's untransformed question as original_query so the system can expand it deterministically with task-specific keywords. Do NOT strip quoted phrases, proper nouns, or technical terms.
13. Set planner_label to a short description for display ONLY — it does NOT change tool behavior.

Return ONLY valid JSON:
{
  "planner_label": "summarize | compare | explain | study_help | locate | clarify | answer",
  "user_goal": "What the user ultimately wants in one sentence",
  "clarify_needed": false,
  "clarifying_question": "",
  "sub_questions": [
    {
      "id": "sq1",
      "question": "What specific thing do we need to find out?",
      "scope": {
        "document_ids": ["id1"],
        "section_hint": "optional section hint",
        "concepts": ["concept1"]
      },
      "evidence_needed": "What kind of passages or data would answer this?"
    }
  ],
  "tool_calls": [
    {"name": "get_source_outline", "args": {"document_ids": ["id1"]}},
    {"name": "find_sections", "args": {"section_query": "Chapter 1", "document_ids": ["id1"]}},
    {"name": "get_section_chunks", "args": {"section_paths": [], "document_ids": ["id1"], "limit": 15}},
    {"name": "retrieve_chunks", "args": {"query": "rewritten query", "document_ids": ["id1"], "top_k": 8}},
    {"name": "web_search", "args": {"query": "external clarification query", "max_results": 5}}
  ]
}
"""


# ---------------------------------------------------------------------------
# Plan types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class SubQuestion:
    id: str
    question: str = ""
    scope: dict = field(default_factory=dict)
    evidence_needed: str = ""


@dataclass
class PlanResult:
    """Planner output.

    Attributes:
        task (planner_label): Observability-only label. Do NOT use for control flow.
        user_goal: One-sentence description of what the user wants.
        sub_questions: Decomposed information needs.
        tool_calls: Whitelisted tool calls to execute.
        clarify_needed: Whether the agent should ask clarification instead of answering.
        clarifying_question: The question to ask the user if clarification needed.
    """
    task: str  # kept as 'task' for backward compat; same as planner_label
    tool_calls: list[ToolCall] = field(default_factory=list)
    user_goal: str = ""
    sub_questions: list[SubQuestion] = field(default_factory=list)
    clarify_needed: bool = False
    clarifying_question: str = ""


def _augment_structural_tool_calls(
    query: str,
    tool_calls: list[ToolCall],
    document_ids: list[str],
) -> list[ToolCall]:
    """Add deterministic structural tools for explicit chapter/section queries."""
    if not document_ids:
        return tool_calls

    section_numbers = list(dict.fromkeys(_SECTION_REF_RE.findall(query or "")))
    if not section_numbers:
        return tool_calls

    bounded_doc_ids = document_ids[:5]
    augmented = list(tool_calls)

    def has_tool(name: str) -> bool:
        return any(tc.name == name for tc in augmented)

    def has_find_for_section(section_number: str) -> bool:
        for tc in augmented:
            if tc.name != "find_sections":
                continue
            hint = str(
                tc.args.get("section_query")
                or tc.args.get("query")
                or tc.args.get("chapter")
                or ""
            )
            if re.search(rf"\b{re.escape(section_number)}\b", hint):
                return True
        return False

    for section_number in section_numbers[:3]:
        section_hint = f"Chapter {section_number}"
        if not has_find_for_section(section_number):
            augmented.append(ToolCall(
                name="find_sections",
                args={"section_query": section_hint, "document_ids": bounded_doc_ids},
            ))

    if not has_tool("get_source_outline"):
        augmented.append(ToolCall(
            name="get_source_outline",
            args={"document_ids": bounded_doc_ids},
        ))

    asks_for_content = bool(_SECTION_CONTENT_RE.search(query or ""))
    if asks_for_content and not has_tool("get_section_chunks"):
        augmented.append(ToolCall(
            name="get_section_chunks",
            args={"section_paths": [], "document_ids": bounded_doc_ids, "limit": 20},
        ))
    if asks_for_content and not has_tool("retrieve_chunks"):
        augmented.append(ToolCall(
            name="retrieve_chunks",
            args={"query": query, "document_ids": bounded_doc_ids, "top_k": 8},
        ))

    return augmented


# A numbered chapter/section reference makes the query scoped, not a
# whole-document overview ("What does Chapter 3 cover?"). Same shape as
# structural_answers._SECTION_REF_RE.
_SCOPED_OVERVIEW_RE = re.compile(
    r"\b(?:chapter|ch|section|sec)\s*#?\s*\d{1,3}\b",
    re.IGNORECASE,
)


def _augment_overview_tool_calls(
    query: str,
    tool_calls: list[ToolCall],
    document_ids: list[str],
) -> list[ToolCall]:
    """Route whole-document overview queries through deterministic coverage.

    Branch C (roadmap Phase 2): similarity ranking is a poor proxy for
    whole-source coverage, so for an unscoped overview ("what does this
    source cover?") the plan's similarity ``retrieve_chunks`` calls are
    replaced by a ``get_source_chunks`` sweep (evenly-spread representative
    chunks across the document). Outline/metadata calls are kept. Scoped
    overviews — section-tool plans and numbered chapter references — keep
    their targeted plan. Routed sweeps carry ``_coverage`` so the tool can
    apply coverage sampling only where this routing made the call.
    """
    if not document_ids or not is_overview_query(query):
        return tool_calls
    if _SCOPED_OVERVIEW_RE.search(query or ""):
        return tool_calls
    if any(tc.name in ("find_sections", "get_section_chunks") for tc in tool_calls):
        # The plan already scopes the query to specific sections ("Summarize the
        # methods section", "Summarize Chapter 1"). A whole-document body sweep
        # would bury the targeted sections under the rest of the book.
        return tool_calls
    if not any(tc.name == "retrieve_chunks" for tc in tool_calls):
        # Planner already asked for coverage (or nothing retrievable): keep it.
        if any(tc.name == "get_source_chunks" for tc in tool_calls):
            return tool_calls
        return tool_calls + [ToolCall(
            name="get_source_chunks",
            args={"document_ids": document_ids[:5], "limit": 20, "_coverage": True},
        )]
    routed = [
        tc for tc in tool_calls
        if tc.name != "retrieve_chunks"
    ]
    routed.append(ToolCall(
        name="get_source_chunks",
        args={"document_ids": document_ids[:5], "limit": 20, "_coverage": True},
    ))
    return routed


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class AgentPlanner:
    """LLM-based planner that decides which tools to call and decomposes queries."""

    def __init__(self, llm_client):
        self._llm = llm_client

    def plan(
        self,
        query: str,
        history: list[dict] | None,
        document_ids: list[str],
        allow_web: bool = False,
    ) -> PlanResult:
        """Plan tool calls for the given query.

        Args:
            query: User's natural-language query.
            history: Recent conversation history.
            document_ids: Currently selected document IDs.
            allow_web: Whether web_search is available for this request.

        Returns:
            PlanResult with task label, sub_questions, and tool calls.

        Raises:
            RuntimeError: if LLM API call fails.
        """
        # Build context
        context = f"Query: {query}\n"
        context += f"Selected source count: {len(document_ids)}\n"
        context += f"Internet research enabled: {allow_web}\n"
        if not allow_web:
            context += "Do not call web_search for this request.\n"
        if document_ids:
            context += f"Document IDs: {document_ids[:5]}\n"
        if history:
            context += "Recent conversation:\n"
            for h in history[-4:]:
                role = h.get("role", "unknown")
                text = h.get("text", "")[:200]
                context += f"  [{role}]: {text}\n"

        try:
            result = self._llm.complete_json(
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_prompt=context,
            )
        except Exception as exc:
            logger.error(
                "Planner LLM call failed; falling back to direct retrieval: %s", exc
            )
            return self._fallback_plan(query, document_ids)

        data = result.data

        # ── planner_label (observability only) ──
        label = data.get("planner_label", "") or data.get("task", "answer_directly")
        label = self._normalize_label(label)

        # ── user_goal ──
        user_goal = data.get("user_goal", "")

        # ── clarification ──
        clarify_needed = bool(data.get("clarify_needed", False))
        clarifying_question = data.get("clarifying_question", "")

        # ── sub_questions ──
        sub_questions: list[SubQuestion] = []
        for sq in data.get("sub_questions", []):
            if not isinstance(sq, dict):
                continue
            sub_questions.append(SubQuestion(
                id=sq.get("id", ""),
                question=sq.get("question", ""),
                scope=sq.get("scope", {}) or {},
                evidence_needed=sq.get("evidence_needed", ""),
            ))

        # ── tool calls (whitelist + boundary) ──
        raw_calls = data.get("tool_calls", [])
        tool_calls: list[ToolCall] = []
        for tc in raw_calls:
            name = tc.get("name", "")
            if name not in WHITELIST_TOOLS:
                logger.warning("Planner requested non-whitelisted tool '%s', skipping", name)
                continue
            args = tc.get("args", {})
            if name == "web_search":
                if not allow_web:
                    logger.info("Planner requested web_search while disabled; skipping")
                    continue
                tool_calls.append(ToolCall(name=name, args=args))
                continue
            # Enforce document_ids boundary
            if "document_ids" in args and isinstance(args["document_ids"], list):
                args["document_ids"] = [did for did in args["document_ids"] if did in document_ids]
            if not args.get("document_ids"):
                args["document_ids"] = document_ids[:5]
            tool_calls.append(ToolCall(name=name, args=args))

        tool_calls = _augment_structural_tool_calls(query, tool_calls, document_ids)
        tool_calls = _augment_overview_tool_calls(query, tool_calls, document_ids)

        logger.info(
            "Planner: label=%s tools=%s sub_qs=%d clarify=%s",
            label, [t.name for t in tool_calls], len(sub_questions), clarify_needed,
        )
        return PlanResult(
            task=label,
            tool_calls=tool_calls,
            user_goal=user_goal,
            sub_questions=sub_questions,
            clarify_needed=clarify_needed,
            clarifying_question=clarifying_question,
        )

    def _fallback_plan(self, query: str, document_ids: list[str]) -> PlanResult:
        """Deterministic fallback when the planner LLM call fails.

        Just retrieve chunks for the original query — never crash to 503 because
        the planner's JSON was truncated or malformed.
        """
        return PlanResult(
            task="specific_search",
            tool_calls=_augment_overview_tool_calls(
                query,
                [ToolCall(
                    name="retrieve_chunks",
                    args={"query": query, "document_ids": document_ids[:5], "top_k": 8},
                )],
                document_ids,
            ),
            user_goal=query,
        )

    @staticmethod
    def _normalize_label(raw: str) -> str:
        """Normalize planner labels.

        - Known labels pass through unchanged.
        - Tool-name-as-task mistakes are aliased to safe labels.
        - Completely unknown fall back to 'answer_directly'.
        - This is purely for observability — NOT used in control flow.
        """
        label = raw.strip().lower() if raw else "answer_directly"

        if label in KNOWN_LABELS:
            return label

        # Check tool-name aliases
        if label in _NORMALIZE_TASK_ALIASES:
            aliased = _NORMALIZE_TASK_ALIASES[label]
            logger.info("Planner label '%s' aliased to '%s'", label, aliased)
            return aliased

        logger.warning("Planner returned unknown label '%s', using 'answer_directly'", label)
        return "answer_directly"
