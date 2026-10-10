"""Follow-up query expansion for the ask path.

Turns "continue / go deeper / elaborate" follow-ups into a self-contained
query + plan grounded in the prior exchange, instead of re-asking which concept
the user means.
"""

from __future__ import annotations

import re

from .agent_planner import PlanResult, ToolCall


_FOLLOW_UP_EXPAND_RE = re.compile(
    r"\b("
    r"continue|deepen|deeper|expand|elaborate|more\s+detail|more\s+details|"
    r"tell\s+me\s+more|go\s+deeper|explain\s+more|didn'?t\s+understand|"
    r"not\s+clear|confused"
    r")\b|"
    r"(?:\u7ee7\u7eed|\u518d\u8bb2|\u6df1\u5165|\u8be6\u7ec6|"
    r"\u5c55\u5f00|\u591a\u8bf4|\u8bb2\u89e3)"
    r"|(?:\u6ca1|\u4e0d)(?:\u542c|\u770b|\u61c2|\u660e\u767d)",
    re.IGNORECASE,
)


def _latest_history_text(history: list[dict], role: str) -> str:
    for item in reversed(history or []):
        if item.get("role") == role and str(item.get("text") or "").strip():
            return str(item.get("text") or "").strip()
    return ""


def _latest_history_intent(history: list[dict], role: str) -> str:
    for item in reversed(history or []):
        if item.get("role") == role and str(item.get("intent") or "").strip():
            return str(item.get("intent") or "").strip()
    return ""


def _build_follow_up_query(query: str, history: list[dict], follow_up: bool) -> str:
    if not follow_up or not history:
        return query
    previous_user = _latest_history_text(history, "user")
    previous_answer = _latest_history_text(history, "assistant")
    if not previous_user and not previous_answer:
        return query
    answer_excerpt = previous_answer[:1200]
    return (
        "Follow up on the previous exchange. The user is asking about the same concept or answer, "
        "so deepen or re-explain it directly instead of asking which concept they mean.\n\n"
        f"Previous user question: {previous_user}\n"
        f"Previous assistant answer: {answer_excerpt}\n"
        f"User follow-up: {query}\n\n"
        "Answer the follow-up by continuing that prior topic."
    )


def _should_continue_prior_topic(query: str, history: list[dict], follow_up: bool) -> bool:
    if not follow_up or not history:
        return False
    if not _latest_history_text(history, "user") or not _latest_history_text(history, "assistant"):
        return False
    return bool(_FOLLOW_UP_EXPAND_RE.search(query or ""))


def _fallback_follow_up_plan(
    query: str,
    history: list[dict],
    document_ids: list[str],
    original_plan: PlanResult,
    allow_web: bool = False,
) -> PlanResult:
    previous_user = _latest_history_text(history, "user")
    previous_intent = _latest_history_intent(history, "assistant")
    task = previous_intent or original_plan.task
    if task in {"clarify", "ask_clarification"}:
        task = "explain_concept"
    search_query = " ".join(part for part in [previous_user, query] if part).strip() or query
    tool_calls = [
        ToolCall(
            name="retrieve_chunks",
            args={
                "query": search_query,
                "original_query": query,
                "document_ids": document_ids[:5],
                "top_k": 8,
            },
        )
    ]
    if allow_web:
        tool_calls.append(ToolCall(
            name="web_search",
            args={
                "query": search_query,
                "max_results": 5,
            },
        ))
    return PlanResult(
        task=task or "explain_concept",
        user_goal=(
            "Continue the previous answer and address the user's follow-up "
            "without asking which concept they mean."
        ),
        clarify_needed=False,
        clarifying_question="",
        tool_calls=tool_calls,
    )
