"""ReAct evidence loop: the model sees each tool result and picks the next tool.

The planner's own tool calls run first (same first round as the old flow), so
easy questions cost one extra LLM turn. Then, up to MAX_REACT_STEPS tool calls,
the model writes a short reason and calls one tool via native function
calling, or stops. The loop only gathers evidence: the existing evaluator and
answer generator still decide and write the answer.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace

from .agent_planner import PlanResult, ToolCall
from .tools import QAToolExecutor, ToolResults

logger = logging.getLogger(__name__)

MAX_REACT_STEPS = 8
MAX_REACT_SECONDS = 60
_OBS_CHUNKS = 8
_OBS_CHARS = 500


class ReactError(RuntimeError):
    """Malformed model output or unsupported client; caller falls back."""


_DOC_IDS = {"type": "array", "items": {"type": "string"},
            "description": "Optional. Omit to use all selected sources."}


def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {**props, "document_ids": _DOC_IDS},
                       "required": required},
    }}


TOOL_SCHEMAS = [
    _tool("retrieve_chunks",
          "Hybrid keyword + semantic search for passages. Use specific terms from the book.",
          {"query": {"type": "string"}}, ["query"]),
    _tool("find_sections",
          "Find sections whose titles match a hint such as 'Chapter 5' or 'hash tables'.",
          {"section_query": {"type": "string"}}, ["section_query"]),
    _tool("get_section_chunks",
          "Read the passages of given sections (exact paths from find_sections/outline) or of a path prefix.",
          {"section_paths": {"type": "array", "items": {"type": "string"}},
           "section_prefix": {"type": "string"},
           "limit": {"type": "integer", "description": "Max passages, default 20."}}, []),
    _tool("get_source_outline", "Table of contents of the sources.", {}, []),
    _tool("get_source_metadata", "Filename, page count and chunk count of the sources.", {}, []),
]
_WEB_SCHEMA = _tool("web_search", "Search the public web. Only for outside context, never for claims about the books.",
                    {"query": {"type": "string"}}, ["query"])

SYSTEM_PROMPT = """You gather evidence from the user's uploaded sources so that another step can answer a study question. You do not write the answer yourself.

Each turn: first write one or two short sentences saying what you will look for next and why, then call one tool. When the passages shown already cover every part of the question, or more searching is pointless, reply with one short sentence and no tool call.

Rules:
- Search with specific terms the book would use (names, technical words), not the whole question.
- A question naming a chapter or section: find_sections, then get_section_chunks.
- A passage that is cut off, or an answer that continues elsewhere: read that section with get_section_chunks.
- A question with two parts usually needs a search per part.
- Never repeat a search you already made.
- Most questions need one to three searches. Stop as soon as the passages answer the question.

Sources:
{docs}"""


def tool_label(name: str, args: dict) -> str:
    """Short user-facing description of a tool call."""
    if name == "retrieve_chunks":
        return f'Searching for "{args.get("query", "")}"'
    if name == "find_sections":
        return f'Looking for section "{args.get("section_query", "")}"'
    if name == "get_section_chunks":
        target = args.get("section_prefix") or "; ".join(args.get("section_paths") or [])
        return f"Reading {target}" if target else "Reading sections"
    if name == "get_source_outline":
        return "Looking at the outline"
    if name == "get_source_metadata":
        return "Checking source details"
    if name == "web_search":
        return f'Searching the web for "{args.get("query", "")}"'
    return name


def _observe(r: ToolResults, seen: set[str]) -> str:
    """Compact text view of one tool result for the model; new passages only."""
    lines: list[str] = []
    for t in r.trace:
        if t.get("status") == "error":
            lines.append(f"Tool error: {t.get('error', '')[:200]}")
    for m in r.metadata:
        lines.append(f"Source: {m.get('filename', '')} ({m.get('page_count', '?')} pages)")
    for doc in r.outline:
        lines.append(f"Outline of {doc.get('filename', '')}:")
        lines += [f"- {s.get('section_path', '')} (p.{s.get('page_start', '?')})"
                  for s in (doc.get("sections") or [])[:80]]
    if r.section_matches:
        lines.append("Matching sections:")
        lines += [f"- {m.section_path} (p.{m.page_start}-{m.page_end}, {m.chunk_count} passages)"
                  for m in r.section_matches[:15]]
    new = [c for c in r.chunks if c.chunk_id not in seen]
    for c in new[:_OBS_CHUNKS]:
        seen.add(c.chunk_id)
        text = " ".join(c.text.split())[:_OBS_CHARS]
        lines.append(f"[{c.section_path or c.title} | p.{c.page_start}-{c.page_end}] {text}")
    if r.chunks and not new:
        lines.append("No new passages (all already seen).")
    for w in r.web_results[:5]:
        lines.append(f"[web: {w.get('title', '')}] {str(w.get('content') or w.get('text_snippet') or '')[:300]}")
    return "\n".join(lines) or "Nothing found."


def run_react(
    llm,
    tools: QAToolExecutor,
    plan: PlanResult,
    *,
    query: str,
    notebook_id: str,
    document_ids: list[str],
    documents: list[dict],
    top_k: int,
    allow_web: bool,
    on_event=None,
) -> tuple[ToolResults, list[dict], list[str]]:
    """Gather evidence. Returns (merged results, trace dicts, labels of where it looked)."""
    from .ask_pipeline import _merge_tool_results  # avoid import cycle

    if not hasattr(llm, "complete_with_tools"):
        raise ReactError("LLM client has no function calling")
    emit = on_event or (lambda _e: None)
    t0 = time.monotonic()
    merged = ToolResults()
    trace: list[dict] = []
    looked: list[str] = []
    seen: set[str] = set()

    per_call: list[list[str]] = []  # chunk ids per tool call, in rank order

    def run(call: ToolCall) -> ToolResults:
        r = tools.execute(
            plan=replace(plan, tool_calls=[call]),
            notebook_id=notebook_id, document_ids=document_ids, top_k=top_k,
            allow_web=allow_web,
            user_query=call.args.get("query") or query,
        )
        _merge_tool_results(merged, r)
        trace.extend(r.trace)
        per_call.append([c.chunk_id for c in r.chunks])
        return r

    def done(status: str, args: dict):
        trace.append({"name": "react_stop", "status": status, "args": args})
        _interleave(merged, per_call)
        return merged, trace, looked

    # Step 0: the planner's first look.
    first_obs = []
    for call in plan.tool_calls:
        label = tool_label(call.name, call.args)
        looked.append(label)
        emit({"type": "step", "reasoning": "", "label": label})
        first_obs.append(f"{label}:\n{_observe(run(call), seen)}")

    docs = "\n".join(f"- {d.get('filename', '')} (id {d.get('id', '')})" for d in documents) or "- (none)"
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(docs=docs)},
        {"role": "user", "content": f"Question: {query}\n\nFirst look:\n\n" + "\n\n".join(first_obs)},
    ]
    schemas = TOOL_SCHEMAS + ([_WEB_SCHEMA] if allow_web else [])
    steps = 0
    while steps < MAX_REACT_STEPS and time.monotonic() - t0 < MAX_REACT_SECONDS:
        msg = llm.complete_with_tools(messages, schemas, max_tokens=400)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        reasoning = (msg.get("content") or "").strip()
        if not calls:
            return done("ok", {"steps": steps, "note": reasoning[:200]})
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                raise ReactError(f"bad tool arguments for {name}") from exc
            if not isinstance(args, dict):
                raise ReactError(f"bad tool arguments for {name}")
            label = tool_label(name, args)
            looked.append(label)
            emit({"type": "step", "reasoning": reasoning, "label": label})
            reasoning = ""  # one reasoning line per model turn
            steps += 1
            if steps > MAX_REACT_STEPS:
                obs = "Step limit reached; not run."
            else:
                obs = _observe(run(ToolCall(name=name, args=args)), seen)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": obs})
    return done("max_steps", {"steps": steps})


def _interleave(merged: ToolResults, per_call: list[list[str]]) -> None:
    """Reorder chunks round-robin across tool calls (each call's best first).

    The generator reads only the first MAX_BOOK_CITATIONS chunks; plain merge
    order would let the first search fill that budget and drop what later,
    targeted steps found.
    """
    by_id = {c.chunk_id: c for c in merged.chunks}
    order: list[str] = []
    for rank in range(max((len(ids) for ids in per_call), default=0)):
        for ids in per_call:
            if rank < len(ids) and ids[rank] not in order:
                order.append(ids[rank])
    placed = set(order)
    rest = [c for c in merged.chunks if c.chunk_id not in placed]
    merged.chunks = [by_id[i] for i in order if i in by_id] + rest
