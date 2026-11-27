"""ReAct evidence loop: stop, step cap, malformed output, not-found answer."""

import pytest

from app.services.qa.agent_planner import PlanResult, ToolCall
from app.services.qa.ask_pipeline import _not_found_answer
from app.services.qa.react_loop import MAX_REACT_STEPS, ReactError, _interleave, run_react
from app.services.qa.tools import ToolResults
from app.services.retrieval.base import RetrievedChunk


class FakeTools:
    def __init__(self):
        self.calls = []

    def execute(self, plan, notebook_id, document_ids, top_k, allow_web, user_query):
        call = plan.tool_calls[0]
        self.calls.append(call)
        n = len(self.calls)
        return ToolResults(
            chunks=[RetrievedChunk(chunk_id=f"c{n}", document_id="d1", text=f"passage {n}", section_path="Ch 1")],
            trace=[{"name": call.name, "status": "ok", "args": call.args}],
        )


class FakeLLM:
    """Replays scripted assistant turns; records the messages it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def complete_with_tools(self, messages, tools, max_tokens=None):
        self.seen.append(list(messages))
        return self.turns.pop(0) if self.turns else self.turns_default()

    def turns_default(self):
        return _call("retrieve_chunks", '{"query": "more"}')


def _call(name, arguments, content="Looking further."):
    return {"role": "assistant", "content": content,
            "tool_calls": [{"id": f"id-{name}", "type": "function",
                            "function": {"name": name, "arguments": arguments}}]}


def _run(llm, tools, events=None):
    plan = PlanResult(task="specific_search", tool_calls=[ToolCall(name="retrieve_chunks", args={"query": "q"})])
    return run_react(llm, tools, plan, query="What is X?", notebook_id="nb", document_ids=["d1"],
                     documents=[{"id": "d1", "filename": "book.pdf"}], top_k=10, allow_web=False,
                     on_event=(events.append if events is not None else None))


def test_runs_requested_tool_then_stops():
    tools, events = FakeTools(), []
    llm = FakeLLM([_call("find_sections", '{"section_query": "Chapter 5"}', "Need chapter 5."),
                   {"role": "assistant", "content": "Enough."}])
    merged, trace, looked = _run(llm, tools, events)
    assert [c.name for c in tools.calls] == ["retrieve_chunks", "find_sections"]
    assert {c.chunk_id for c in merged.chunks} == {"c1", "c2"}
    assert trace[-1]["name"] == "react_stop" and trace[-1]["status"] == "ok"
    assert looked == ['Searching for "q"', 'Looking for section "Chapter 5"']
    assert events[1] == {"type": "step", "reasoning": "Need chapter 5.", "label": 'Looking for section "Chapter 5"'}
    # the tool result went back to the model as a tool message
    assert llm.seen[1][-1]["role"] == "tool" and "passage 2" in llm.seen[1][-1]["content"]


def test_step_cap():
    tools = FakeTools()
    _merged, trace, _ = _run(FakeLLM([]), tools)
    assert len(tools.calls) == 1 + MAX_REACT_STEPS  # planner's first look + 8
    assert trace[-1]["status"] == "max_steps"


def test_malformed_arguments_raise_for_fallback():
    with pytest.raises(ReactError):
        _run(FakeLLM([_call("retrieve_chunks", "{not json")]), FakeTools())


def test_client_without_function_calling_raises():
    with pytest.raises(ReactError):
        _run(object(), FakeTools())


def test_interleave_keeps_each_calls_best_in_front():
    def ck(i):
        return RetrievedChunk(chunk_id=i, document_id="d", text=i)
    merged = ToolResults(chunks=[ck(i) for i in ["a1", "a2", "a3", "b1", "b2"]])
    _interleave(merged, [["a1", "a2", "a3"], ["b1", "a2", "b2"]])
    assert [c.chunk_id for c in merged.chunks] == ["a1", "b1", "a2", "a3", "b2"]


def test_not_found_answer_lists_places_once():
    text = _not_found_answer(['Searching for "a"', 'Searching for "a"', "Looking at the outline"])
    assert text.count('Searching for "a"') == 1 and "Looking at the outline" in text
