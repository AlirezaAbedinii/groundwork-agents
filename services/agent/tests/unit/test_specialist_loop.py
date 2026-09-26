"""Acceptance tests for the specialist's native tool-call loop.

Each test pins one rule of the loop's contract: what it sends the LLM client
(messages in, native tool calls out), how every tool result goes back to the
model, how it uses the tool registry and the human-approval gate, and when an
attempt fails. The model is scripted and the tools are in-process stubs, so
nothing here reads fixtures from disk.
"""

import itertools
import json
import threading
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from orchestrator.agents.specialists import ResearchSpecialist
from orchestrator.agents.specialists.base import FEEDBACK_MARKER, SpecialistError, SpecialistResult
from orchestrator.config import get_settings
from orchestrator.llm.mock import LLMResponse, ToolCall
from orchestrator.tools.base import InMemoryInvocationStore, ToolContext, ToolExecutionError, ToolSpec
from orchestrator.tools.registry import ToolRegistry

pytestmark = pytest.mark.anyio

SPEC = {"id": "s1", "description": "Compare Chroma and Qdrant", "expected_output_format": "a bullet list"}
CTX = ToolContext(task_id="t1", specialist="research", subtask_id="s1")

_call_ids = itertools.count(1)


# --- the scripted model --------------------------------------------------------


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id=f"call_{next(_call_ids)}", name=name, arguments=arguments)


def turn(*tool_calls: ToolCall, text: str = "") -> LLMResponse:
    """A model turn that asks for one or more tool calls."""
    return LLMResponse(text=text, tool_calls=tool_calls)


def final(text: str) -> LLMResponse:
    """A model turn with no tool calls: the deliverable."""
    return LLMResponse(text=text)


class ScriptedLLM:
    """Plays the scripted turns in order and records what each chat call received."""

    def __init__(self, *responses: LLMResponse):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, agent, messages, *, tools=None, output_schema=None, producer_provider=None):
        # a copy: the loop keeps appending to the list it passed in
        self.calls.append(
            {"agent": agent, "messages": list(messages), "tools": tools, "output_schema": output_schema}
        )
        if not self.responses:
            raise AssertionError(f"chat call {len(self.calls)} was not scripted")
        return self.responses.pop(0)


class ScriptedGate:
    """Answers each sensitive call with the next scripted decision and records the consultation."""

    def __init__(self, store: InMemoryInvocationStore, *decisions: dict):
        self.store = store
        self.decisions = list(decisions)
        self.calls: list[dict] = []

    def __call__(self, tool_name: str, arguments: dict, iteration: int, transcript: list[str]) -> dict:
        self.calls.append(
            {
                "tool": tool_name,
                "arguments": dict(arguments),
                "iteration": iteration,
                "transcript": list(transcript),
                "invocations_so_far": len(self.store.records),
            }
        )
        if not self.decisions:
            raise AssertionError(f"the gate was consulted for an unscripted call: {tool_name}({arguments})")
        return self.decisions.pop(0)


# --- stub tools ----------------------------------------------------------------


class EchoIn(BaseModel):
    msg: str
    delay: float = 0.0


class EchoOut(BaseModel):
    msg: str


def make_registry(store: InMemoryInvocationStore, barrier: threading.Barrier | None = None) -> ToolRegistry:
    """echo (sensitive when msg == "danger"), boom (always fails), once (one call per task)."""

    def echo(args: EchoIn, ctx: ToolContext) -> EchoOut:
        if barrier is not None:
            barrier.wait()  # returns only when every call of the batch is running at the same time
        time.sleep(args.delay)
        return EchoOut(msg=args.msg)

    def boom(args: EchoIn, ctx: ToolContext) -> EchoOut:
        raise ToolExecutionError("boom failed: kaput")

    common = dict(input_schema=EchoIn, output_schema=EchoOut, owners=frozenset({"research"}))
    registry = ToolRegistry(store)
    registry.register(
        ToolSpec(name="echo", description="Echo the message back", rate_limit=20, handler=echo,
                 sensitive_when=lambda args: args.msg == "danger", **common)
    )
    registry.register(ToolSpec(name="boom", description="A tool that always fails", rate_limit=20, handler=boom,
                               **common))
    registry.register(ToolSpec(name="once", description="Echo, allowed once per task", rate_limit=1,
                               handler=lambda args, ctx: EchoOut(msg=args.msg), **common))
    return registry


@pytest.fixture()
def store() -> InMemoryInvocationStore:
    return InMemoryInvocationStore()


def tool_messages(messages) -> list[ToolMessage]:
    return [message for message in messages if isinstance(message, ToolMessage)]


# --- 1. the first call ---------------------------------------------------------


async def test_a_reply_without_tool_calls_is_the_result_after_one_chat(store):
    registry = make_registry(store)
    llm = ScriptedLLM(final("- Chroma: embedded\n- Qdrant: server"))

    result = await ResearchSpecialist(llm, registry).execute(SPEC, {"s0": "PRIOR-OUTPUT"}, None, CTX)

    assert result == SpecialistResult(output="- Chroma: embedded\n- Qdrant: server", tool_calls=[])
    (only,) = llm.calls
    assert only["agent"] == "research"
    assert only["output_schema"] is None
    # the tools travel natively, as the registry's definitions ...
    assert only["tools"] == registry.tool_definitions_for("research")
    system, human = only["messages"]
    assert isinstance(system, SystemMessage) and isinstance(human, HumanMessage)
    # ... and never as text
    for definition in only["tools"]:
        assert definition["function"]["description"] not in system.text + human.text
    assert ResearchSpecialist.ROLE in system.text
    assert SPEC["description"] in human.text
    assert SPEC["expected_output_format"] in human.text
    assert "PRIOR-OUTPUT" in human.text  # inputs from completed subtasks
    assert not store.records


# --- 2. one call, one answer ---------------------------------------------------


async def test_a_tool_call_is_answered_by_one_tool_message(store):
    long_msg = "x" * 3000
    first = turn(call("echo", msg=long_msg), text="Checking the sources.")
    llm = ScriptedLLM(first, final("DONE"))

    result = await ResearchSpecialist(llm, make_registry(store)).execute(SPEC, {}, None, CTX)

    assert result == SpecialistResult(output="DONE", tool_calls=["echo"])
    (record,) = store.records  # the registry ran it, with the loop's context
    assert (record.tool_name, record.status, record.arguments) == ("echo", "success", {"msg": long_msg})
    assert (record.task_id, record.subtask_id) == (CTX.task_id, CTX.subtask_id)

    system, human, assistant, answer = llm.calls[1]["messages"]
    assert isinstance(system, SystemMessage) and isinstance(human, HumanMessage)
    assert isinstance(assistant, AIMessage) and isinstance(answer, ToolMessage)
    (requested,) = first.tool_calls
    assert assistant.text == "Checking the sources."
    assert [(c["id"], c["name"], c["args"]) for c in assistant.tool_calls] == [
        (requested.id, "echo", {"msg": long_msg})
    ]
    assert (answer.tool_call_id, answer.name, answer.status) == (requested.id, "echo", "success")
    assert answer.content == json.dumps({"msg": long_msg})[:2000]  # the output as JSON, cut at 2000 chars


# --- 3. a batch is concurrent, ordered, and one turn -----------------------------


async def test_two_calls_in_one_turn_run_together_and_count_as_one_turn(store, monkeypatch):
    # the batch plus the final answer fit a budget of two turns only if the batch is one turn
    monkeypatch.setattr(get_settings(), "max_tool_iterations", 2)
    # both calls must be in flight together to pass the barrier; the first one then finishes last
    registry = make_registry(store, barrier=threading.Barrier(2, timeout=5))
    batch = turn(call("echo", msg="first", delay=0.1), call("echo", msg="second"))
    llm = ScriptedLLM(batch, final("BOTH"))

    result = await ResearchSpecialist(llm, registry).execute(SPEC, {}, None, CTX)

    assert [r.status for r in store.records] == ["success", "success"], "the two calls did not run concurrently"
    assert result == SpecialistResult(output="BOTH", tool_calls=["echo", "echo"])
    assert len(llm.calls) == 2
    *_, assistant, first_answer, second_answer = llm.calls[1]["messages"]
    assert len(assistant.tool_calls) == 2
    # answered in call order, not in completion order
    assert [first_answer.tool_call_id, second_answer.tool_call_id] == [c.id for c in batch.tool_calls]
    assert [json.loads(first_answer.content), json.loads(second_answer.content)] == [
        {"msg": "first"}, {"msg": "second"},
    ]


# --- 4. tool errors are results; anything else fails the attempt ------------------


async def test_tool_errors_go_back_to_the_model_and_other_errors_escape(store):
    llm = ScriptedLLM(turn(call("boom", msg="x")), final("RECOVERED"))

    result = await ResearchSpecialist(llm, make_registry(store)).execute(SPEC, {}, None, CTX)

    # a ToolError becomes a result the model sees, and the attempt goes on;
    # the call still ran, so it is listed
    assert result == SpecialistResult(output="RECOVERED", tool_calls=["boom"])
    (error,) = tool_messages(llm.calls[1]["messages"])
    assert error.status == "error"
    assert "boom failed: kaput" in error.content
    assert [r.status for r in store.records] == ["failure"]

    class BrokenStore(InMemoryInvocationStore):
        def record(self, record):
            raise RuntimeError("invocation store unavailable")

    # not a ToolError (here the invocation log itself fails): not for the model, it fails the attempt
    llm = ScriptedLLM(turn(call("echo", msg="hi")), final("unreachable"))
    with pytest.raises(RuntimeError, match="invocation store unavailable"):
        await ResearchSpecialist(llm, make_registry(BrokenStore())).execute(SPEC, {}, None, CTX)
    assert len(llm.calls) == 1


# --- 5. rate limits are tool errors too ------------------------------------------


async def test_a_rate_limited_call_comes_back_as_an_error_result(store):
    llm = ScriptedLLM(turn(call("once", msg="a")), turn(call("once", msg="b")), final("DONE"))

    result = await ResearchSpecialist(llm, make_registry(store)).execute(SPEC, {}, None, CTX)

    assert result == SpecialistResult(output="DONE", tool_calls=["once", "once"])
    assert [r.status for r in store.records] == ["success", "rate_limited"]
    allowed, limited = tool_messages(llm.calls[2]["messages"])
    assert (allowed.status, limited.status) == ("success", "error")
    assert "Rate limit of 1 calls per task exceeded" in limited.content


# --- 6. the turn budget --------------------------------------------------------


async def test_running_out_of_turns_fails_the_attempt(store, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_tool_iterations", 3)
    llm = ScriptedLLM(*(turn(call("echo", msg=f"try {n}")) for n in range(4)))

    with pytest.raises(SpecialistError, match="exceeded 3 tool iterations"):
        await ResearchSpecialist(llm, make_registry(store)).execute(SPEC, {}, None, CTX)
    assert len(llm.calls) == 3


# --- 7. the human-approval gate ---------------------------------------------------


async def test_the_gate_decides_sensitive_calls_before_the_batch_runs(store):
    gate = ScriptedGate(
        store,
        {"action": "reject", "payload": {}, "notes": "no external posts"},
        {"action": "take_over", "payload": {"output": "HUMAN-RESULT"}, "notes": ""},
        {"action": "modify", "payload": {"arguments": {"msg": "edited"}}, "notes": ""},
    )
    llm = ScriptedLLM(
        turn(call("echo", msg="hello"), call("echo", msg="danger")),  # turn 0: rejected
        turn(call("echo", msg="danger")),  # turn 1: taken over
        turn(call("echo", msg="danger")),  # turn 2: modified
        final("DONE"),
    )

    result = await ResearchSpecialist(llm, make_registry(store)).execute(SPEC, {}, None, CTX, gate=gate)

    # only sensitive calls reach the gate, one consultation each, with the turn index
    assert [(c["tool"], c["arguments"], c["iteration"]) for c in gate.calls] == [
        ("echo", {"msg": "danger"}, 0),
        ("echo", {"msg": "danger"}, 1),
        ("echo", {"msg": "danger"}, 2),
    ]
    assert gate.calls[0]["invocations_so_far"] == 0  # decided before any call of its batch ran
    assert all(isinstance(line, str) for c in gate.calls for line in c["transcript"])
    assert gate.calls[-1]["transcript"]  # later decisions see what has happened so far

    # rejected and taken-over calls never reach the registry; the modified one runs as edited
    assert [r.arguments for r in store.records] == [{"msg": "hello"}, {"msg": "edited"}]
    assert result == SpecialistResult(output="DONE", tool_calls=["echo", "echo[human]", "echo"])

    hello, denied, by_hand, edited = tool_messages(llm.calls[3]["messages"])
    assert json.loads(hello.content) == {"msg": "hello"}
    assert (denied.status, denied.content) == ("error", "denied by human reviewer: no external posts")
    assert by_hand.content == "(human-provided) HUMAN-RESULT"
    assert json.loads(edited.content) == {"msg": "edited"}


# --- 8. reviewer feedback -------------------------------------------------------


async def test_reviewer_feedback_is_in_the_human_message_only_when_given(store):
    with_feedback = ScriptedLLM(final("V2"))
    await ResearchSpecialist(with_feedback, make_registry(store)).execute(SPEC, {}, "add citations", CTX)
    human = with_feedback.calls[0]["messages"][1]
    assert FEEDBACK_MARKER in human.text
    assert "add citations" in human.text

    without = ScriptedLLM(final("V1"))
    await ResearchSpecialist(without, make_registry(store)).execute(SPEC, {}, None, CTX)
    assert all(FEEDBACK_MARKER not in message.text for message in without.calls[0]["messages"])


# --- 9. what every provider requires ---------------------------------------------


async def test_every_tool_call_is_answered_right_after_its_turn(store):
    gate = ScriptedGate(
        store,
        {"action": "take_over", "payload": {"output": "BY-HAND"}, "notes": ""},
        {"action": "reject", "payload": {}, "notes": ""},
    )
    llm = ScriptedLLM(
        turn(call("echo", msg="a"), call("boom", msg="b"), call("echo", msg="danger")),
        turn(call("echo", msg="danger"), call("once", msg="c")),
        final("DONE"),
    )

    await ResearchSpecialist(llm, make_registry(store)).execute(SPEC, {}, None, CTX, gate=gate)

    assert len(llm.calls) == 3
    # for every assistant turn with n tool calls, the next n messages answer exactly those ids, in order
    for chat in llm.calls:
        messages = chat["messages"]
        for index, message in enumerate(messages):
            if isinstance(message, AIMessage) and message.tool_calls:
                expected = [c["id"] for c in message.tool_calls]
                answers = messages[index + 1 : index + 1 + len(expected)]
                assert all(isinstance(answer, ToolMessage) for answer in answers), messages
                assert [answer.tool_call_id for answer in answers] == expected
