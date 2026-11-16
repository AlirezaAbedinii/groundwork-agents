"""Per-task ceilings on model calls and spend (max_task_llm_calls, max_task_cost_usd).

TracedLLMClient refuses the call that would pass either ceiling with
TaskBudgetExceeded. The graph lets it through as the end of the run rather than a
failed attempt, and the runner records the task as failed with the reason.
"""

import asyncio
import functools
import json
from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.db.repo import InMemoryTaskRepo
from orchestrator.graph import runner
from orchestrator.graph.builder import build_graph
from orchestrator.hitl.queue import InMemoryApprovalQueue
from orchestrator.llm.mock import LLMResponse, MockLLMClient, ToolCall
from orchestrator.llm.pricing import cost_usd
from orchestrator.observability.tracing import TaskBudgetExceeded, TracedLLMClient, task_run_span
from orchestrator.tools.base import InMemoryInvocationStore
from orchestrator.tools.defaults import build_default_registry
from orchestrator.rules import CURRENT_RULES_VERSION

pytestmark = pytest.mark.anyio


class EndlessLLM:
    """A model that never stops: every answer asks for one more tool call."""

    def __init__(self, model: str = "mock", prompt_tokens: int = 0, completion_tokens: int = 0):
        self.model, self.prompt_tokens, self.completion_tokens = model, prompt_tokens, completion_tokens
        self.calls = 0

    async def chat(self, agent, messages, *, tools=None, output_schema=None, producer_provider=None):
        self.calls += 1
        await asyncio.sleep(0)  # let concurrent callers interleave
        return LLMResponse(
            tool_calls=(ToolCall(id=f"call-{self.calls}", name="web_search", arguments={"query": "more"}),),
            model=self.model, prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
        )


class Counting:
    """Passes calls through to a client and counts them."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    async def chat(self, *args, **kwargs):
        self.calls += 1
        return await self.inner.chat(*args, **kwargs)


class CallStore:
    """DBLLMCallStore's record and totals, in memory; two clients may share one, as two workers share the table."""

    def __init__(self):
        self.rows: list[dict] = []

    def record(self, **fields) -> None:
        self.rows.append(fields)

    def totals(self, task_id: str) -> tuple[int, float]:
        rows = [row for row in self.rows if row["task_id"] == task_id]
        return len(rows), sum(row["cost_usd"] for row in rows)


async def calls_until_stopped(client: TracedLLMClient, task_id: str = "t1", limit: int = 1000) -> TaskBudgetExceeded:
    with task_run_span(task_id):
        for _ in range(limit):
            try:
                await client.chat("research", [])
            except TaskBudgetExceeded as error:
                return error
    raise AssertionError(f"no ceiling stopped {limit} calls")


# --- the client -------------------------------------------------------------------


async def test_the_call_ceiling_refuses_the_call_after_the_last_allowed_one():
    endless = EndlessLLM()
    client = TracedLLMClient(endless, max_calls=60, max_cost_usd=0)

    error = await calls_until_stopped(client)

    assert endless.calls == 60  # the refused call never reached the model
    assert (error.task_id, error.calls) == ("t1", 60)
    assert str(error) == "task budget exceeded: 60 calls, $0.0000 (limit 60 calls)"


async def test_the_spend_ceiling_is_passed_by_at_most_one_call():
    endless = EndlessLLM(model="gpt-4o", prompt_tokens=10_000, completion_tokens=1_000)  # $0.035 a call
    per_call = cost_usd("gpt-4o", 10_000, 1_000)
    client = TracedLLMClient(endless, max_calls=0, max_cost_usd=0.10)

    error = await calls_until_stopped(client)

    assert endless.calls == 3  # $0.07 left room for one more; $0.105 did not
    assert error.cost_usd == pytest.approx(3 * per_call)
    assert error.cost_usd < 0.10 + per_call
    assert "(limit $0.10)" in str(error)


async def test_ceilings_of_zero_disable_both():
    endless = EndlessLLM(model="gpt-4o", prompt_tokens=10_000, completion_tokens=1_000)
    client = TracedLLMClient(endless, max_calls=0, max_cost_usd=0)

    with task_run_span("t1"):
        for _ in range(200):
            await client.chat("research", [])

    assert client.usage("t1") == (200, pytest.approx(200 * cost_usd("gpt-4o", 10_000, 1_000)))


async def test_the_settings_are_the_defaults_and_are_read_at_each_call(monkeypatch):
    from orchestrator.config import Settings, get_settings

    defaults = Settings()
    assert (defaults.max_task_llm_calls, defaults.max_task_cost_usd) == (60, 0.50)

    monkeypatch.setattr(get_settings(), "max_task_llm_calls", 3)
    monkeypatch.setattr(get_settings(), "max_task_cost_usd", 0)
    endless = EndlessLLM()

    await calls_until_stopped(TracedLLMClient(endless))

    assert endless.calls == 3


async def test_concurrent_calls_of_one_task_cannot_all_slip_under_the_call_ceiling():
    endless = EndlessLLM()
    client = TracedLLMClient(endless, max_calls=3, max_cost_usd=0)

    with task_run_span("t1"):
        results = await asyncio.gather(*(client.chat("research", []) for _ in range(8)), return_exceptions=True)

    assert endless.calls == 3
    assert sum(isinstance(result, TaskBudgetExceeded) for result in results) == 5


async def test_totals_are_seeded_from_the_store_so_a_resume_or_another_worker_counts_them():
    store = CallStore()
    first = TracedLLMClient(EndlessLLM(), calls=store, max_calls=10, max_cost_usd=0)
    with task_run_span("t1"):
        for _ in range(7):
            await first.chat("research", [])
    first.forget("t1")  # the run paused; the runner drops the task's totals
    assert first.usage("t1") is None

    endless = EndlessLLM()
    second = TracedLLMClient(endless, calls=store, max_calls=10, max_cost_usd=0)  # another worker resumes it
    error = await calls_until_stopped(second)

    assert endless.calls == 3
    assert error.calls == 10
    assert len(store.rows) == 10


async def test_tasks_have_separate_totals_and_calls_outside_a_task_are_not_limited():
    client = TracedLLMClient(EndlessLLM(), max_calls=2, max_cost_usd=0)

    await calls_until_stopped(client, "t1")
    await calls_until_stopped(client, "t2")
    for _ in range(5):  # memory consolidation runs with no task
        await client.chat("memory", [])

    assert client.usage("t1") == (2, 0) and client.usage("t2") == (2, 0)


def test_the_exception_survives_pickling():
    import pickle

    error = pickle.loads(pickle.dumps(TaskBudgetExceeded("t1", 60, 0.25, 60, 0.5)))

    assert (error.task_id, error.calls, error.cost_usd) == ("t1", 60, 0.25)
    assert str(error) == "task budget exceeded: 60 calls, $0.2500 (limit 60 calls, $0.50)"


# --- the graph and the runner ---------------------------------------------------------


def fx(directory, name: str, agent: str, text: str, match=None, tool_calls=None) -> None:
    payload = {"agent": agent, "response": {"text": text}}
    if tool_calls:
        payload["response"]["tool_calls"] = tool_calls
    if match:
        payload["match"] = match
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def one_subtask_plan(tmp_path, *, endless: bool) -> None:
    """One research subtask; with ``endless`` the specialist calls a missing tool on every turn."""
    plan = {
        "task_summary": "test", "confidence": 0.9,
        "subtasks": [{
            "id": "s1", "description": "look it up", "specialist": "research", "tools": ["web_search"],
            "required_inputs": [],
            "expected_output_format": "plain text", "estimated_complexity": "low", "depends_on": [],
        }],
    }
    fx(tmp_path, "plan", "supervisor", json.dumps(plan), match=["Create an execution plan"])
    fx(tmp_path, "synth", "supervisor", "DONE", match=["Synthesize the final deliverable"])
    if endless:
        fx(tmp_path, "research", "research", "", tool_calls=[{"name": "bogus_tool", "arguments": {}}])
    else:
        fx(tmp_path, "research", "research", "RESEARCH-OK")
    fx(tmp_path, "reviewer", "reviewer", json.dumps({"score": 5, "feedback": ""}))


def graph_for(llm, repo, queue, checkpointer=None):
    return build_graph(
        llm=llm,
        registry=build_default_registry(InMemoryInvocationStore()),
        repo=repo,
        checkpointer=checkpointer if checkpointer is not None else MemorySaver(),
        approvals=queue,
    )


def initial(task_id: str, request: str) -> dict:
    return {"task_id": task_id, "request": request, "subtask_results": {}, "dispatch_log": [],
            "rules_version": CURRENT_RULES_VERSION}


async def test_a_specialist_that_never_stops_ends_the_task_at_the_ceiling_not_the_attempt_cap(tmp_path, monkeypatch):
    from orchestrator.config import get_settings

    # unattended: every subtask escalation only notifies, so the attempt cap alone
    # would allow 5 attempts of 5 turns each (26 calls with the plan)
    monkeypatch.setitem(get_settings().approval_level_overrides, "specialist_double_failure", "notify")
    one_subtask_plan(tmp_path, endless=True)
    model = Counting(MockLLMClient(tmp_path))
    llm = TracedLLMClient(model, max_calls=12, max_cost_usd=0)
    repo, queue = InMemoryTaskRepo(), InMemoryApprovalQueue()
    graph = graph_for(llm, repo, queue)
    task_id = repo.create_task("nobody watching")

    with pytest.raises(TaskBudgetExceeded) as raised:
        with task_run_span(task_id):
            await graph.ainvoke(initial(task_id, "nobody watching"), {"configurable": {"thread_id": task_id}})

    assert model.calls == 12
    assert raised.value.calls == 12
    # the refused call was not taken for a failed attempt: no retry, no escalation for it
    subtask = repo.subtasks[task_id]["s1"]
    assert (subtask["status"], subtask["attempts"], subtask["error"]) == ("failed", 3, "task budget exceeded")
    assert [row["gate_key"] for row in queue.list(status="notified")] == ["subtask:s1:a2"]


async def test_a_normal_run_is_unchanged_under_the_default_ceilings(tmp_path):
    one_subtask_plan(tmp_path, endless=False)
    llm = TracedLLMClient(MockLLMClient(tmp_path))
    repo, queue = InMemoryTaskRepo(), InMemoryApprovalQueue()
    graph = graph_for(llm, repo, queue)
    task_id = repo.create_task("look it up")

    with task_run_span(task_id):
        state = await graph.ainvoke(initial(task_id, "look it up"), {"configurable": {"thread_id": task_id}})

    assert state["final_output"] == "RESEARCH-OK"  # one subtask: no synthesis call
    assert repo.tasks[task_id]["status"] == "completed"
    assert llm.usage(task_id) == (3, 0)  # plan, research, review


@pytest.fixture()
def production(monkeypatch, tmp_path):
    """The runner with its graph and repo swapped for in-memory ones that share one call store."""
    repo, queue, store, checkpointer = InMemoryTaskRepo(), InMemoryApprovalQueue(), CallStore(), MemorySaver()
    model = Counting(MockLLMClient(tmp_path))
    llm = TracedLLMClient(model, calls=store, max_calls=0, max_cost_usd=0)

    @asynccontextmanager
    async def production_graph(*, discover: bool = True):
        yield graph_for(llm, repo, queue, checkpointer)

    deps = functools.lru_cache(lambda: {"llm": llm})
    deps()
    monkeypatch.setattr(runner, "production_graph", production_graph)
    monkeypatch.setattr(runner, "_dependencies", deps)
    monkeypatch.setattr(runner, "DBTaskRepo", lambda: repo)
    return repo, queue, store, model, llm


async def test_the_runner_fails_the_task_with_the_reason_and_a_resume_counts_earlier_calls(tmp_path, production):
    repo, queue, store, model, llm = production
    one_subtask_plan(tmp_path, endless=True)
    task_id = repo.create_task("look it up")

    await runner.run_task(task_id)

    # two attempts of five turns each, then a pause for a human
    assert (model.calls, len(store.rows)) == (11, 11)
    assert repo.tasks[task_id]["status"] == "awaiting_approval"
    assert llm.usage(task_id) is None  # dropped when the run ended

    llm._max_calls = 13  # a resume may make two more calls in all
    (row,) = queue.list(status="pending")
    queue.resolve(row["id"], action="approve")
    await runner.resume_task(task_id, {"action": "approve", "payload": {}, "notes": ""})

    assert model.calls == 13
    assert repo.tasks[task_id]["status"] == "failed"
    assert repo.tasks[task_id]["error"] == "task budget exceeded: 13 calls, $0.0000 (limit 13 calls)"
    assert llm.usage(task_id) is None
    assert queue.list(status="pending") == []
