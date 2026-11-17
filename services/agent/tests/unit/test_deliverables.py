"""Deliverables as a checked postcondition: the verify step after synthesis.

Each file the plan lists as a deliverable must have a successful file_write at exactly
its path in the task's invocation log. A missing one sends its producer back once, with
feedback naming the path, and the outputs are synthesized again; still missing, the task
fails naming the file. The full graph runs on scripted mock fixtures, an in-memory repo
and an in-memory invocation store.
"""

import json

import pytest

from orchestrator.config import get_settings
from orchestrator.db.repo import InMemoryTaskRepo
from orchestrator.graph.builder import build_graph
from orchestrator.graph.edges import check_deliverables, deliverable_feedback, route_after_verify, written_paths
from orchestrator.hitl.queue import InMemoryApprovalQueue
from orchestrator.llm.messages import render_messages
from orchestrator.llm.mock import MockLLMClient
from orchestrator.rules import CURRENT_RULES_VERSION
from orchestrator.tools.base import InMemoryInvocationStore, InvocationRecord
from orchestrator.tools.defaults import build_default_registry

pytestmark = pytest.mark.anyio

FEEDBACK = "Required file not written: `notes.md`"


def fx(directory, name: str, agent: str, text: str = "", match: list[str] | None = None,
       writes: list[str] | None = None) -> None:
    """A fixture; ``writes`` makes it a turn that calls file_write once per path."""
    response: dict = {"text": text}
    if writes:
        response["tool_calls"] = [
            {"name": "file_write", "arguments": {"path": path, "content": f"content of {path}"}} for path in writes
        ]
    payload = {"agent": agent, "response": response}
    if match:
        payload["match"] = match
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def step(sid: str, specialist: str, tools: list[str], description: str, depends_on: list[str] | None = None) -> dict:
    return {"id": sid, "description": description, "specialist": specialist, "tools": tools,
            "depends_on": depends_on or []}


def two_step_plan(directory) -> None:
    """Research, then a writing subtask that is to write notes.md."""
    plan = {
        "task_summary": "notes",
        "subtasks": [step("s1", "research", ["web_search"], "look the topic up"),
                     step("s2", "writing", ["file_write"], "write the notes", ["s1"])],
        "deliverables": [{"path": "notes.md", "producer": "s2"}],
        "confidence": 0.9,
    }
    fx(directory, "plan", "supervisor", json.dumps(plan), match=["Create an execution plan"])
    fx(directory, "synth", "supervisor", "DONE", match=["Synthesize the final deliverable"])
    fx(directory, "research", "research", "FACTS")
    fx(directory, "reviewer", "reviewer", json.dumps({"score": 5, "feedback": "fine"}))


class Recording:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    async def chat(self, agent, messages, **kwargs):
        self.calls.append((agent, render_messages(messages)))
        return await self.inner.chat(agent, messages, **kwargs)

    def prompts(self, agent: str, needle: str = "") -> list[str]:
        return [prompt for who, prompt in self.calls if who == agent and needle in prompt]


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path / "workspaces")
    return tmp_path / "workspaces"


@pytest.fixture()
def fixtures(tmp_path):
    directory = tmp_path / "fixtures"
    directory.mkdir()
    return directory


async def run(fixtures, request: str = "write notes", rules_version: int = CURRENT_RULES_VERSION):
    repo, store, llm = InMemoryTaskRepo(), InMemoryInvocationStore(), Recording(MockLLMClient(fixtures))
    graph = build_graph(llm=llm, registry=build_default_registry(store), repo=repo, checkpointer=None,
                        approvals=InMemoryApprovalQueue())
    task_id = repo.create_task(request, rules_version=rules_version)
    state = await graph.ainvoke({"task_id": task_id, "request": request, "subtask_results": {}, "dispatch_log": [],
                                 "rules_version": rules_version})
    return repo.tasks[task_id], state, llm, store


# --- the graph --------------------------------------------------------------------


async def test_a_missing_file_gets_one_corrective_dispatch_then_the_task_fails_naming_it(fixtures):
    two_step_plan(fixtures)
    fx(fixtures, "writing", "writing", "NOTES WITHOUT A FILE")

    task, state, llm, _ = await run(fixtures)

    assert task["status"] == "failed"
    assert task["error"] == "Required file not written: `notes.md`; s2 was already sent back once to write it"
    assert task["final_output"] is None  # nothing delivered
    assert state["dispatch_log"] == [["s1"], ["s2"], ["s2"]]  # the producer alone, once more
    first, corrective = llm.prompts("writing")
    assert FEEDBACK not in first
    assert f"Reviewer feedback: {FEEDBACK}. Call file_write with exactly that path" in corrective
    assert len(llm.prompts("supervisor", "Synthesize the final deliverable")) == 2  # synthesized again
    assert state["subtask_results"]["s2"]["attempts"] == 2
    assert state["deliverables_redispatched"] == ["s2"]


async def test_a_file_written_under_the_wrong_name_is_corrected(fixtures, workspace):
    two_step_plan(fixtures)
    fx(fixtures, "w_a", "writing", match=["write the notes"], writes=["docs/notes.md"])
    fx(fixtures, "w_b", "writing", "WROTE", match=["write the notes", "-> file_write returned"])
    fx(fixtures, "w_c", "writing", match=["write the notes", FEEDBACK], writes=["notes.md"])
    fx(fixtures, "w_d", "writing", "WROTE AGAIN", match=["write the notes", FEEDBACK, "-> file_write returned"])

    task, state, _, store = await run(fixtures)

    assert task["status"] == "completed"
    assert task["final_output"] == "DONE"
    assert written_paths(store.records) == {"docs/notes.md", "notes.md"}
    assert (workspace / task["task_id"] / "notes.md").is_file()
    assert state["subtask_results"]["s2"]["output"] == "WROTE AGAIN"
    assert state["deliverables_outcome"] == "ok"


async def test_stray_files_beside_the_deliverable_do_not_fail(fixtures):
    two_step_plan(fixtures)
    fx(fixtures, "w_a", "writing", match=["write the notes"], writes=["draft.md", "notes.md", "notes-v2.md"])
    fx(fixtures, "w_b", "writing", "WROTE", match=["write the notes", "-> file_write returned"])

    task, state, _, _ = await run(fixtures)

    assert task["status"] == "completed"
    assert state["dispatch_log"] == [["s1"], ["s2"]]
    assert "deliverables_redispatched" not in state


async def test_a_failed_write_does_not_count(fixtures):
    two_step_plan(fixtures)
    fx(fixtures, "w_a", "writing", match=["write the notes"], writes=["../notes.md"])  # escapes the workspace
    fx(fixtures, "w_b", "writing", "TRIED", match=["write the notes", "-> file_write failed"])

    task, state, _, store = await run(fixtures)

    assert [r.status for r in store.records if r.tool_name == "file_write"] == ["failure", "failure"]
    assert task["status"] == "failed"
    assert task["error"].startswith("Required file not written: `notes.md`")


async def test_a_producer_with_no_attempt_left_is_not_sent_back(fixtures, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_subtask_attempts", 1)
    two_step_plan(fixtures)
    fx(fixtures, "writing", "writing", "NOTES WITHOUT A FILE")

    task, state, llm, _ = await run(fixtures)

    assert task["status"] == "failed"
    assert task["error"] == "Required file not written: `notes.md`; s2 has made all 1 attempts it may"
    assert state["dispatch_log"] == [["s1"], ["s2"]]
    assert len(llm.prompts("writing")) == 1


async def test_a_one_subtask_plan_delivers_the_corrected_output_without_synthesis(fixtures):
    plan = {"task_summary": "notes", "subtasks": [step("s1", "research", ["web_search", "file_write"], "write the notes")],
            "deliverables": [{"path": "notes.md", "producer": "s1"}], "confidence": 0.9}
    fx(fixtures, "plan", "supervisor", json.dumps(plan), match=["Create an execution plan"])
    fx(fixtures, "reviewer", "reviewer", json.dumps({"score": 5, "feedback": "fine"}))
    fx(fixtures, "r_a", "research", "FORGOT THE FILE", match=["write the notes"])
    fx(fixtures, "r_b", "research", match=["write the notes", FEEDBACK], writes=["notes.md"])
    fx(fixtures, "r_c", "research", "NOTES SAVED", match=["write the notes", FEEDBACK, "-> file_write returned"])

    task, state, llm, _ = await run(fixtures)

    assert task["status"] == "completed"
    assert task["final_output"] == "NOTES SAVED"
    assert llm.prompts("supervisor", "Synthesize the final deliverable") == []


async def test_with_the_setting_off_a_missing_file_is_not_checked(fixtures, monkeypatch):
    monkeypatch.setattr(get_settings(), "verify_deliverables", False)
    two_step_plan(fixtures)
    fx(fixtures, "writing", "writing", "NOTES WITHOUT A FILE")

    task, state, _, _ = await run(fixtures)

    assert task["status"] == "completed"
    assert state["dispatch_log"] == [["s1"], ["s2"]]


async def test_a_task_under_rules_without_deliverables_is_not_checked(fixtures):
    two_step_plan(fixtures)
    fx(fixtures, "writing", "writing", "NOTES WITHOUT A FILE")

    task, state, _, _ = await run(fixtures, rules_version=2)

    assert task["status"] == "completed"
    assert state["dispatch_log"] == [["s1"], ["s2"]]


async def test_a_plan_without_deliverables_is_not_checked(fixtures):
    two_step_plan(fixtures)
    plan = json.loads(json.loads((fixtures / "plan.json").read_text())["response"]["text"])
    fx(fixtures, "plan", "supervisor", json.dumps({**plan, "deliverables": []}), match=["Create an execution plan"])
    fx(fixtures, "writing", "writing", "NOTES WITHOUT A FILE")

    task, state, _, _ = await run(fixtures)

    assert task["status"] == "completed"
    assert state["deliverables_outcome"] == "ok"


# --- the pure parts -----------------------------------------------------------------


def record(tool: str, status: str, path: str | None = None) -> InvocationRecord:
    return InvocationRecord(task_id="t", specialist="writing", tool_name=tool, status=status, latency_ms=0.0,
                            arguments={} if path is None else {"path": path, "content": "x"})


def test_only_a_successful_file_write_writes_a_path_exactly_as_given():
    records = [record("file_write", "success", "a.md"), record("file_write", "failure", "b.md"),
               record("file_write", "rate_limited", "c.md"), record("file_read", "success", "d.md"),
               record("file_write", "success", "./e.md"), record("file_write", "success")]

    assert written_paths(records) == {"a.md", "./e.md"}


def state_with(results: dict, sent_back: list[str] | None = None) -> dict:
    plan = {"subtasks": [step("s1", "research", ["file_write"], "a"), step("s2", "writing", ["file_write"], "b")],
            "deliverables": [{"path": "a.md", "producer": "s1"}, {"path": "b.md", "producer": "s2"},
                             {"path": "c.md", "producer": "s2"}],
            "confidence": 0.9}
    return {"plan": plan, "subtask_results": results, "deliverables_redispatched": sent_back or []}


DONE = {"status": "completed", "attempts": 1}


def test_each_producer_is_sent_back_for_its_own_missing_files():
    check = check_deliverables(state_with({"s1": DONE, "s2": DONE}), {"a.md"}, 5)

    assert check.missing == ["b.md", "c.md"]
    assert check.redispatch == {"s2": ["b.md", "c.md"]}
    assert check.failure is None
    assert deliverable_feedback(check.redispatch["s2"]) == (
        "Required files not written: `b.md`, `c.md`. Call file_write with exactly each of those paths "
        "and the complete content, then reply with your deliverable."
    )


def test_nothing_missing_is_nothing_to_do():
    assert check_deliverables(state_with({"s1": DONE, "s2": DONE}), {"a.md", "b.md", "c.md", "x.md"}, 5).missing == []


def test_a_take_over_is_not_sent_back_and_fails_the_task_naming_every_missing_file():
    check = check_deliverables(state_with({"s1": DONE, "s2": {**DONE, "human_provided": True}}), set(), 5)

    assert check.redispatch == {}
    assert check.failure == ("Required files not written: `a.md`, `b.md`, `c.md`; "
                             "s2's output was a human's take-over")


def test_a_producer_sent_back_once_is_not_sent_back_again():
    check = check_deliverables(state_with({"s1": DONE, "s2": DONE}, sent_back=["s1"]), {"b.md", "c.md"}, 5)

    assert check.failure == "Required file not written: `a.md`; s1 was already sent back once to write it"


def test_the_route_after_verify():
    assert route_after_verify({"deliverables_outcome": "ok"}) == "final_gate"
    assert route_after_verify({}) == "final_gate"
    assert route_after_verify({"deliverables_outcome": "redispatch"}) == "schedule"
    assert route_after_verify({"deliverables_outcome": "failed"}) == "end"
