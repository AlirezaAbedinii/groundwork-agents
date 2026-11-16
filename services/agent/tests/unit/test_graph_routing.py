"""Graph conditional-edge tests: run the full LangGraph with scripted mock
fixtures, an in-memory repo, queue, and invocation store (no Postgres).
Escalation tests use MemorySaver so interrupts can pause and resume. The graph
has async nodes, so every run goes through ainvoke."""

import json

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from orchestrator.db.repo import InMemoryTaskRepo
from orchestrator.graph.builder import build_graph
from orchestrator.hitl.queue import InMemoryApprovalQueue
from orchestrator.llm.mock import MockLLMClient
from orchestrator.tools.base import InMemoryInvocationStore
from orchestrator.tools.defaults import build_default_registry

pytestmark = pytest.mark.anyio


def fx(
    directory, name: str, agent: str, text: str,
    match: list[str] | None = None, tool_calls: list[dict] | None = None,
) -> None:
    payload = {"agent": agent, "response": {"text": text}}
    if tool_calls:
        payload["response"]["tool_calls"] = tool_calls
    if match:
        payload["match"] = match
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def plan_text(subtasks: list[dict], confidence: float) -> str:
    return json.dumps({"task_summary": "test", "subtasks": subtasks, "confidence": confidence})


def subtask(sid: str, specialist: str, description: str, depends_on: list[str] | None = None) -> dict:
    return {
        "id": sid,
        "description": description,
        "specialist": specialist,
        "required_inputs": [],
        "expected_output_format": "plain text",
        "estimated_complexity": "low",
        "depends_on": depends_on or [],
    }


def tool_call(tool: str, arguments: dict) -> dict:
    """A native tool call for fx(..., tool_calls=[...])."""
    return {"name": tool, "arguments": arguments}


def verdict(score: int, feedback: str = "") -> str:
    return json.dumps({"score": score, "feedback": feedback})


@pytest.fixture()
def repo():
    return InMemoryTaskRepo()


async def run(tmp_path, repo, request_text: str, store: InMemoryInvocationStore | None = None):
    graph = build_graph(
        llm=MockLLMClient(tmp_path),
        registry=build_default_registry(store if store is not None else InMemoryInvocationStore()),
        repo=repo,
        checkpointer=None,
        approvals=InMemoryApprovalQueue(),  # an unexpected escalation must not reach Postgres
    )
    task_id = repo.create_task(request_text)
    return task_id, await graph.ainvoke(
        {"task_id": task_id, "request": request_text, "subtask_results": {}, "dispatch_log": []}
    )


def build(tmp_path, repo, queue, checkpointer):
    return build_graph(
        llm=MockLLMClient(tmp_path),
        registry=build_default_registry(InMemoryInvocationStore()),
        repo=repo,
        checkpointer=checkpointer,
        approvals=queue,
    )


async def run_until_pause(tmp_path, repo, request_text: str, queue, checkpointer=None):
    """Build an interrupt-capable graph (MemorySaver) and run to the pause."""
    graph = build(tmp_path, repo, queue, checkpointer or MemorySaver())
    task_id = repo.create_task(request_text)
    config = {"configurable": {"thread_id": task_id}}
    state = await graph.ainvoke(
        {"task_id": task_id, "request": request_text, "subtask_results": {}, "dispatch_log": []},
        config,
    )
    return task_id, state, graph, config


async def test_happy_path_with_parallel_wave(tmp_path, repo):
    fx(tmp_path, "plan", "supervisor",
       plan_text([subtask("s1", "research", "look up A"),
                  subtask("s2", "research", "look up B"),
                  subtask("s3", "writing", "write summary", depends_on=["s1", "s2"])], 0.9),
       match=["Create an execution plan"])
    fx(tmp_path, "synth", "supervisor", "DONE", match=["Synthesize the final deliverable"])
    fx(tmp_path, "research", "research", "RESEARCH-OK")
    fx(tmp_path, "writing", "writing", "SUMMARY-OK")
    fx(tmp_path, "reviewer", "reviewer", verdict(5, "solid"))

    task_id, state = await run(tmp_path, repo, "compare A and B")

    assert state["final_output"] == "DONE"
    assert state["dispatch_log"][0] == ["s1", "s2"]  # independent subtasks batched together
    assert state["dispatch_log"][1] == ["s3"]
    assert all(r["status"] == "completed" for r in state["subtask_results"].values())
    assert repo.tasks[task_id]["status"] == "completed"
    assert repo.tasks[task_id]["final_output"] == "DONE"


async def test_every_specialist_sees_the_request_the_plan_paraphrased(tmp_path, repo):
    # the plan drops the inputs the request names; each specialist's fixture answers only
    # when its prompt carries them, and the agent defaults answer when it does not
    request = "Write parse_duration and run it on 1h30m15s, 45s, 2h5m and 90m; report the seconds."
    fx(tmp_path, "plan", "supervisor",
       plan_text([subtask("s1", "code", "Run the function on the provided duration strings"),
                  subtask("s2", "writing", "Report the results", depends_on=["s1"])], 0.9),
       match=["Create an execution plan"])
    fx(tmp_path, "synth", "supervisor", "DONE", match=["Synthesize the final deliverable"])
    fx(tmp_path, "code", "code", "NO-REQUEST")
    fx(tmp_path, "code_with_request", "code", "5415, 45, 7500, 5400",
       match=["Run the function on the provided duration strings", "1h30m15s"])
    fx(tmp_path, "writing", "writing", "NO-REQUEST")
    fx(tmp_path, "writing_with_request", "writing", "REPORT-OK", match=["Report the results", "2h5m"])
    fx(tmp_path, "reviewer", "reviewer", verdict(5, "solid"))

    _, state = await run(tmp_path, repo, request)

    results = state["subtask_results"]
    assert (results["s1"]["output"], results["s2"]["output"]) == ("5415, 45, 7500, 5400", "REPORT-OK")


def test_each_payload_of_a_wave_carries_the_request():
    from orchestrator.graph.edges import compute_wave

    state = {
        "task_id": "t",
        "request": "compare A and B",
        "plan": json.loads(plan_text([subtask("s1", "research", "look up A"),
                                      subtask("s2", "research", "look up B")], 0.9)),
        "subtask_results": {},
    }
    assert [payload["request"] for payload in compute_wave(state)] == ["compare A and B"] * 2


async def test_two_tool_calls_in_one_turn_run_as_one_batch(tmp_path, repo, monkeypatch):
    from orchestrator.config import get_settings

    # the batch plus the final answer fits two turns only if a batch is one turn
    monkeypatch.setattr(get_settings(), "max_tool_iterations", 2)
    fx(tmp_path, "plan", "supervisor",
       plan_text([subtask("s1", "research", "Compare Chroma and Qdrant")], 0.9),
       match=["Create an execution plan"])
    fx(tmp_path, "synth", "supervisor", "DONE", match=["Synthesize the final deliverable"])
    fx(tmp_path, "r_batch", "research", "", match=["Compare Chroma and Qdrant"],
       tool_calls=[{"name": "web_search", "arguments": {"query": "Chroma vector database"}},
                   {"name": "web_search", "arguments": {"query": "Qdrant vector database"}}])
    # the next turn answers only once both results are in the conversation
    fx(tmp_path, "r_done", "research", "BOTH-FOUND",
       match=["Compare Chroma and Qdrant", "-> web_search returned:",
              "[mock] result for Chroma vector database", "[mock] result for Qdrant vector database"])
    fx(tmp_path, "reviewer", "reviewer", verdict(5))

    store = InMemoryInvocationStore()
    task_id, state = await run(tmp_path, repo, "compare Chroma and Qdrant", store=store)

    result = state["subtask_results"]["s1"]
    assert (result["status"], result["output"], result["attempts"]) == ("completed", "BOTH-FOUND", 1)
    assert result["tool_calls"] == ["web_search", "web_search"]
    searches = [record for record in store.records if record.tool_name == "web_search"]
    assert [record.status for record in searches] == ["success", "success"]
    assert sorted(record.arguments["query"] for record in searches) == [
        "Chroma vector database", "Qdrant vector database",
    ]
    assert state["final_output"] == "DONE"


async def test_two_sensitive_calls_in_one_turn_get_one_approval_each(tmp_path, repo):
    posts = [tool_call("api_call", {"method": "POST", "url": f"https://api.github.com/announce/{n}"}) for n in "ab"]
    fx(tmp_path, "plan", "supervisor",
       plan_text([subtask("s1", "research", "Post both announcements")], 0.9),
       match=["Create an execution plan"])
    fx(tmp_path, "synth", "supervisor", "DONE", match=["Synthesize the final deliverable"])
    fx(tmp_path, "r_posts", "research", "", match=["Post both announcements"], tool_calls=posts)
    fx(tmp_path, "r_done", "research", "BOTH-POSTED",
       match=["Post both announcements", "[mock] response from https://api.github.com/announce/a",
              "[mock] response from https://api.github.com/announce/b"])
    fx(tmp_path, "reviewer", "reviewer", verdict(5))

    queue = InMemoryApprovalQueue()
    task_id, state, graph, config = await run_until_pause(tmp_path, repo, "announce twice", queue)

    # the first POST waits for its own approval ...
    (first,) = queue.list(status="pending")
    queue.resolve(first["id"], action="approve")
    await graph.ainvoke(Command(resume={"action": "approve", "payload": {}, "notes": ""}), config)

    # ... and the second for another one, rather than re-pausing on the resolved first
    (second,) = queue.list(status="pending")
    assert second["gate_key"] != first["gate_key"]
    assert [a["proposed_action"]["arguments"]["url"] for a in (first, second)] == [
        "https://api.github.com/announce/a", "https://api.github.com/announce/b",
    ]
    queue.resolve(second["id"], action="approve")
    finished = await graph.ainvoke(Command(resume={"action": "approve", "payload": {}, "notes": ""}), config)

    assert finished["subtask_results"]["s1"]["output"] == "BOTH-POSTED"
    assert finished["final_output"] == "DONE"
    assert repo.tasks[task_id]["status"] == "completed"


async def test_reviewer_rejection_routes_back_with_feedback_then_succeeds(tmp_path, repo):
    fx(tmp_path, "plan", "supervisor",
       plan_text([subtask("s1", "writing", "draft the memo")], 0.9),
       match=["Create an execution plan"])
    fx(tmp_path, "synth", "supervisor", "DONE", match=["Synthesize the final deliverable"])
    # attempt 2 carries the reviewer feedback block, so its fixture has one more needle and wins
    fx(tmp_path, "a_writing_feedback", "writing", "DRAFT-V2", match=["draft the memo", "Reviewer feedback:"])
    fx(tmp_path, "b_writing_first", "writing", "DRAFT-V1", match=["draft the memo"])
    fx(tmp_path, "r_v1", "reviewer", verdict(2, "add citations"), match=["DRAFT-V1"])
    fx(tmp_path, "r_v2", "reviewer", verdict(5, "good"), match=["DRAFT-V2"])

    task_id, state = await run(tmp_path, repo, "write a memo")

    result = state["subtask_results"]["s1"]
    assert result["status"] == "completed"
    assert result["output"] == "DRAFT-V2"
    assert result["attempts"] == 2
    assert result["rework_count"] == 1
    assert repo.subtasks[task_id]["s1"]["review_score"] == 5
    assert repo.tasks[task_id]["status"] == "completed"


async def test_two_specialist_failures_pause_then_human_takes_over(tmp_path, repo):
    fx(tmp_path, "plan", "supervisor",
       plan_text([subtask("s1", "research", "look something up")], 0.9),
       match=["Create an execution plan"])
    fx(tmp_path, "synth", "supervisor", "DONE", match=["Synthesize the final deliverable"])
    # the specialist insists on a tool that does not exist -> attempt errors
    fx(tmp_path, "research", "research", "", tool_calls=[tool_call("bogus_tool", {})])
    fx(tmp_path, "reviewer", "reviewer", verdict(5))

    queue = InMemoryApprovalQueue()
    task_id, state, graph, config = await run_until_pause(tmp_path, repo, "doomed task", queue)

    interrupt_payload = state["__interrupt__"][0].value
    assert interrupt_payload["trigger"] == "specialist_double_failure"
    assert state["subtask_results"]["s1"]["error_count"] == 2
    (row,) = queue.list(status="pending")
    assert row["level"] == "approve_action" and row["gate_key"].startswith("subtask:s1")
    assert repo.tasks[task_id]["status"] == "awaiting_approval"

    finished = await graph.ainvoke(
        Command(resume={"action": "take_over", "payload": {"output": "HUMAN-OUT"},
                        "notes": "did it myself"}),
        config,
    )
    assert finished["subtask_results"]["s1"]["status"] == "completed"
    assert finished["subtask_results"]["s1"]["output"] == "HUMAN-OUT"
    assert finished["subtask_results"]["s1"]["human_provided"] is True
    assert finished["final_output"] == "DONE"
    assert repo.tasks[task_id]["status"] == "completed"
    assert repo.subtasks[task_id]["s1"]["output"] == "HUMAN-OUT"


async def test_a_retry_after_running_out_of_turns_is_told_what_happened(tmp_path, repo):
    fx(tmp_path, "plan", "supervisor",
       plan_text([subtask("s1", "research", "look something up")], 0.9),
       match=["Create an execution plan"])
    fx(tmp_path, "research", "research", "", tool_calls=[tool_call("bogus_tool", {})])

    _, state, _, _ = await run_until_pause(tmp_path, repo, "doomed task", InMemoryApprovalQueue())

    feedback = state["subtask_results"]["s1"]["feedback"]
    assert feedback.startswith("Previous attempt ran out of turns. What happened: -> bogus_tool failed:")
    assert "Unknown tool 'bogus_tool'" in feedback.splitlines()[-1]  # the last exchange's error
    assert "Try a different approach" not in feedback


async def test_low_plan_confidence_pauses_then_approve_resumes(tmp_path, repo):
    fx(tmp_path, "plan", "supervisor",
       plan_text([subtask("s1", "research", "vague thing")], 0.3),
       match=["Create an execution plan"])
    fx(tmp_path, "synth", "supervisor", "DONE", match=["Synthesize the final deliverable"])
    fx(tmp_path, "research", "research", "RESEARCH-OK")
    fx(tmp_path, "reviewer", "reviewer", verdict(5))

    queue = InMemoryApprovalQueue()
    task_id, state, graph, config = await run_until_pause(tmp_path, repo, "do something vague", queue)

    interrupt_payload = state["__interrupt__"][0].value
    assert interrupt_payload["trigger"] == "low_plan_confidence"
    assert interrupt_payload["level"] == "approve_plan"
    assert interrupt_payload["proposed_action"]["type"] == "execute_plan"
    (row,) = queue.list(status="pending")
    assert row["gate_key"] == "plan"
    assert repo.tasks[task_id]["status"] == "awaiting_approval"
    assert not state.get("subtask_results")  # nothing executed while pending

    finished = await graph.ainvoke(Command(resume={"action": "approve", "payload": {}, "notes": "go"}), config)
    assert finished["final_output"] == "DONE"
    assert repo.tasks[task_id]["status"] == "completed"


async def test_rework_exhaustion_pauses_then_reject_terminates(tmp_path, repo):
    fx(tmp_path, "plan", "supervisor",
       plan_text([subtask("s1", "writing", "draft the memo")], 0.9),
       match=["Create an execution plan"])
    fx(tmp_path, "writing", "writing", "DRAFT-V1")
    fx(tmp_path, "reviewer", "reviewer", verdict(2, "still weak"))

    queue = InMemoryApprovalQueue()
    task_id, state, graph, config = await run_until_pause(tmp_path, repo, "write a memo", queue)

    interrupt_payload = state["__interrupt__"][0].value
    assert interrupt_payload["trigger"] == "low_review_score"
    assert state["subtask_results"]["s1"]["rework_count"] == 2

    finished = await graph.ainvoke(
        Command(resume={"action": "reject", "payload": {}, "notes": "not salvageable"}), config
    )
    assert repo.tasks[task_id]["status"] == "rejected"
    assert "not salvageable" in repo.tasks[task_id]["error"]
    assert not finished.get("final_output")


async def test_notify_level_records_but_never_pauses(tmp_path, repo, monkeypatch):
    from orchestrator.config import get_settings

    monkeypatch.setitem(get_settings().approval_level_overrides, "low_plan_confidence", "notify")
    fx(tmp_path, "plan", "supervisor",
       plan_text([subtask("s1", "research", "vague thing")], 0.3),
       match=["Create an execution plan"])
    fx(tmp_path, "synth", "supervisor", "DONE", match=["Synthesize the final deliverable"])
    fx(tmp_path, "research", "research", "RESEARCH-OK")
    fx(tmp_path, "reviewer", "reviewer", verdict(5))

    queue = InMemoryApprovalQueue()
    task_id, state, graph, config = await run_until_pause(tmp_path, repo, "do something vague", queue)

    assert "__interrupt__" not in state  # proceeded without pausing
    assert state["final_output"] == "DONE"
    assert queue.list(status="pending") == []
    (row,) = queue.list(status="notified")
    assert row["trigger"] == "low_plan_confidence"
    assert repo.tasks[task_id]["status"] == "completed"


async def test_invalid_plan_is_retried_once_with_error_then_fails(tmp_path, repo):
    # First planning call returns a cyclic plan; the retry prompt (which embeds
    # the validation error) returns garbage -> task fails after one retry.
    fx(tmp_path, "a_retry", "supervisor", "still not json",
       match=["Your previous plan was invalid"])
    fx(tmp_path, "b_first", "supervisor",
       plan_text([subtask("s1", "research", "a", depends_on=["s2"]),
                  subtask("s2", "research", "b", depends_on=["s1"])], 0.9),
       match=["Create an execution plan"])

    task_id, state = await run(tmp_path, repo, "impossible plan")

    assert state.get("plan") is None
    assert "plan invalid after retry" in state["plan_error"].lower()
    assert repo.tasks[task_id]["status"] == "failed"


# --- attempt allowance ---------------------------------------------------------


def doomed_plan(tmp_path, sids: tuple[str, ...] = ("s1",)) -> None:
    """Every research attempt calls a tool that does not exist until its turns run out."""
    fx(tmp_path, "plan", "supervisor",
       plan_text([subtask(sid, "research", f"look up {sid}") for sid in sids], 0.9),
       match=["Create an execution plan"])
    fx(tmp_path, "synth", "supervisor", "DONE", match=["Synthesize the final deliverable"])
    fx(tmp_path, "research", "research", "", tool_calls=[tool_call("bogus_tool", {})])
    fx(tmp_path, "reviewer", "reviewer", verdict(5))


async def decide(graph, config, queue, action: str = "approve", payload: dict | None = None, notes: str = ""):
    """Resolve the one pending approval and resume the run with that decision."""
    (row,) = queue.list(status="pending")
    queue.resolve(row["id"], action=action, payload=payload, notes=notes)
    return await graph.ainvoke(
        Command(resume={"action": action, "payload": payload or {}, "notes": notes}), config
    )


def paused_at(state) -> dict:
    """The step of the escalation the run is paused at."""
    return state["__interrupt__"][0].value["current_step"]


def counters(state, sid: str = "s1") -> tuple:
    result = state["subtask_results"][sid]
    return result["attempts"], result["error_count"], result["attempt_allowance"]


async def test_an_approval_grants_exactly_one_more_attempt(tmp_path, repo):
    doomed_plan(tmp_path)
    queue = InMemoryApprovalQueue()
    task_id, state, graph, config = await run_until_pause(tmp_path, repo, "doomed task", queue)
    assert counters(state) == (2, 2, 3)  # attempts, error_count, attempt_allowance

    # the counters are kept, so the one attempt the approval bought escalates again at once
    state = await decide(graph, config, queue)
    assert counters(state) == (3, 3, 3)
    step = paused_at(state)
    assert (step["sid"], step["attempts"]) == ("s1", 3)
    (row,) = queue.list(status="pending")
    assert row["gate_key"] == "subtask:s1:a3"

    state = await decide(graph, config, queue)
    assert counters(state) == (4, 4, 4)
    assert repo.subtasks[task_id]["s1"]["attempts"] == 4


async def test_the_allowance_escalates_an_attempt_that_no_counter_catches(tmp_path, repo):
    fx(tmp_path, "plan", "supervisor",
       plan_text([subtask("s1", "writing", "draft the memo")], 0.9),
       match=["Create an execution plan"])
    # two weak drafts exhaust the rework cycles; the approved third attempt runs out of turns
    fx(tmp_path, "a_writing_retry", "writing", "", match=["draft the memo", "UNLUCKY-RETRY"],
       tool_calls=[tool_call("bogus_tool", {})])
    fx(tmp_path, "b_writing", "writing", "DRAFT-V1", match=["draft the memo"])
    fx(tmp_path, "reviewer", "reviewer", verdict(2, "still weak"))

    queue = InMemoryApprovalQueue()
    task_id, state, graph, config = await run_until_pause(tmp_path, repo, "write a memo", queue)
    assert state["__interrupt__"][0].value["trigger"] == "low_review_score"

    state = await decide(graph, config, queue, notes="UNLUCKY-RETRY")
    result = state["subtask_results"]["s1"]
    # one error is below the double-failure limit, but the attempt was the last one allowed
    assert (result["status"], result["attempts"], result["error_count"], result["rework_count"]) == (
        "failed_attempt", 3, 1, 2,
    )
    escalation = state["__interrupt__"][0].value
    assert escalation["trigger"] == "specialist_double_failure"
    assert "attempt 3, the last one it was allowed" in escalation["reasoning"]


async def test_a_subtask_waiting_for_its_escalation_does_not_run_meanwhile(tmp_path, repo):
    doomed_plan(tmp_path, sids=("s1", "s2"))
    queue = InMemoryApprovalQueue()
    task_id, state, graph, config = await run_until_pause(tmp_path, repo, "doomed pair", queue)
    # both failed twice in the same waves; gather takes s1 first
    assert paused_at(state)["sid"] == "s1"
    assert [counters(state, sid)[0] for sid in ("s1", "s2")] == [2, 2]

    # s1's approval re-runs s1; s2 still has one attempt of its allowance and uses it
    state = await decide(graph, config, queue)
    assert paused_at(state)["sid"] == "s1"
    assert [counters(state, sid)[0] for sid in ("s1", "s2")] == [3, 3]

    # now s2 has made every attempt it was allowed: it waits while s1 runs again
    state = await decide(graph, config, queue)
    assert paused_at(state)["sid"] == "s1"
    assert [counters(state, sid)[0] for sid in ("s1", "s2")] == [4, 3]

    # with s1 taken over nothing can run, so s2's escalation comes next, with no attempt between
    state = await decide(graph, config, queue, action="take_over", payload={"output": "HUMAN-OUT"})
    assert paused_at(state)["sid"] == "s2"
    assert counters(state, "s2")[0] == 3
    assert state["dispatch_log"] == [["s1", "s2"]] * 3 + [["s1"]]


async def test_a_resume_after_a_restart_keeps_the_allowance(tmp_path, repo):
    doomed_plan(tmp_path)
    queue = InMemoryApprovalQueue()
    checkpointer = MemorySaver()
    task_id, state, graph, config = await run_until_pause(
        tmp_path, repo, "doomed task", queue, checkpointer=checkpointer
    )
    state = await decide(graph, config, queue)
    assert counters(state) == (3, 3, 3)

    # a new graph over the same checkpoints, as after a process restart
    restarted = build(tmp_path, repo, queue, checkpointer)
    snapshot = await restarted.aget_state(config)
    assert counters(snapshot.values) == (3, 3, 3)
    state = await decide(restarted, config, queue)
    assert counters(state) == (4, 4, 4)
    # and the cap still holds on the restarted graph
    state = await decide(restarted, config, queue)
    state = await decide(restarted, config, queue)
    assert counters(state)[0] == 5
    assert repo.tasks[task_id]["status"] == "failed"


# --- attempt cap ---------------------------------------------------------------

BUDGET_ERROR = "Attempt budget exhausted for subtask s1 after 5 attempts"


@pytest.mark.parametrize(
    ("action", "payload", "status"),
    [
        ("approve", None, "failed"),
        ("modify", {"feedback": "one more go"}, "failed"),
        ("take_over", {"output": "HUMAN-OUT"}, "completed"),
        ("reject", None, "rejected"),
    ],
)
async def test_decisions_at_the_attempt_cap(tmp_path, repo, action, payload, status):
    doomed_plan(tmp_path)
    queue = InMemoryApprovalQueue()
    task_id, state, graph, config = await run_until_pause(tmp_path, repo, "doomed task", queue)
    approvals = 0
    while paused_at(state)["attempts"] < 5:
        state = await decide(graph, config, queue)
        approvals += 1
    assert approvals == 3  # at attempts 2, 3 and 4, one more attempt each

    escalation = state["__interrupt__"][0].value
    assert escalation["proposed_action"]["type"] == "fail_task"
    assert escalation["current_step"]["max_attempts"] == 5
    assert "approving or modifying ends the task as failed" in escalation["reasoning"]

    state = await decide(graph, config, queue, action=action, payload=payload)
    assert "__interrupt__" not in state
    assert repo.tasks[task_id]["status"] == status
    assert state["subtask_results"]["s1"]["attempts"] == 5  # never a sixth
    assert repo.subtasks[task_id]["s1"]["attempts"] == 5
    if status == "failed":
        assert repo.tasks[task_id]["error"] == BUDGET_ERROR
        assert state["hitl_decision"] == {"gate": "subtask", "action": "fail", "requested": action, "sid": "s1"}
        assert not state.get("final_output")
    if status == "completed":
        assert state["subtask_results"]["s1"]["output"] == "HUMAN-OUT"
        assert state["final_output"] == "DONE"


@pytest.mark.parametrize("trigger", ["low_review_score", "specialist_double_failure"])
async def test_a_notify_override_ends_at_the_attempt_cap(tmp_path, repo, monkeypatch, trigger):
    from orchestrator.config import get_settings

    # with no human in the loop, every escalation used to grant a fresh pair of attempts
    monkeypatch.setitem(get_settings().approval_level_overrides, trigger, "notify")
    if trigger == "low_review_score":
        fx(tmp_path, "plan", "supervisor",
           plan_text([subtask("s1", "writing", "draft the memo")], 0.9),
           match=["Create an execution plan"])
        fx(tmp_path, "writing", "writing", "DRAFT-V1")
        fx(tmp_path, "reviewer", "reviewer", verdict(2, "still weak"))
    else:
        doomed_plan(tmp_path)

    queue = InMemoryApprovalQueue()
    task_id, state, graph, config = await run_until_pause(tmp_path, repo, "nobody watching", queue)

    assert "__interrupt__" not in state  # nobody was asked
    assert state["subtask_results"]["s1"]["attempts"] == 5
    notified = queue.list(status="notified")
    assert [row["gate_key"] for row in notified] == [f"subtask:s1:a{n}" for n in (2, 3, 4, 5)]
    assert {row["trigger"] for row in notified} == {trigger}
    assert notified[-1]["proposed_action"]["type"] == "fail_task"
    assert repo.tasks[task_id]["status"] == "failed"
    assert repo.tasks[task_id]["error"] == BUDGET_ERROR
    assert state["hitl_decision"]["requested"] == "notified"


def test_an_entry_from_before_the_allowance_is_held_and_escalated():
    from orchestrator.graph.edges import compute_wave, dispatch

    # a checkpoint written before attempt_allowance existed: three attempts, no allowance
    state = {
        "task_id": "t",
        "plan": json.loads(plan_text([subtask("s1", "research", "look up s1")], 0.9)),
        "subtask_results": {"s1": {"sid": "s1", "status": "rework", "attempts": 3, "rework_count": 0}},
    }
    assert compute_wave(state) == []  # the default allowance, 3, is spent
    assert dispatch({**state, "current_wave": []}) == "gather"
    state["subtask_results"]["s1"]["attempts"] = 2
    (payload,) = compute_wave(state)
    assert payload["prior"]["attempt_allowance"] is None  # execute fills in the default
