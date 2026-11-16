"""Replay acceptance: a strict replay reproduces the original step-for-step
with zero API calls; a fork at step k replays the prefix, applies the human
modification, diverges afterwards, and the comparison shows exactly that."""

import json
from datetime import datetime, timedelta, timezone

import sqlalchemy as sa

from orchestrator.db.repo import DBLLMCallStore, DBTaskRepo
from orchestrator.db.session import get_engine

REQUEST = (
    "Compare open-source vector databases: gather facts about Chroma from the web, "
    "compute the GitHub star ranking from the demo database, generate a comparison "
    "table using Python, and write a comparison memo saved as memo.md."
)


def _tools_run(task_id: str) -> list[str]:
    with get_engine().connect() as connection:
        return sorted(connection.execute(
            sa.text("SELECT tool_name FROM tool_invocations WHERE task_id = :t AND status = 'success'"),
            {"t": task_id},
        ).scalars())


def _run_original(client) -> dict:
    task_id = client.post("/tasks", json={"request": REQUEST}).json()["task_id"]
    bundle = client.get(f"/tasks/{task_id}").json()
    assert bundle["status"] == "completed"
    return bundle


def test_strict_replay_reproduces_original_with_zero_api_calls(client):
    original = _run_original(client)
    original_id = original["task_id"]

    launched = client.post(f"/replay/{original_id}", json={}).json()
    assert launched["mode"] == "replay"
    replay_id = launched["replay_task_id"]

    replay = client.get(f"/tasks/{replay_id}").json()
    assert replay["status"] == "completed"
    assert replay["replay_of"] == original_id

    # reproduces the original step-for-step
    assert replay["final_output"] == original["final_output"]
    assert {s["sid"]: s["output"] for s in replay["subtasks"]} == {
        s["sid"]: s["output"] for s in original["subtasks"]
    }

    # zero API calls: every response came from the recording (there is no
    # fallback client in strict mode, so nothing else was even reachable)
    replay_calls = client.get(f"/replay/{replay_id}/steps").json()["steps"]
    assert replay_calls
    assert all(call["model"].startswith("replay:") for call in replay_calls)
    assert client.get(f"/traces/{replay_id}/costs").json()["total_usd"] == 0.0

    # the recorded tool calls were served back, so the same tools ran again
    assert _tools_run(replay_id) == _tools_run(original_id) != []


def test_fork_at_step_k_replays_prefix_and_diverges_after(client):
    original = _run_original(client)
    original_id = original["task_id"]

    steps = client.get(f"/replay/{original_id}/steps").json()["steps"]
    synth_index = next(
        i for i, call in enumerate(steps)
        if call["agent"] == "supervisor" and "Synthesize the final deliverable" in call["prompt"]
    )
    modified_text = "FINAL-FORKED: memo rewritten by the reviewer during replay"

    launched = client.post(
        f"/replay/{original_id}",
        json={"llm_call_id": steps[synth_index]["id"], "response_text": modified_text},
    ).json()
    assert launched["mode"] == "fork"
    fork_id = launched["replay_task_id"]

    fork = client.get(f"/tasks/{fork_id}").json()
    assert fork["status"] == "completed"
    assert fork["final_output"] == modified_text  # step k took the human's response

    # the fork's calls: everything before k replayed, step k is the override,
    # and nothing ran after it (extraction is disabled during replays)
    fork_calls = client.get(f"/replay/{fork_id}/steps").json()["steps"]
    models = [call["model"] for call in fork_calls]
    assert len(fork_calls) == synth_index + 1
    assert models.count("replay:modified") == 1
    assert all(model.startswith("replay:") for model in models)

    # side-by-side comparison: steps < k match, step k diverges
    comparison = client.get(f"/replay/{fork_id}/compare").json()
    assert comparison["original_task_id"] == original_id
    paired = [s for s in comparison["steps"] if s["original"] and s["fork"]]
    synth_step = next(
        s for s in paired if s["fork"]["model"] == "replay:modified"
    )
    assert synth_step["diverged"] is True
    prefix = [s for s in paired if s is not synth_step]
    assert prefix and all(not s["diverged"] for s in prefix)

    assert comparison["final_output"]["diverged"] is True
    assert comparison["final_output"]["fork"] == modified_text
    # subtask outputs were replayed identically
    assert all(not diff["diverged"] for diff in comparison["subtasks"].values())


def test_fork_requires_matching_call_and_paired_fields(client):
    original = _run_original(client)
    original_id = original["task_id"]

    # unknown llm_call_id -> 404
    response = client.post(
        f"/replay/{original_id}", json={"llm_call_id": "nope", "response_text": "x"}
    )
    assert response.status_code == 404

    # one field without the other -> validation error
    response = client.post(f"/replay/{original_id}", json={"llm_call_id": "abc"})
    assert response.status_code == 422

    # compare on a non-replay task -> 400
    assert client.get(f"/replay/{original_id}/compare").status_code == 400


# --- recordings from before rules versions ---------------------------------------

LEGACY_REQUEST = "A request recorded before the plan lint existed."


def _legacy_recording(subtask: dict, specialist_output: str) -> str:
    """A task as recorded before rules versions: no version, a plan whose subtask has no
    `tools` key, and a synthesis call although the plan has one subtask."""
    repo = DBTaskRepo()
    task_id = repo.create_task(LEGACY_REQUEST, rules_version=None)
    plan = {"task_summary": "legacy", "subtasks": [subtask], "confidence": 0.9}
    calls = [
        ("supervisor", "Create an execution plan (the old prompt)", json.dumps(plan)),
        (subtask["specialist"], "Subtask: (the old prompt)", specialist_output),
        ("reviewer", "Review (the old prompt)", json.dumps({"score": 5, "feedback": "fine"})),
        ("supervisor", "Synthesize the final deliverable (the old prompt)", "OLD-FINAL"),
        ("memory", "Extract memories (a replay never makes this call)", "{}"),
    ]
    start = datetime.now(timezone.utc)
    for n, (agent, prompt, response) in enumerate(calls):
        DBLLMCallStore().record(task_id=task_id, agent=agent, model="gpt-4o-mini", prompt=prompt,
                                response=response, tool_calls=[], created_at=start + timedelta(seconds=n))
    repo.set_final_output(task_id, "OLD-FINAL")
    return task_id


def _legacy_subtask(specialist: str, description: str) -> dict:
    return {"id": "s1", "description": description, "specialist": specialist, "required_inputs": [],
            "expected_output_format": "plain text", "estimated_complexity": "low", "depends_on": []}


def _replay(client, original_id: str) -> dict:
    replay_id = client.post(f"/replay/{original_id}", json={}).json()["replay_task_id"]
    return client.get(f"/tasks/{replay_id}").json()


def test_a_recording_from_before_the_lint_replays_without_it(client):
    # its research subtask names no tools, which the current lint would send back for a retry
    original = _legacy_recording(_legacy_subtask("research", "Look up what the cache command removes"), "NOTES")
    assert client.get(f"/tasks/{original}").json()["rules_version"] == 1

    replay = _replay(client, original)

    assert (replay["status"], replay["error"]) == ("completed", None)
    assert replay["rules_version"] == 1
    assert replay["final_output"] == "OLD-FINAL"


def test_a_one_subtask_recording_from_before_versions_replays_its_synthesis(client):
    original = _legacy_recording(_legacy_subtask("writing", "Write the release note"), "WRITER-OUTPUT")

    replay = _replay(client, original)

    assert (replay["status"], replay["error"]) == ("completed", None)
    assert replay["final_output"] == "OLD-FINAL"  # not the writer's output

