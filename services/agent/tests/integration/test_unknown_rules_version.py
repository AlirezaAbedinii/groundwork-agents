"""A rules version this build does not know (orchestrator/rules.py) is refused before a
run, a resume or a replay starts: the task fails with the reason and nothing runs."""

from orchestrator.db.repo import DBLLMCallStore, DBTaskRepo
from orchestrator.graph.runner import production_graph, run_task
from orchestrator.observability.replay import run_replay
from orchestrator.workers.loop import run_in_worker_loop

REQUEST = (
    "Compare open-source vector databases: gather facts about Chroma from the web, "
    "compute the GitHub star ranking from the demo database, generate a comparison "
    "table using Python, and write a comparison memo saved as memo.md."
)
REFUSED = "Unknown rules version 99; this build knows [1, 2, 3, 4]"


def _pending(client, task_id: str) -> dict:
    (approval,) = [a for a in client.get("/approvals", params={"status": "pending"}).json()["approvals"]
                   if a["task_id"] == task_id]
    return approval


async def _checkpoint_rules_version(task_id: str, version: int) -> None:
    """Write ``version`` into the task's checkpoint, as a newer build's run would leave it."""
    async with production_graph(discover=False) as graph:
        await graph.aupdate_state({"configurable": {"thread_id": task_id}}, {"rules_version": version})


def test_a_resume_at_the_final_gate_under_an_unknown_version_is_refused(client):
    task_id = client.post("/tasks", json={"request": REQUEST, "require_human_review": True}).json()["task_id"]
    client.post(f"/approvals/{_pending(client, task_id)['id']}/resolve", json={"action": "approve"})
    final_gate = _pending(client, task_id)
    assert final_gate["gate_key"] == "final"
    run_in_worker_loop(_checkpoint_rules_version(task_id, 99))

    client.post(f"/approvals/{final_gate['id']}/resolve", json={"action": "approve"})

    bundle = client.get(f"/tasks/{task_id}").json()
    assert (bundle["status"], bundle["error"]) == ("failed", REFUSED)
    assert bundle["final_output"] is None  # nothing was delivered


def test_a_run_under_an_unknown_version_is_refused(client):
    task_id = DBTaskRepo().create_task(REQUEST, rules_version=99)

    run_in_worker_loop(run_task(task_id))

    bundle = client.get(f"/tasks/{task_id}").json()
    assert (bundle["status"], bundle["error"], bundle["plan"]) == ("failed", REFUSED, None)
    assert DBLLMCallStore().for_task(task_id) == []


def test_a_replay_under_an_unknown_version_is_refused(client):
    original = client.post("/tasks", json={"request": REQUEST}).json()["task_id"]
    replay_id = DBTaskRepo().create_task(REQUEST, replay_of=original, rules_version=99)

    run_in_worker_loop(run_replay(replay_id, original))

    bundle = client.get(f"/tasks/{replay_id}").json()
    assert (bundle["status"], bundle["error"]) == ("failed", REFUSED)
    assert DBLLMCallStore().for_task(replay_id) == []
