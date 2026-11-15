"""The per-task ceilings against Postgres: the store's totals, and a run and a resume
through the API that end at the call ceiling."""

import pytest

from orchestrator.config import get_settings
from orchestrator.db.repo import DBLLMCallStore

DOOMED_REQUEST = "TRIGGER-FAILURE: research the cursed topic thoroughly."


def _call(task_id: str | None, cost: float) -> dict:
    return dict(
        span_id=None, task_id=task_id, agent="research", model="gpt-4o", prompt="p", response="r",
        tool_calls=[], prompt_tokens=1, completion_tokens=1, cost_usd=cost,
    )


def test_totals_count_and_sum_one_tasks_calls():
    store = DBLLMCallStore()
    for task_id, cost in (("t1", 0.01), ("t1", 0.02), ("t1", 0.0), ("t2", 0.5), (None, 0.5)):
        store.record(**_call(task_id, cost))

    calls, cost = store.totals("t1")

    assert calls == 3
    assert cost == pytest.approx(0.03)
    assert store.totals("unknown") == (0, 0.0)


def test_a_run_that_reaches_the_call_ceiling_fails_with_the_reason(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_task_llm_calls", 4)

    task_id = client.post("/tasks", json={"request": DOOMED_REQUEST}).json()["task_id"]

    bundle = client.get(f"/tasks/{task_id}").json()
    assert bundle["status"] == "failed"
    assert bundle["error"].startswith("task budget exceeded: 4 calls, $")
    assert DBLLMCallStore().totals(task_id)[0] == 4
    assert client.get("/approvals", params={"status": "pending"}).json()["approvals"] == []


def test_a_resume_counts_the_calls_made_before_the_pause(client, monkeypatch):
    task_id = client.post("/tasks", json={"request": DOOMED_REQUEST}).json()["task_id"]
    assert client.get(f"/tasks/{task_id}").json()["status"] == "awaiting_approval"
    made = DBLLMCallStore().totals(task_id)[0]  # the plan, then two attempts of every turn

    monkeypatch.setattr(get_settings(), "max_task_llm_calls", made + 1)
    (approval,) = client.get("/approvals", params={"status": "pending"}).json()["approvals"]
    response = client.post(f"/approvals/{approval['id']}/resolve", json={"action": "approve", "notes": ""})
    assert response.status_code == 200

    bundle = client.get(f"/tasks/{task_id}").json()
    assert bundle["status"] == "failed"
    assert bundle["error"].startswith(f"task budget exceeded: {made + 1} calls")
    assert DBLLMCallStore().totals(task_id)[0] == made + 1
