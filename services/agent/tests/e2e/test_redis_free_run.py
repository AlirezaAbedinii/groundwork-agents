"""An inline run with working memory in Postgres never touches Redis.

This is how a deployment without Redis runs (RUN_MODE=inline,
WORKING_MEMORY_BACKEND=postgres). REDIS_URL points at a closed port here, so a
Redis call in the run would fail the task ("Connection refused"), and one in the
approval chat or the purge would fail the request.
"""

import pytest
import redis
from fastapi.testclient import TestClient

from orchestrator.config import get_settings
from orchestrator.graph import runner
from orchestrator.main import create_app
from orchestrator.memory.working import PostgresWorkingMemory, get_working_memory

REQUEST = (
    "Compare open-source vector databases: gather facts about Chroma from the web, "
    "compute the GitHub star ranking from the demo database, generate a comparison "
    "table using Python, and write a comparison memo saved as memo.md."
)
CLOSED_PORT_URL = "redis://127.0.0.1:1/0"


@pytest.fixture()
def redis_free_client(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "working_memory_backend", "postgres")
    monkeypatch.setattr(settings, "redis_url", CLOSED_PORT_URL)
    runner._dependencies.cache_clear()  # the runner rebuilds its working memory from these settings
    yield TestClient(create_app())
    runner._dependencies.cache_clear()  # later tests get the Redis store back


def _pending_one(client, task_id):
    rows = client.get("/approvals", params={"status": "pending"}).json()["approvals"]
    (row,) = [r for r in rows if r["task_id"] == task_id]
    return row


def test_inline_run_keeps_working_memory_in_postgres(redis_free_client):
    client = redis_free_client
    with pytest.raises(redis.exceptions.ConnectionError):
        redis.Redis.from_url(CLOSED_PORT_URL).ping()  # the premise: no Redis to reach
    memory = get_working_memory()
    assert isinstance(memory, PostgresWorkingMemory)

    task_id = client.post("/tasks", json={"request": REQUEST, "require_human_review": True}).json()["task_id"]

    # paused at the plan gate, with the plan in Postgres
    assert client.get(f"/tasks/{task_id}").json()["status"] == "awaiting_approval"
    assert memory.get_user(task_id) == "default"
    assert len(memory.get_plan(task_id)["subtasks"]) == 4

    # the chat panel reads its working-memory snapshot from Postgres
    approval = _pending_one(client, task_id)
    chat = client.post(f"/approvals/{approval['id']}/chat", json={"question": "What will this plan do?"})
    assert chat.status_code == 200

    # approve: the run resumes, and its outputs land in Postgres before synthesis
    client.post(f"/approvals/{approval['id']}/resolve", json={"action": "approve"})
    final = _pending_one(client, task_id)
    assert final["gate_key"] == "final"
    assert len(memory.get_subtask_outputs(task_id)) == 4

    # approve the deliverable: the task completes and its working memory is cleared
    client.post(f"/approvals/{final['id']}/resolve", json={"action": "approve"})
    assert client.get(f"/tasks/{task_id}").json()["status"] == "completed"
    assert memory.snapshot(task_id) == {"fields": {}, "errors": []}

    # the user-data purge clears working memory through the same backend
    purged = client.delete("/memory/users/default").json()
    assert purged["deleted"]["working_memory_tasks"] == 0
