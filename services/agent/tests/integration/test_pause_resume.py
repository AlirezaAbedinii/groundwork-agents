"""Pause → approve → resume, end to end through the API.

A user-requested review pauses the run at the plan gate before any agent
work, the approval row packages the full decision context, the chat panel
answers grounded questions, and approving resumes exactly where it paused —
up to the final-deliverable gate and through to completion.
"""

import sqlalchemy as sa

from orchestrator.db.session import get_engine
from orchestrator.rules import CURRENT_RULES_VERSION

REQUEST = (
    "Compare open-source vector databases: gather facts about Chroma from the web, "
    "compute the GitHub star ranking from the demo database, generate a comparison "
    "table using Python, and write a comparison memo saved as memo.md."
)


def _tool_invocation_count(task_id: str) -> int:
    with get_engine().connect() as connection:
        return connection.execute(
            sa.text("SELECT count(*) FROM tool_invocations WHERE task_id = :t"), {"t": task_id}
        ).scalar()


def test_pause_approve_resume_end_to_end(client):
    created = client.post("/tasks", json={"request": REQUEST, "require_human_review": True})
    task_id = created.json()["task_id"]

    # --- paused at the plan gate, before any agent work -------------------
    bundle = client.get(f"/tasks/{task_id}").json()
    assert bundle["status"] == "awaiting_approval"
    assert all(s["status"] == "pending" for s in bundle["subtasks"])
    assert _tool_invocation_count(task_id) == 0

    pending = client.get("/approvals", params={"status": "pending"}).json()["approvals"]
    (approval,) = [a for a in pending if a["task_id"] == task_id]
    assert approval["trigger"] == "user_requested"
    assert approval["level"] == "approve_plan"
    assert approval["gate_key"] == "plan"

    # full context package
    context = approval["context"]
    assert context["task"]["request"] == REQUEST
    assert len(context["plan"]["subtasks"]) == 4
    assert context["completed_steps"] == {}
    assert approval["proposed_action"]["type"] == "execute_plan"
    assert "requested human review" in approval["reasoning"]

    # --- chat panel: grounded answer about the paused task -----------------
    # (the fixture only matches when the prompt actually carries task context)
    chat = client.post(
        f"/approvals/{approval['id']}/chat", json={"question": "What will this plan do?"}
    )
    assert chat.status_code == 200
    assert "waiting for your approval" in chat.json()["answer"]

    # --- approve: resumes exactly where it paused --------------------------
    resolved = client.post(
        f"/approvals/{approval['id']}/resolve", json={"action": "approve", "notes": "looks right"}
    ).json()
    assert resolved["resumed"] is True
    assert resolved["approval"]["review_seconds"] is not None
    assert resolved["approval"]["review_seconds"] >= 0

    # work happened only after the approval, then paused again at the final gate
    bundle = client.get(f"/tasks/{task_id}").json()
    assert bundle["status"] == "awaiting_approval"
    assert all(s["status"] == "completed" for s in bundle["subtasks"])
    assert _tool_invocation_count(task_id) > 0

    pending = client.get("/approvals", params={"status": "pending"}).json()["approvals"]
    (final_approval,) = [a for a in pending if a["task_id"] == task_id]
    assert final_approval["gate_key"] == "final"
    assert final_approval["proposed_action"]["type"] == "deliver"
    assert final_approval["proposed_action"]["final_output"].startswith("FINAL")

    # --- approve the deliverable: task completes ----------------------------
    client.post(f"/approvals/{final_approval['id']}/resolve", json={"action": "approve"})
    bundle = client.get(f"/tasks/{task_id}").json()
    assert bundle["status"] == "completed"
    assert bundle["final_output"].startswith("FINAL")

    # human review time recorded on every resolved approval
    all_rows = client.get("/approvals", params={"task_id": task_id}).json()["approvals"]
    assert len(all_rows) == 2
    assert all(row["status"] == "resolved" for row in all_rows)
    assert all(row["review_seconds"] is not None for row in all_rows)


def test_a_modified_plan_that_fails_the_lint_is_refused_and_the_approval_stays_pending(client):
    task_id = client.post("/tasks", json={"request": REQUEST, "require_human_review": True}).json()["task_id"]
    bundle = client.get(f"/tasks/{task_id}").json()
    assert bundle["rules_version"] == CURRENT_RULES_VERSION
    pending = client.get("/approvals", params={"status": "pending"}).json()["approvals"]
    (approval,) = [a for a in pending if a["task_id"] == task_id]
    star_ranking = next(s for s in bundle["plan"]["subtasks"] if s["tools"] == ["db_query"])
    # one subtask, so the planned memo.md deliverable (whose producer is gone) goes too
    misassigned = {**bundle["plan"], "subtasks": [{**star_ranking, "specialist": "research"}], "deliverables": []}

    refused = client.post(f"/approvals/{approval['id']}/resolve",
                          json={"action": "modify", "payload": {"plan": misassigned}})

    assert refused.status_code == 422
    assert refused.json()["detail"].startswith(
        f"Modified plan fails the plan lint: {star_ranking['id']}: 'db_query' belongs to analysis, not research"
    )
    assert client.get(f"/approvals/{approval['id']}").json()["status"] == "pending"
    assert client.get(f"/tasks/{task_id}").json()["status"] == "awaiting_approval"
    assert _tool_invocation_count(task_id) == 0

    # given back to the specialist that owns db_query, the same plan is accepted and the run resumes
    fixed = {**misassigned, "subtasks": [{**star_ranking, "specialist": "analysis"}]}
    accepted = client.post(f"/approvals/{approval['id']}/resolve", json={"action": "modify", "payload": {"plan": fixed}})
    assert accepted.status_code == 200
    assert client.get(f"/tasks/{task_id}").json()["subtasks"][0]["specialist"] == "analysis"
