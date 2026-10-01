"""GET /traces/{task_id}/tools returns the tool invocation log over HTTP (MOCK_LLM).

The rate-limited case runs with the MCP server in test_mcp_rag_tools.py.
"""

import sqlalchemy as sa

from orchestrator.db.session import get_engine

REQUEST = (
    "Compare open-source vector databases: gather facts about Chroma from the web, "
    "compute the GitHub star ranking from the demo database, generate a comparison "
    "table using Python, and write a comparison memo saved as memo.md."
)


def logged(task_id: str) -> list[tuple]:
    with get_engine().connect() as connection:
        return [
            tuple(row)
            for row in connection.execute(
                sa.text(
                    "SELECT tool_name, status, specialist, subtask_sid, arguments, output, error "
                    "FROM tool_invocations WHERE task_id = :t ORDER BY created_at, id"
                ),
                {"t": task_id},
            )
        ]


def test_a_completed_tasks_tool_calls_match_the_invocation_log(client):
    task_id = client.post("/tasks", json={"request": REQUEST}).json()["task_id"]
    assert client.get(f"/tasks/{task_id}").json()["status"] == "completed"

    body = client.get(f"/traces/{task_id}/tools").json()

    calls = body["calls"]
    assert body["task_id"] == task_id
    assert [
        (c["tool_name"], c["status"], c["specialist"], c["subtask_sid"], c["arguments"], c["output"], c["error"])
        for c in calls
    ] == logged(task_id)
    assert {"web_search", "db_query", "code_exec", "file_write"} <= {c["tool_name"] for c in calls}
    assert [c["created_at"] for c in calls] == sorted(c["created_at"] for c in calls)
    assert all(isinstance(c["latency_ms"], float) and isinstance(c["sensitive"], bool) for c in calls)


def test_an_unknown_task_is_404(client):
    response = client.get("/traces/0123456789abcdef0123456789abcdef/tools")

    assert response.status_code == 404
