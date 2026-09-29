"""Retrieval tools over MCP in full task runs through the API (MOCK_LLM).

The real MCP server (mcp/server.py) runs under uvicorn in a thread, over a fake
RAG API, and the agent reaches it by URL as in production: the runner discovers
the rag_* tools on the first run, and the research specialist calls them through
the registry. Fixtures it_80..it_88 script the model's side.
"""

import asyncio
import importlib.util
import json
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import sqlalchemy as sa
import uvicorn

from orchestrator.config import get_settings
from orchestrator.db.session import get_engine
from orchestrator.graph import runner

MCP_DIR = Path(__file__).resolve().parents[4] / "mcp"
REQUEST = "Using the Ferry documentation, explain what FERRY-429 means and how clients should retry."
LIMIT_REQUEST = "TRIGGER-RAG-LIMIT: keep listing the Ferry documentation sources until the tool refuses."
VECTOR_REQUEST = (
    "Compare open-source vector databases: gather facts about Chroma from the web, "
    "compute the GitHub star ranking from the demo database, generate a comparison "
    "table using Python, and write a comparison memo saved as memo.md."
)
DOCUMENTS = {
    "01-overview.md": 5,
    "02-quickstart.md": 7,
    "03-configuration.md": 6,
    "04-error-codes.md": 4,
    "05-rate-limits.md": 4,
    "06-architecture.md": 5,
    "README.md": 6,
}
# tests/e2e/test_specialist_tool_use.py's ownership mirror, plus the RAG tools (RAG_MCP_POLICY).
OWNERS = {
    "web_search": {"research"},
    "api_call": {"research"},
    "db_query": {"analysis"},
    "code_exec": {"analysis", "code"},
    "file_read": {"analysis", "writing", "code"},
    "file_write": {"analysis", "writing", "code"},
    "rag_search": {"research"},
    "rag_ask": {"research"},
    "rag_list_sources": {"research"},
}


def fake_rag(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/search":
        text = "FERRY-429 — Rate Limit Exceeded. The response includes a Retry-After header (in seconds). " * 8
        hits = [
            {"chunk_id": "a3f9c2e17b4d8e01", "text": text, "score": 0.91,
             "source_file": "04-error-codes.md", "section_heading": "Request errors"},
            {"chunk_id": "5be80d6a91c2f473", "text": text, "score": 0.74,
             "source_file": "05-rate-limits.md", "section_heading": "What happens when you exceed the limit"},
        ]
        return httpx.Response(200, json={"query": json.loads(request.content)["query"], "mode": "hybrid", "hits": hits})
    if request.url.path == "/v1/documents":
        documents = [{"source_file": name, "chunks": chunks} for name, chunks in DOCUMENTS.items()]
        return httpx.Response(200, json={"documents": documents, "total_chunks": sum(DOCUMENTS.values())})
    return httpx.Response(404, json={"detail": "Not Found"})


@pytest.fixture(scope="module")
def mcp_server_url():
    """The real MCP server on Streamable HTTP, on a free port, over the fake RAG API."""
    sys.path.append(str(MCP_DIR))  # server.py imports tool_models from its own directory
    spec = importlib.util.spec_from_file_location("rag_mcp_server", MCP_DIR / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    http = httpx.AsyncClient(base_url="http://rag.test", transport=httpx.MockTransport(fake_rag))
    app = module.build_server("http://rag.test", http=http).streamable_http_app()

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.01)
    yield f"http://127.0.0.1:{sock.getsockname()[1]}/mcp"
    server.should_exit = True
    thread.join(10)
    sock.close()
    asyncio.run(http.aclose())
    sys.path.remove(str(MCP_DIR))


@pytest.fixture()
def mcp_on(monkeypatch, mcp_server_url):
    """MCP_RAG_URL points at the server, and the runner starts from a registry without the RAG tools."""
    monkeypatch.setattr(get_settings(), "mcp_rag_url", mcp_server_url)
    runner._registry.cache_clear()
    yield
    runner._registry.cache_clear()  # later tests get a registry without them


def invocations(task_id: str) -> list[dict]:
    with get_engine().connect() as connection:
        return [
            dict(row)
            for row in connection.execute(
                sa.text(
                    "SELECT specialist, subtask_sid, tool_name, status, arguments, output, error "
                    "FROM tool_invocations WHERE task_id = :t ORDER BY created_at"
                ),
                {"t": task_id},
            ).mappings()
        ]


def run(client, request: str) -> str:
    task_id = client.post("/tasks", json={"request": request}).json()["task_id"]
    assert client.get(f"/tasks/{task_id}").json()["status"] == "completed"
    return task_id


def test_research_answers_from_the_teams_docs_over_mcp(client, mcp_on):
    task_id = run(client, REQUEST)

    rag = [row for row in invocations(task_id) if row["tool_name"].startswith("rag_")]
    assert sorted((row["specialist"], row["tool_name"], row["status"]) for row in rag) == [
        ("research", "rag_list_sources", "success"),
        ("research", "rag_search", "success"),
    ]
    assert all("04-error-codes.md" in json.dumps(row["output"]) for row in rag)
    search = next(row for row in rag if row["tool_name"] == "rag_search")
    assert search["arguments"] == {"query": "FERRY-429"}

    trace = client.get(f"/traces/{task_id}").json()
    spans = {span["id"]: span for span in trace["spans"]}
    rag_spans = [span for span in trace["spans"] if span["name"] == "tool:rag_search"]
    assert rag_spans
    for span in rag_spans:  # under the research subtask's specialist span
        parent = spans[span["parent_id"]]
        assert (parent["kind"], parent["attributes"]["sid"]) == ("specialist", "s1")
    research_prompts = [call["prompt"] for call in trace["llm_calls"] if call["agent"] == "research"]
    assert any("-> rag_search returned:" in prompt for prompt in research_prompts)
    assert any("04-error-codes.md" in prompt for prompt in research_prompts)  # the retrieval reached the model


def test_the_rate_limit_applies_to_an_mcp_tool_and_the_model_sees_the_refusal(client, mcp_on):
    task_id = run(client, LIMIT_REQUEST)

    listed = [row for row in invocations(task_id) if row["tool_name"] == "rag_list_sources"]
    assert [row["status"] for row in listed] == ["success", "success", "success", "rate_limited"]
    assert "Rate limit of 3 calls per task exceeded" in listed[-1]["error"]

    llm_calls = client.get(f"/traces/{task_id}").json()["llm_calls"]
    last_research_prompt = [call["prompt"] for call in llm_calls if call["agent"] == "research"][-1]
    assert "-> rag_list_sources failed: Rate limit of 3 calls per task exceeded" in last_research_prompt


def test_with_mcp_on_the_existing_flow_runs_and_every_call_respects_ownership(client, mcp_on):
    task_id = run(client, VECTOR_REQUEST)

    rows = invocations(task_id)
    assert rows
    for row in rows:
        assert row["specialist"] in OWNERS[row["tool_name"]], f"{row['specialist']} used {row['tool_name']}"
    # This run discovered the RAG tools for research, though its script never calls them.
    research_tools = {spec.name for spec in runner._registry().tools_for("research")}
    assert {"rag_search", "rag_ask", "rag_list_sources"} <= research_tools


def test_the_ferry_plan_still_wins_with_memories_of_earlier_tasks_in_the_prompt(client, mcp_on):
    # The first task leaves memories about vector databases, which the planner
    # injects into the next prompt; the Ferry plan must still outrank that one.
    run(client, VECTOR_REQUEST)
    task_id = run(client, REQUEST)

    used = {(row["specialist"], row["tool_name"]) for row in invocations(task_id)}
    assert ("research", "rag_search") in used
    assert ("research", "web_search") not in used
