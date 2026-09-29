"""The real MCP server (mcp/server.py) behind the real adapter and registry, over a fake RAG API.

The server is loaded by path, since the mcp/ project isn't a package; its RAG
API is an httpx.MockTransport that records every request. No service runs.
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
import uvicorn

from orchestrator.tools.base import InMemoryInvocationStore, ToolContext, ToolExecutionError
from orchestrator.tools.mcp_client import discover_tools
from orchestrator.tools.mcp_models import McpToolPolicy
from orchestrator.tools.registry import ToolRegistry

MCP_DIR = Path(__file__).resolve().parents[4] / "mcp"
RESEARCH = frozenset({"research"})
POLICY = {
    "search": McpToolPolicy(owners=RESEARCH, rate_limit=10),
    "ask": McpToolPolicy(owners=RESEARCH, rate_limit=5),
    "list_sources": McpToolPolicy(owners=RESEARCH, rate_limit=3),
}
CHUNK = "FERRY-429 — Rate Limit Exceeded. The response includes a Retry-After header. " * 12


@pytest.fixture(scope="module")
def server_module():
    """mcp/server.py, imported by path; it imports tool_models from its own directory."""
    sys.path.append(str(MCP_DIR))
    try:
        spec = importlib.util.spec_from_file_location("rag_mcp_server", MCP_DIR / "server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(MCP_DIR))


@pytest.fixture()
def rag_requests() -> list[httpx.Request]:
    return []


@pytest.fixture()
def rag_server(server_module, rag_requests):
    """The real server over a fake RAG API: search answers, ask is keyless (503), documents list."""

    def fake_rag(request: httpx.Request) -> httpx.Response:
        rag_requests.append(request)
        if request.url.path == "/v1/search":
            hit = {
                "chunk_id": "a3f9c2e17b4d8e01",
                "text": CHUNK,
                "score": 0.91,
                "source_file": "04-error-codes.md",
                "section_heading": "Request errors",
            }
            return httpx.Response(200, json={"query": json.loads(request.content)["query"], "mode": "hybrid", "hits": [hit]})
        if request.url.path == "/v1/ask":
            return httpx.Response(503, json={"detail": "Missing required configuration: OPENAI_API_KEY."})
        if request.url.path == "/v1/documents":
            return httpx.Response(200, json={"documents": [{"source_file": "04-error-codes.md", "chunks": 4}], "total_chunks": 4})
        return httpx.Response(404, json={"detail": "Not Found"})

    http = httpx.AsyncClient(base_url="http://rag.test", transport=httpx.MockTransport(fake_rag))
    yield server_module.build_server("http://rag.test", http=http)
    asyncio.run(http.aclose())


@pytest.fixture()
def served_rag(rag_server):
    """The same server on Streamable HTTP (uvicorn in a thread, free port); yields its URL."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(rag_server.streamable_http_app(), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.01)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(10)
    sock.close()


def registry_for(target) -> tuple[ToolRegistry, InMemoryInvocationStore]:
    store = InMemoryInvocationStore()
    registry = ToolRegistry(store)
    for spec in asyncio.run(discover_tools(target, POLICY, prefix="rag_")):
        registry.register(spec)
    return registry, store


CTX = ToolContext(task_id="t1", specialist="research", subtask_id="s1")


def test_the_real_server_in_process_through_the_registry(rag_server):
    registry, store = registry_for(rag_server)

    definitions = {d["function"]["name"]: d["function"] for d in registry.tool_definitions_for("research")}
    assert sorted(definitions) == ["rag_ask", "rag_list_sources", "rag_search"]
    assert definitions["rag_search"]["description"].startswith("Search the team's indexed documentation")

    hit = registry.invoke("rag_search", {"query": "FERRY-429"}, CTX)["content"]["hits"][0]
    assert (hit["source_file"], hit["section_heading"]) == ("04-error-codes.md", "Request errors")
    assert CHUNK.startswith(hit["snippet"]) and len(hit["snippet"]) <= 300

    with pytest.raises(ToolExecutionError, match="retrieval service returned 503: Missing required configuration"):
        registry.invoke("rag_ask", {"question": "What does FERRY-429 mean?"}, CTX)
    assert [record.status for record in store.records] == ["success", "failure"]


def test_the_real_server_over_streamable_http(served_rag):
    registry, store = registry_for(served_rag)

    output = registry.invoke("rag_list_sources", {}, CTX)

    assert output == {"content": {"sources": [{"source_file": "04-error-codes.md", "chunks": 4}], "total_chunks": 4}}
    assert [record.status for record in store.records] == ["success"]


def test_arguments_the_server_would_reject_stop_in_the_registry(rag_server, rag_requests):
    registry, store = registry_for(rag_server)

    with pytest.raises(ToolExecutionError, match="Invalid arguments"):
        registry.invoke("rag_search", {"query": "FERRY-429", "top_k": 99}, CTX)
    assert rag_requests == []  # nothing reached the RAG API

    registry.invoke("rag_search", {"query": "FERRY-429"}, CTX)
    assert [(r.url.path, json.loads(r.content)) for r in rag_requests] == [
        ("/v1/search", {"query": "FERRY-429", "top_k": 3})
    ]
