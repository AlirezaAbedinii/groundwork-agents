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
from langchain_core.messages import ToolMessage

from orchestrator.agents.specialists import ResearchSpecialist
from orchestrator.llm.mock import LLMResponse, ToolCall
from orchestrator.tools.base import InMemoryInvocationStore, ToolContext, ToolExecutionError
from orchestrator.tools.defaults import RAG_MCP_POLICY
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
            body = json.loads(request.content)
            hits = [
                {
                    "chunk_id": f"a3f9c2e17b4d8e{n:02d}",
                    "text": CHUNK,
                    "score": 0.91 - n / 100,
                    "source_file": "04-error-codes.md",
                    "section_heading": "Request errors",
                }
                for n in range(body["top_k"])
            ]
            return httpx.Response(200, json={"query": body["query"], "mode": "hybrid", "hits": hits})
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


def registry_for(target, policy=POLICY) -> tuple[ToolRegistry, InMemoryInvocationStore]:
    store = InMemoryInvocationStore()
    registry = ToolRegistry(store)
    for spec in asyncio.run(discover_tools(target, policy, prefix="rag_")):
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


class SearchThenFinish:
    """A model that asks for one ten-hit search, then finishes; keeps the messages of its second turn."""

    def __init__(self) -> None:
        search = ToolCall(id="call_1", name="rag_search", arguments={"query": "FERRY-429", "top_k": 10})
        self.turns = [LLMResponse(text="", tool_calls=(search,)), LLMResponse(text="DONE")]
        self.second_turn: list = []

    async def chat(self, agent, messages, *, tools=None, output_schema=None, producer_provider=None):
        self.second_turn = list(messages)
        return self.turns.pop(0)


def test_a_ten_hit_search_reaches_the_model_as_whole_hits_within_its_budget(rag_server):
    registry, store = registry_for(rag_server, RAG_MCP_POLICY)  # the agent's own policy: 4000 chars for search
    llm = SearchThenFinish()
    task = {"id": "s1", "description": "Explain FERRY-429", "expected_output_format": "plain text"}

    asyncio.run(ResearchSpecialist(llm, registry).execute(task, {}, None, CTX))

    answer = llm.second_turn[-1]
    assert isinstance(answer, ToolMessage) and answer.status == "success"
    assert len(answer.content) <= 4000
    content = json.loads(answer.content)["content"]
    every_hit = store.records[0].output["content"]["hits"]  # what the tool returned, in full
    shown = content["truncated"]["shown"]
    assert content["truncated"] == {"shown": shown, "of": 10} and len(every_hit) == 10
    assert content["hits"] == every_hit[:shown] and shown > 3  # whole hits, more than a default search (top_k 3)
