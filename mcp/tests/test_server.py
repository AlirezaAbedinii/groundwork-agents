"""Acceptance tests for server.py: the retrieval service's API as MCP tools.

These tests are the server's spec. Most build ``build_server(RAG_URL, http=rag_http)``
over the fake RAG API in conftest.py and call it the way an agent does, through
``mcp.Client``, in-process. The contract they pin:

- Three tools, ``search``, ``ask`` and ``list_sources``, annotated read-only and
  closed-world. Their docstrings are the descriptions the model reads.
- Each call sends one request to the RAG API (``POST /v1/search``, ``POST /v1/ask``,
  ``GET /v1/documents``) and returns a model from tool_models.py. Arguments left
  at None are not sent, so the RAG API applies its own defaults.
- Results fit the agent loop's budget: ``search`` returns snippets of at most
  SNIPPET_CHARS, and ``ask`` drops the retrieved contexts.
- A refused answer is a result, not an error.
- Failures reach the model as readable tool errors:
  "retrieval service returned {status}: {detail}" and
  "retrieval service unreachable at {rag_url}".
- Bad arguments are rejected by the SDK before the tool body runs.
- ``main()``: ``--transport http`` (default, Streamable HTTP on /mcp) or ``stdio``,
  ``--host`` 127.0.0.1, ``--port`` 8001.
"""

import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from mcp import Client
from mcp.server.mcpserver import MCPServer

from conftest import ANSWER, CHUNKS, RAG_URL, REFUSAL, SOURCES, chunk_text
from server import build_server, main
from tool_models import (
    DEFAULT_TOP_K,
    RESULT_BUDGET_CHARS,
    SNIPPET_CHARS,
    AskResult,
    SearchResult,
    SourcesResult,
)

pytestmark = pytest.mark.anyio

EVERY_TOOL = [
    pytest.param("search", {"query": "FERRY-429"}, id="search"),
    pytest.param("ask", {"question": "What does FERRY-429 mean?"}, id="ask"),
    pytest.param("list_sources", {}, id="list_sources"),
]


@pytest.fixture
async def client(rag_http):
    """An MCP client connected in-process to the server under test."""
    async with Client(build_server(RAG_URL, http=rag_http)) as client:
        yield client


def text(result) -> str:
    """What the model reads from a tool result."""
    return "\n".join(block.text for block in result.content)


def nullable(prop: dict) -> tuple[dict, bool]:
    """A property's non-null schema, and whether the property also accepts null."""
    variants = prop.get("anyOf", [prop])
    non_null = [v for v in variants if v.get("type") != "null"]
    assert len(non_null) == 1, prop
    return non_null[0], len(non_null) < len(variants)


# --- What an MCP client discovers -------------------------------------------------


async def test_lists_exactly_the_three_read_only_tools(client):
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {"search", "ask", "list_sources"}
    for tool in tools.values():
        assert tool.description and tool.description.strip()
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.open_world_hint is False


async def test_schemas_are_the_contract_the_model_sees(client):
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    search = tools["search"].input_schema
    assert search["required"] == ["query"]
    props = search["properties"]
    assert set(props) == {"query", "top_k", "mode"}
    assert props["query"]["type"] == "string"
    assert (props["query"]["minLength"], props["query"]["maxLength"]) == (1, 2000)
    top_k, top_k_nullable = nullable(props["top_k"])
    assert (top_k["type"], top_k["minimum"], top_k["maximum"]) == ("integer", 1, 20)
    assert props["top_k"]["default"] == DEFAULT_TOP_K and not top_k_nullable
    mode, mode_nullable = nullable(props["mode"])
    assert sorted(mode["enum"]) == ["dense", "hybrid"] and mode_nullable
    assert props["mode"]["default"] is None

    ask = tools["ask"].input_schema
    assert ask["required"] == ["question"]
    props = ask["properties"]
    assert set(props) == {"question", "mode", "top_k"}
    assert (props["question"]["minLength"], props["question"]["maxLength"]) == (1, 2000)
    top_k, top_k_nullable = nullable(props["top_k"])
    assert (top_k["type"], top_k["minimum"], top_k["maximum"]) == ("integer", 1, 20)
    assert props["top_k"]["default"] is None and top_k_nullable
    mode, mode_nullable = nullable(props["mode"])
    assert sorted(mode["enum"]) == ["dense", "hybrid"] and mode_nullable

    assert not tools["list_sources"].input_schema.get("properties")

    # Returning the result models gives clients an output schema and structured content.
    assert tools["search"].output_schema == SearchResult.model_json_schema()
    assert tools["ask"].output_schema == AskResult.model_json_schema()
    assert tools["list_sources"].output_schema == SourcesResult.model_json_schema()


# --- One request per call, results shaped for the agent -----------------------------


async def test_search_sends_one_request_and_returns_ranked_snippets(client, fake_rag):
    result = await client.call_tool("search", {"query": "FERRY-429"})

    assert result.is_error is False
    # top_k takes the tool's default; mode is left out, so the RAG API picks it.
    sent = {"query": "FERRY-429", "top_k": DEFAULT_TOP_K}
    assert fake_rag.sent() == [("POST", "/v1/search", sent)]
    found = SearchResult.model_validate(result.structured_content)
    assert (found.query, found.mode) == ("FERRY-429", "hybrid")  # the mode the RAG API used
    assert [hit.chunk_id for hit in found.hits] == [chunk[0] for chunk in CHUNKS[:DEFAULT_TOP_K]]
    first = found.hits[0]
    assert (first.source_file, first.section_heading) == ("04-error-codes.md", "Request errors")
    assert first.score == pytest.approx(1 / 61, abs=5e-4)  # rounding is fine, reordering isn't
    for hit in found.hits:
        assert 0 < len(hit.snippet) <= SNIPPET_CHARS
        assert chunk_text(fake_rag.chunk_chars).startswith(hit.snippet.removesuffix("…"))

    # A second call on the same injected client: the tool must not close it.
    result = await client.call_tool("search", {"query": "retry", "mode": "dense", "top_k": 5})

    assert result.is_error is False
    sent = {"query": "retry", "mode": "dense", "top_k": 5}
    assert fake_rag.sent()[1] == ("POST", "/v1/search", sent)
    found = SearchResult.model_validate(result.structured_content)
    assert found.mode == "dense"
    assert found.hits[3].section_heading is None  # a null heading passes through


async def test_ask_returns_the_answer_and_citations_without_contexts(client, fake_rag):
    result = await client.call_tool("ask", {"question": "What does FERRY-429 mean?"})

    assert result.is_error is False
    assert fake_rag.sent() == [("POST", "/v1/ask", {"question": "What does FERRY-429 mean?"})]
    assert set(result.structured_content) == {
        "answer",
        "refused",
        "confidence",
        "citations",
        "cost_usd",
    }
    answered = AskResult.model_validate(result.structured_content)
    assert (answered.answer, answered.refused) == (ANSWER, False)
    assert answered.confidence == pytest.approx(0.8123, abs=5e-4)
    assert answered.cost_usd == pytest.approx(0.000261)
    assert [(c.index, c.chunk_id, c.source_file, c.supported) for c in answered.citations] == [
        (1, CHUNKS[0][0], "04-error-codes.md", True),
        (2, CHUNKS[1][0], "05-rate-limits.md", True),
    ]

    await client.call_tool("ask", {"question": "Retry policy?", "mode": "dense", "top_k": 4})
    assert fake_rag.sent()[1] == (
        "POST",
        "/v1/ask",
        {"question": "Retry policy?", "mode": "dense", "top_k": 4},
    )


async def test_a_refused_answer_is_a_result_not_an_error(client, fake_rag):
    fake_rag.refuse = True

    result = await client.call_tool("ask", {"question": "What is the meaning of life?"})

    assert result.is_error is False
    refused = AskResult.model_validate(result.structured_content)
    assert (refused.refused, refused.answer, refused.citations) == (True, REFUSAL, [])


async def test_list_sources_reports_documents_and_chunk_counts(client, fake_rag):
    result = await client.call_tool("list_sources", {})

    assert result.is_error is False
    assert fake_rag.sent() == [("GET", "/v1/documents", None)]
    listed = SourcesResult.model_validate(result.structured_content)
    assert {source.source_file: source.chunks for source in listed.sources} == SOURCES
    assert listed.total_chunks == sum(SOURCES.values())


@pytest.mark.parametrize(("tool", "arguments"), EVERY_TOOL)
async def test_results_at_default_arguments_fit_the_agent_loop_budget(
    client, fake_rag, tool, arguments
):
    fake_rag.chunk_chars = 2000  # long chunks come in; only snippets may go out

    result = await client.call_tool(tool, arguments)

    assert result.is_error is False
    # The agent's registry wraps the result as {"content": ...} before the loop cuts it.
    assert len(json.dumps({"content": result.structured_content})) <= RESULT_BUDGET_CHARS


# --- Failures the model can read and act on -----------------------------------------


@pytest.mark.parametrize(("tool", "arguments"), EVERY_TOOL)
async def test_a_rag_error_status_reaches_the_model_with_its_detail(
    client, fake_rag, tool, arguments
):
    detail = (
        "Missing required configuration: OPENAI_API_KEY. "
        "Set it in your environment or .env file (see .env.example)."
    )
    fake_rag.fail = (503, {"detail": detail})

    result = await client.call_tool(tool, arguments)

    assert result.is_error is True
    assert f"retrieval service returned 503: {detail}" in text(result)


async def test_an_error_body_without_json_is_reported_as_text(client, fake_rag):
    fake_rag.fail = (502, "Bad Gateway")

    result = await client.call_tool("list_sources", {})

    assert result.is_error is True
    assert "retrieval service returned 502: Bad Gateway" in text(result)


@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectError("connection refused"), httpx.ReadTimeout("timed out")],
    ids=["connect", "timeout"],
)
@pytest.mark.parametrize(("tool", "arguments"), EVERY_TOOL)
async def test_an_unreachable_rag_is_a_readable_tool_error(
    client, fake_rag, tool, arguments, failure
):
    fake_rag.raises = failure

    result = await client.call_tool(tool, arguments)

    assert result.is_error is True
    assert f"retrieval service unreachable at {RAG_URL}" in text(result)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        pytest.param("search", {"query": "FERRY-429", "top_k": 99}, id="search-top_k-99"),
        pytest.param("search", {"query": ""}, id="search-empty-query"),
        pytest.param("search", {"query": "FERRY-429", "mode": "sparse"}, id="search-bad-mode"),
        pytest.param("ask", {"question": ""}, id="ask-empty-question"),
        pytest.param("ask", {"question": "FERRY-429?", "top_k": 0}, id="ask-top_k-0"),
    ],
)
async def test_bad_arguments_fail_before_any_request(client, fake_rag, tool, arguments):
    result = await client.call_tool(tool, arguments)

    assert result.is_error is True
    assert fake_rag.requests == []


# --- Configuration, transports and the command line ---------------------------------


async def test_the_rag_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("RAG_API_URL", "http://127.0.0.1:9")  # nothing listens there

    async with Client(build_server()) as client:
        result = await client.call_tool("list_sources", {})

    assert result.is_error is True
    assert "retrieval service unreachable at http://127.0.0.1:9" in text(result)


@pytest.fixture
def served(rag_http):
    """The server under test on Streamable HTTP (uvicorn in a thread, free port)."""
    app = build_server(RAG_URL, http=rag_http).streamable_http_app()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
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


async def test_serves_streamable_http(served):
    async with Client(served) as client:
        protocol = client.protocol_version
        tools = (await client.list_tools()).tools
        result = await client.call_tool("list_sources", {})

    assert protocol == "2026-07-28"
    assert sorted(tool.name for tool in tools) == ["ask", "list_sources", "search"]
    assert result.is_error is False
    assert result.structured_content["total_chunks"] == sum(SOURCES.values())


@pytest.fixture
def runs(monkeypatch) -> list[tuple[str, dict]]:
    """Records MCPServer.run(transport, **options) instead of serving."""
    calls: list[tuple[str, dict]] = []

    def run(self, transport="stdio", **options):
        calls.append((transport, options))

    monkeypatch.setattr(MCPServer, "run", run)
    return calls


def test_main_serves_streamable_http_on_localhost_8001_by_default(runs):
    main([])

    [(transport, options)] = runs
    assert transport == "streamable-http"
    assert (options["host"], options["port"]) == ("127.0.0.1", 8001)
    assert options.get("streamable_http_path", "/mcp") == "/mcp"


def test_main_takes_host_and_port(runs):
    main(["--host", "0.0.0.0", "--port", "9001"])

    [(transport, options)] = runs
    assert (transport, options["host"], options["port"]) == ("streamable-http", "0.0.0.0", 9001)


def test_main_serves_stdio_on_request(runs):
    main(["--transport", "stdio"])

    assert runs == [("stdio", {})]


def test_main_rejects_an_unknown_transport(runs, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--transport", "carrier-pigeon"])

    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert runs == []
