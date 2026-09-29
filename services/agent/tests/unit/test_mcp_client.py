"""The MCP client adapter: server tools become registry ToolSpecs, so they're governed like local tools.

Discovery turns each tool a policy allows into a ToolSpec (``rag_`` + its name);
the registry then applies permissions, rate limits, validation, logging and the
approval check to it exactly as to a local tool. Runs against the in-process
stub in conftest.py, which records what reaches the server.
"""

import asyncio
import logging

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from orchestrator.tools.base import (
    InMemoryInvocationStore,
    RateLimitExceededError,
    ToolContext,
    ToolExecutionError,
    ToolPermissionError,
)
from orchestrator.tools.mcp_client import call_tool, discover_tools
from orchestrator.tools.mcp_models import McpToolOutput, McpToolPolicy, McpUnavailableError
from orchestrator.tools.registry import ToolRegistry

RESEARCH = frozenset({"research"})
POLICY = {
    "search": McpToolPolicy(owners=RESEARCH, rate_limit=10),
    "ask": McpToolPolicy(owners=RESEARCH, rate_limit=5),
    "list_sources": McpToolPolicy(owners=RESEARCH, rate_limit=3),
}
UNREACHABLE = "http://127.0.0.1:9/mcp"  # nothing listens on the discard port


def discover(target, policy=POLICY):
    return asyncio.run(discover_tools(target, policy, prefix="rag_"))


def registry_with(specs) -> tuple[ToolRegistry, InMemoryInvocationStore]:
    store = InMemoryInvocationStore()
    registry = ToolRegistry(store)
    for spec in specs:
        registry.register(spec)
    return registry, store


def ctx(specialist: str = "research") -> ToolContext:
    return ToolContext(task_id="t1", specialist=specialist, subtask_id="s1")


# --- Discovery -------------------------------------------------------------------


def test_discovery_registers_only_the_tools_the_policy_allows(rag_stub, caplog):
    caplog.set_level(logging.INFO)
    policy = POLICY | {"summarize": McpToolPolicy(owners=frozenset({"writing"}), rate_limit=1)}

    specs = discover(rag_stub.server, policy)

    assert sorted(spec.name for spec in specs) == ["rag_ask", "rag_list_sources", "rag_search"]
    # Offered by the server without a policy entry: skipped (deny by default), noted at INFO.
    assert any(r.levelno == logging.INFO and "delete_everything" in r.getMessage() for r in caplog.records)
    # Allowed by the policy but not offered by the server: a WARNING.
    assert any(r.levelno == logging.WARNING and "summarize" in r.getMessage() for r in caplog.records)
    assert rag_stub.calls == []  # discovery lists tools; it calls none


def test_specs_carry_the_policy_and_the_servers_description(rag_stub):
    policy = POLICY | {"ask": McpToolPolicy(owners=frozenset({"research", "analysis"}), rate_limit=2, sensitive=True)}

    specs = {spec.name: spec for spec in discover(rag_stub.server, policy)}

    ask = specs["rag_ask"]
    assert (ask.owners, ask.rate_limit, ask.sensitive) == (frozenset({"research", "analysis"}), 2, True)
    assert specs["rag_search"].description == "Search the team's docs."
    assert all(spec.output_schema is McpToolOutput for spec in specs.values())


def test_the_llm_sees_the_servers_parameters_and_only_owners_see_the_tools(rag_stub):
    registry, _ = registry_with(discover(rag_stub.server))

    definitions = {d["function"]["name"]: d for d in registry.tool_definitions_for("research")}

    parameters = definitions["rag_search"]["function"]["parameters"]
    assert parameters["required"] == ["query"]
    assert set(parameters["properties"]) == {"query", "top_k", "mode"}  # the server's Context isn't one
    top_k = parameters["properties"]["top_k"]
    assert (top_k["type"], top_k["minimum"], top_k["maximum"], top_k["default"]) == ("integer", 1, 20, 3)
    assert parameters["properties"]["query"]["minLength"] == 1
    assert parameters["properties"]["mode"]["anyOf"][0]["enum"] == ["dense", "hybrid"]
    for name, definition in definitions.items():
        assert convert_to_openai_tool(definition) == definition, name
    assert not [d for d in registry.tool_definitions_for("writing") if d["function"]["name"].startswith("rag_")]


def test_discovering_an_unreachable_server_raises_unavailable():
    with pytest.raises(McpUnavailableError, match="ConnectError"):
        discover(UNREACHABLE)


# --- Calls through the registry -----------------------------------------------------


def test_a_call_reaches_the_server_under_its_own_name_with_only_the_arguments_set(rag_stub):
    registry, store = registry_with(discover(rag_stub.server))

    output = registry.invoke("rag_search", {"query": "FERRY-429"}, ctx())

    assert output == {
        "content": {"query": "FERRY-429", "hits": [{"source_file": "04-error-codes.md", "snippet": "FERRY-429: rate limited"}]}
    }
    assert [record.status for record in store.records] == ["success"]
    # The unprefixed name, top_k filled in from the server's default, and no mode key at all.
    assert rag_stub.calls == [("search", {"query": "FERRY-429", "top_k": 3})]


def test_a_result_without_structured_content_is_its_text(rag_stub):
    registry, _ = registry_with(discover(rag_stub.server, POLICY | {"version": McpToolPolicy(RESEARCH, 1)}))

    assert registry.invoke("rag_version", {}, ctx()) == {"content": "rag-stub 1.0"}


def test_a_server_error_is_a_tool_error_the_model_can_read(rag_stub):
    registry, store = registry_with(discover(rag_stub.server))
    rag_stub.fail = "retrieval service returned 503: Missing required configuration: OPENAI_API_KEY."

    with pytest.raises(ToolExecutionError, match="retrieval service returned 503: Missing required configuration"):
        registry.invoke("rag_ask", {"question": "What does FERRY-429 mean?"}, ctx())

    assert [record.status for record in store.records] == ["failure"]


def test_an_unreachable_server_is_a_tool_error_naming_the_cause():
    with pytest.raises(ToolExecutionError) as error:
        call_tool(UNREACHABLE, "search", {"query": "FERRY-429"}, timeout_s=5)

    message = str(error.value)
    assert message.startswith("MCP server unreachable")
    assert "ConnectError" in message
    assert "TaskGroup" not in message  # the anyio ExceptionGroup is unwrapped to its cause


# --- The registry governs MCP tools like local ones -----------------------------------


def test_permissions_apply_before_the_server_is_called(rag_stub):
    registry, store = registry_with(discover(rag_stub.server))

    with pytest.raises(ToolPermissionError):
        registry.invoke("rag_search", {"query": "FERRY-429"}, ctx("writing"))

    assert [record.status for record in store.records] == ["rejected"]
    assert rag_stub.calls == []


def test_rate_limits_apply_per_task(rag_stub):
    registry, store = registry_with(discover(rag_stub.server, POLICY | {"search": McpToolPolicy(RESEARCH, 1)}))

    registry.invoke("rag_search", {"query": "FERRY-429"}, ctx())
    with pytest.raises(RateLimitExceededError):
        registry.invoke("rag_search", {"query": "retry"}, ctx())

    assert [record.status for record in store.records] == ["success", "rate_limited"]
    assert len(rag_stub.calls) == 1


@pytest.mark.parametrize(
    "arguments",
    [{"query": ""}, {"query": "FERRY-429", "top_k": 99}, {"query": "FERRY-429", "delete": True}],
    ids=["empty-query", "top_k-99", "unknown-argument"],
)
def test_arguments_are_validated_before_the_server_is_called(rag_stub, arguments):
    registry, store = registry_with(discover(rag_stub.server))

    with pytest.raises(ToolExecutionError, match="Invalid arguments"):
        registry.invoke("rag_search", arguments, ctx())

    assert [record.status for record in store.records] == ["failure"]
    assert rag_stub.calls == []


def test_the_policy_decides_what_needs_approval_not_the_servers_annotations(rag_stub):
    registry, _ = registry_with(discover(rag_stub.server, POLICY | {"ask": McpToolPolicy(RESEARCH, 5, sensitive=True)}))

    assert registry.is_sensitive_call("rag_ask", {"question": "What does FERRY-429 mean?"}) is True
    # The stub marks search destructive; the policy says it isn't sensitive, and the policy wins.
    assert registry.is_sensitive_call("rag_search", {"query": "FERRY-429"}) is False


@pytest.mark.anyio
async def test_calls_run_at_once_from_worker_threads_as_the_specialist_loop_runs_them(rag_stub):
    registry, store = registry_with(await discover_tools(rag_stub.server, POLICY, prefix="rag_"))

    outputs = await asyncio.gather(
        asyncio.to_thread(registry.invoke, "rag_search", {"query": "FERRY-429"}, ctx()),
        asyncio.to_thread(registry.invoke, "rag_list_sources", {}, ctx()),
    )

    assert outputs[0]["content"]["query"] == "FERRY-429"
    assert outputs[1]["content"]["total_chunks"] == 8
    assert sorted(record.status for record in store.records) == ["success", "success"]
