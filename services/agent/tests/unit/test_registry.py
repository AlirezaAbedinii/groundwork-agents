import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from orchestrator.llm.router import SPECIALISTS
from orchestrator.tools.base import (
    InMemoryInvocationStore,
    RateLimitExceededError,
    ToolContext,
    ToolExecutionError,
    ToolPermissionError,
    ToolSpec,
    UnknownToolError,
)
from orchestrator.tools.defaults import build_default_registry
from orchestrator.tools.registry import ToolRegistry


class EchoIn(BaseModel):
    msg: str


class EchoOut(BaseModel):
    msg: str


def _echo_spec(rate_limit: int = 2) -> ToolSpec:
    return ToolSpec(
        name="echo",
        description="Echo the message back",
        input_schema=EchoIn,
        output_schema=EchoOut,
        owners=frozenset({"research"}),
        rate_limit=rate_limit,
        handler=lambda args, ctx: EchoOut(msg=args.msg),
        sensitive_when=lambda args: args.msg == "danger",
    )


@pytest.fixture()
def store():
    return InMemoryInvocationStore()


@pytest.fixture()
def registry(store):
    reg = ToolRegistry(store)
    reg.register(_echo_spec())
    return reg


def _ctx(specialist: str = "research") -> ToolContext:
    return ToolContext(task_id="t1", specialist=specialist, subtask_id="s1")


def test_success_is_logged_with_io_and_latency(registry, store):
    output = registry.invoke("echo", {"msg": "hi"}, _ctx())
    assert output == {"msg": "hi"}
    (record,) = store.records
    assert record.status == "success"
    assert record.arguments == {"msg": "hi"}
    assert record.output == {"msg": "hi"}
    assert record.latency_ms >= 0
    assert record.sensitive is False


def test_non_owner_specialist_is_rejected_and_logged(registry, store):
    with pytest.raises(ToolPermissionError):
        registry.invoke("echo", {"msg": "hi"}, _ctx(specialist="writing"))
    (record,) = store.records
    assert record.status == "rejected"
    assert "not permitted" in record.error


def test_unknown_tool_is_rejected_and_logged(registry, store):
    with pytest.raises(UnknownToolError):
        registry.invoke("teleport", {}, _ctx())
    assert store.records[0].status == "rejected"


def test_rate_limit_of_two_blocks_third_call(registry, store):
    registry.invoke("echo", {"msg": "1"}, _ctx())
    registry.invoke("echo", {"msg": "2"}, _ctx())
    with pytest.raises(RateLimitExceededError):
        registry.invoke("echo", {"msg": "3"}, _ctx())
    assert [r.status for r in store.records] == ["success", "success", "rate_limited"]
    # a different task is not affected by t1's counter
    other = ToolContext(task_id="t2", specialist="research")
    assert registry.invoke("echo", {"msg": "4"}, other) == {"msg": "4"}


def test_invalid_arguments_are_logged_as_failure(registry, store):
    with pytest.raises(ToolExecutionError, match="Invalid arguments"):
        registry.invoke("echo", {"wrong": 1}, _ctx())
    assert store.records[0].status == "failure"


def test_sensitive_flag_recorded_per_invocation(registry, store):
    registry.invoke("echo", {"msg": "danger"}, _ctx())
    assert store.records[0].sensitive is True


def test_handler_exception_logged_as_failure(store):
    def boom(args, ctx):
        raise RuntimeError("kaput")

    spec = ToolSpec(
        name="boom", description="always fails", input_schema=EchoIn, output_schema=EchoOut,
        owners=frozenset({"research"}), rate_limit=5, handler=boom,
    )
    registry = ToolRegistry(store)
    registry.register(spec)
    with pytest.raises(ToolExecutionError, match="kaput"):
        registry.invoke("boom", {"msg": "x"}, _ctx())
    assert store.records[0].status == "failure"


def test_tool_definitions_cover_owned_tools_with_their_schema(registry):
    (definition,) = registry.tool_definitions_for("research")
    assert definition["type"] == "function"
    assert definition["function"]["name"] == "echo"
    assert definition["function"]["description"] == "Echo the message back"
    parameters = definition["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["required"] == ["msg"]
    assert parameters["properties"]["msg"]["type"] == "string"

    assert registry.tool_definitions_for("writing") == []


def test_default_tools_all_produce_native_definitions():
    registry = build_default_registry(InMemoryInvocationStore())
    definitions = {
        definition["function"]["name"]: definition
        for specialist in SPECIALISTS
        for definition in registry.tool_definitions_for(specialist)
    }
    assert set(definitions) == {"web_search", "file_read", "file_write", "code_exec", "db_query", "api_call"}
    for name, definition in definitions.items():
        assert definition["type"] == "function", name
        assert definition["function"]["description"], name
        assert definition["function"]["parameters"]["type"] == "object", name
        assert definition["function"]["parameters"]["required"], name
        # what bind_tools runs on every tool: a well-formed definition passes through unchanged
        assert convert_to_openai_tool(definition) == definition, name
