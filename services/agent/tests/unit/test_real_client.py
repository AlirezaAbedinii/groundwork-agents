"""RealLLMClient.chat against a stub chat model: no provider is ever reached."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from orchestrator.llm import clients
from orchestrator.llm.clients import RealLLMClient
from orchestrator.llm.mock import ToolCall
from orchestrator.llm.router import ModelChoice
from orchestrator.llm.structured import StructuredOutputError


class _StubChatModel:
    """Records what the client binds and returns a canned ainvoke result."""

    def __init__(self, result):
        self.result = result
        self.bound_tools = None
        self.structured_schema = None
        self.structured_kwargs = None
        self.received = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    def with_structured_output(self, schema, **kwargs):
        self.structured_schema = schema
        self.structured_kwargs = kwargs
        return self

    async def ainvoke(self, messages):
        self.received = messages
        return self.result


class _Verdict(BaseModel):
    score: int = Field(ge=1, le=5)
    feedback: str


@pytest.fixture()
def stub(monkeypatch):
    """Install a stub chat model and let each test choose the routed provider."""
    holder = {}

    def install(result, provider="openai", model="gpt-4o-mini"):
        holder["model"] = _StubChatModel(result)
        monkeypatch.setattr(RealLLMClient, "_chat_model", staticmethod(lambda p, m: holder["model"]))
        monkeypatch.setattr(clients, "route", lambda agent, producer_provider=None: ModelChoice(provider, model))
        return holder["model"]

    return install


_MESSAGES = [SystemMessage(content="You are research."), HumanMessage(content="Gather facts about Chroma")]


@pytest.mark.anyio
async def test_chat_binds_tool_dicts_and_maps_tool_calls(stub):
    tools = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}]
    model = stub(
        AIMessage(
            content="Searching.",
            tool_calls=[{"id": "call_1", "name": "web_search", "args": {"query": "Chroma"}, "type": "tool_call"}],
            usage_metadata={"input_tokens": 120, "output_tokens": 9, "total_tokens": 129},
        )
    )

    response = await RealLLMClient().chat("research", _MESSAGES, tools=tools)

    assert model.bound_tools is tools
    assert model.received is _MESSAGES
    assert model.structured_schema is None
    assert response.text == "Searching."
    assert response.tool_calls == (ToolCall(id="call_1", name="web_search", arguments={"query": "Chroma"}),)
    assert response.parsed is None
    assert (response.model, response.prompt_tokens, response.completion_tokens) == ("openai:gpt-4o-mini", 120, 9)

    with pytest.raises(ValueError, match="mutually exclusive"):
        await RealLLMClient().chat("research", _MESSAGES, tools=tools, output_schema=_Verdict)


@pytest.mark.anyio
async def test_chat_structured_output_is_strict_on_openai_only(stub):
    verdict = _Verdict(score=4, feedback="fine")
    raw = AIMessage(content=verdict.model_dump_json(), usage_metadata={"input_tokens": 50, "output_tokens": 12, "total_tokens": 62})
    result = {"raw": raw, "parsed": verdict, "parsing_error": None}

    model = stub(result, provider="openai", model="gpt-4o")
    response = await RealLLMClient().chat("supervisor", _MESSAGES, output_schema=_Verdict)
    assert model.structured_schema is _Verdict
    assert model.structured_kwargs == {"method": "json_schema", "include_raw": True, "strict": True}
    assert model.bound_tools is None
    assert response.parsed == verdict
    assert response.text == verdict.model_dump_json()
    assert response.tool_calls == ()
    assert (response.model, response.prompt_tokens, response.completion_tokens) == ("openai:gpt-4o", 50, 12)

    model = stub(result, provider="anthropic", model="claude-haiku-4-5")
    response = await RealLLMClient().chat("reviewer", _MESSAGES, output_schema=_Verdict, producer_provider="openai")
    assert model.structured_kwargs == {"method": "json_schema", "include_raw": True}
    assert response.parsed == verdict
    assert response.model == "anthropic:claude-haiku-4-5"


@pytest.mark.anyio
async def test_chat_parsing_error_becomes_structured_output_error(stub):
    stub({"raw": AIMessage(content="not json"), "parsed": None, "parsing_error": ValueError("score must be <= 5")})

    with pytest.raises(StructuredOutputError) as excinfo:
        await RealLLMClient().chat("supervisor", _MESSAGES, output_schema=_Verdict)
    assert "_Verdict" in str(excinfo.value)
    assert "score must be <= 5" in str(excinfo.value)
