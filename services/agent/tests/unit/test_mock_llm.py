import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from orchestrator.llm.messages import render_messages
from orchestrator.llm.mock import FixtureNotFoundError, MockLLMClient, fixture_key
from orchestrator.llm.structured import StructuredOutputError


def _write_fixture(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.anyio
async def test_plays_back_exact_recorded_call(tmp_path):
    messages = [HumanMessage(content="Plan the following task: compare vector databases")]
    prompt = render_messages(messages)
    key = fixture_key("supervisor", prompt)
    _write_fixture(
        tmp_path / f"{key}.json",
        {
            "agent": "supervisor",
            "prompt": prompt,
            "response": {"text": "PLAN", "model": "gpt-4o", "prompt_tokens": 10, "completion_tokens": 5},
        },
    )

    response = await MockLLMClient(tmp_path).chat("supervisor", messages)
    assert response.text == "PLAN"
    assert response.model == "gpt-4o"
    assert (response.prompt_tokens, response.completion_tokens) == (10, 5)


@pytest.mark.anyio
async def test_falls_back_to_agent_default(tmp_path):
    _write_fixture(
        tmp_path / "reviewer.json",
        {"agent": "reviewer", "response": {"text": "APPROVED"}},
    )

    response = await MockLLMClient(tmp_path).chat("reviewer", [HumanMessage(content="any prompt at all")])
    assert response.text == "APPROVED"
    assert response.model == "mock"


@pytest.mark.anyio
async def test_missing_fixture_raises_with_key(tmp_path):
    messages = [HumanMessage(content="unrecorded prompt")]
    with pytest.raises(FixtureNotFoundError) as excinfo:
        await MockLLMClient(tmp_path).chat("writer", messages)
    assert fixture_key("writer", render_messages(messages)) in str(excinfo.value)
    assert "writer.json" in str(excinfo.value)


# --- chat: message-list contract ------------------------------------------


def _conversation():
    return [SystemMessage(content="You are research."), HumanMessage(content="Gather facts about Chroma")]


@pytest.mark.anyio
async def test_chat_plays_back_tool_calls_with_ids(tmp_path):
    messages = _conversation()
    key = fixture_key("research", render_messages(messages))
    _write_fixture(
        tmp_path / f"{key}.json",
        {
            "agent": "research",
            "response": {
                "text": "",
                "tool_calls": [
                    {"name": "web_search", "arguments": {"query": "Chroma vector database"}},
                    {"id": "call_abc", "name": "api_call", "arguments": {"url": "https://example.com"}},
                ],
                "model": "mock-gpt-4o-mini",
                "prompt_tokens": 310,
                "completion_tokens": 40,
            },
        },
    )

    response = await MockLLMClient(tmp_path).chat("research", messages, tools=[{"type": "function"}])

    assert response.text == ""
    assert response.parsed is None
    assert [c.name for c in response.tool_calls] == ["web_search", "api_call"]
    assert response.tool_calls[0].arguments == {"query": "Chroma vector database"}
    assert response.tool_calls[0].id.startswith("mock-")
    assert response.tool_calls[1].id == "call_abc"
    assert len({c.id for c in response.tool_calls}) == 2
    assert (response.model, response.prompt_tokens, response.completion_tokens) == ("mock-gpt-4o-mini", 310, 40)


class _Verdict(BaseModel):
    score: int = Field(ge=1, le=5)
    feedback: str


@pytest.mark.anyio
async def test_chat_output_schema_validates_fixture_text(tmp_path):
    _write_fixture(tmp_path / "reviewer.json", {"agent": "reviewer", "response": {"text": '{"score": 5, "feedback": "ok"}'}})
    client = MockLLMClient(tmp_path)

    response = await client.chat("reviewer", [HumanMessage(content="Review this")], output_schema=_Verdict)
    assert response.parsed == _Verdict(score=5, feedback="ok")
    assert response.text == '{"score": 5, "feedback": "ok"}'
    assert response.tool_calls == ()

    _write_fixture(tmp_path / "reviewer.json", {"agent": "reviewer", "response": {"text": '{"score": 7, "feedback": "ok"}'}})
    with pytest.raises(StructuredOutputError) as excinfo:
        await client.chat("reviewer", [HumanMessage(content="Review this")], output_schema=_Verdict)
    assert "score" in str(excinfo.value)


@pytest.mark.anyio
async def test_chat_prefers_the_fixture_with_the_most_needles(tmp_path):
    _write_fixture(
        tmp_path / "a_first_turn.json",
        {"agent": "research", "match": ["Gather facts about Chroma"], "response": {"text": "first"}},
    )
    _write_fixture(
        tmp_path / "b_after_search.json",
        {
            "agent": "research",
            "match": ["Gather facts about Chroma", "-> web_search returned:"],
            "response": {"text": "second"},
        },
    )
    client = MockLLMClient(tmp_path)
    messages = _conversation()

    assert (await client.chat("research", messages)).text == "first"

    messages += [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "web_search", "args": {"query": "Chroma"}, "type": "tool_call"}]),
        ToolMessage(content='{"hits": 3}', tool_call_id="c1", name="web_search"),
    ]
    assert (await client.chat("research", messages)).text == "second"


@pytest.mark.anyio
async def test_chat_breaks_needle_ties_by_filename(tmp_path):
    _write_fixture(tmp_path / "z_late.json", {"agent": "writer", "match": ["Draft"], "response": {"text": "late"}})
    _write_fixture(tmp_path / "a_early.json", {"agent": "writer", "match": ["Draft"], "response": {"text": "early"}})

    response = await MockLLMClient(tmp_path).chat("writer", [HumanMessage(content="Draft the memo")])
    assert response.text == "early"


def test_render_messages_covers_every_line_kind():
    messages = [
        SystemMessage(content="You are research."),
        HumanMessage(content="Gather facts about Chroma"),
        AIMessage(
            content="Searching.",
            tool_calls=[
                {"id": "c1", "name": "web_search", "args": {"query": "Chroma", "limit": 3}, "type": "tool_call"},
                {"id": "c2", "name": "api_call", "args": {"url": "https://x"}, "type": "tool_call"},
            ],
        ),
        ToolMessage(content='{"hits": []}', tool_call_id="c1", name="web_search"),
        ToolMessage(content="boom", tool_call_id="c2", name="api_call", status="error"),
        AIMessage(content="", tool_calls=[{"id": "c3", "name": "web_search", "args": {}, "type": "tool_call"}]),
    ]

    assert render_messages(messages) == (
        "[system] You are research.\n\n"
        "[user] Gather facts about Chroma\n\n"
        "[assistant] Searching.\n"
        '-> called web_search({"limit": 3, "query": "Chroma"})\n'
        '-> called api_call({"url": "https://x"})\n\n'
        '-> web_search returned: {"hits": []}\n\n'
        "-> api_call failed: boom\n\n"
        "-> called web_search({})"
    )
