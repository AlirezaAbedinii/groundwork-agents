"""The provider client and the budget, against the real SDKs over a mocked transport."""

import json

import anthropic
import httpx
import httpx2  # the Anthropic SDK's fork of httpx
import openai
import pytest
from pydantic import BaseModel, Field

from budget import Budget, BudgetExceeded
from llm import LLM, LLMError, cost_usd, estimate_tokens, load_env, price, strict_schema


class Draft(BaseModel):
    question: str
    quote: str


DRAFT = {"question": "Where is the cache?", "quote": "UV_CACHE_DIR moves it"}


def openai_reply(content=DRAFT, *, refusal=None, finish="stop", tokens=(100, 20)) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": None if refusal else json.dumps(content),
                    "refusal": refusal,
                },
            }
        ],
        "usage": {
            "prompt_tokens": tokens[0],
            "completion_tokens": tokens[1],
            "total_tokens": sum(tokens),
        },
    }


def anthropic_reply(tool_input=DRAFT, *, name="draft", stop="tool_use") -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": name, "input": tool_input}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


class Server:
    """Replies in order and records every request body. Replies are (status, body,
    headers) specs, built with whichever httpx module the SDK under test uses."""

    def __init__(self, *replies: tuple[int, dict, dict]) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.http = httpx

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        status, body, headers = self.replies.pop(0)
        return self.http.Response(status, json=body, headers=headers)


def ok(body: dict) -> tuple[int, dict, dict]:
    return 200, body, {}


def failing(status: int) -> tuple[int, dict, dict]:
    # retry-after-ms keeps the SDKs' backoff short in tests
    return status, {"error": {"message": "busy"}}, {"retry-after-ms": "1"}


def openai_llm(server: Server, budget: Budget, **kwargs) -> LLM:
    client = openai.OpenAI(
        api_key="test",
        max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
    )
    return LLM("openai", "gpt-4o-mini", budget=budget, client=client, **kwargs)


def anthropic_llm(server: Server, budget: Budget) -> LLM:
    server.http = httpx2
    client = anthropic.Anthropic(
        api_key="test",
        max_retries=2,
        http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
    )
    return LLM("anthropic", "claude-haiku-4-5", budget=budget, client=client)


def test_openai_sends_a_strict_schema_and_prices_the_call():
    server, budget = Server(ok(openai_reply())), Budget(1.0)
    draft, usage = openai_llm(server, budget, max_tokens=300).complete(
        "sys", "user", Draft, name="draft"
    )
    assert draft == Draft(**DRAFT)
    body = server.requests[0]
    assert body["model"] == "gpt-4o-mini"
    assert body["max_completion_tokens"] == 300
    assert body["temperature"] == 0.0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "draft"
    schema = fmt["json_schema"]["schema"]
    assert schema["required"] == ["question", "quote"]
    assert schema["additionalProperties"] is False
    # 100 input and 20 output tokens at $0.15 and $0.60 per million
    assert usage.cost_usd == pytest.approx(100 * 0.15e-6 + 20 * 0.60e-6)
    assert budget.calls == 1 and budget.spent_usd == pytest.approx(usage.cost_usd)


def test_anthropic_forces_a_tool_call_with_the_schema():
    server, budget = Server(ok(anthropic_reply())), Budget(1.0)
    draft, usage = anthropic_llm(server, budget).complete("sys", "user", Draft, name="draft")
    assert draft == Draft(**DRAFT)
    body = server.requests[0]
    assert body["system"] == "sys"
    assert body["tool_choice"] == {"type": "tool", "name": "draft"}
    assert body["tools"][0]["name"] == "draft"
    assert body["tools"][0]["input_schema"]["required"] == ["question", "quote"]
    assert "temperature" not in body
    assert usage.cost_usd == pytest.approx(100 * 1.00e-6 + 20 * 5.00e-6)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_errors_are_retried_and_billed_once(status):
    server, budget = Server(failing(status), ok(openai_reply())), Budget(1.0)
    draft, _ = openai_llm(server, budget).complete("sys", "user", Draft)
    assert draft.question == DRAFT["question"]
    assert len(server.requests) == 2
    assert budget.calls == 1


def test_anthropic_retries_too():
    server, budget = Server(failing(529), ok(anthropic_reply())), Budget(1.0)
    anthropic_llm(server, budget).complete("sys", "user", Draft, name="draft")
    assert len(server.requests) == 2 and budget.calls == 1


def test_the_budget_refuses_before_anything_is_sent():
    server = Server(ok(openai_reply()))
    with pytest.raises(BudgetExceeded):
        openai_llm(server, Budget(0.0)).complete("sys", "user", Draft)
    assert server.requests == []


def test_the_worst_case_counts_the_whole_output_allowance():
    # 100 output tokens at $0.60/M is $0.00006, so a $0.00005 cap can't cover the call
    server = Server(ok(openai_reply()))
    with pytest.raises(BudgetExceeded):
        openai_llm(server, Budget(0.00005), max_tokens=100).complete("", "", Draft)
    assert server.requests == []


def test_calls_stop_when_the_next_worst_case_would_not_fit():
    # Each call really costs $0.000027 (100 in, 20 out), but its worst case (the input
    # estimate plus all 100 output tokens) is about $0.00009, so a $0.0001 cap allows
    # one call and refuses the second before sending it.
    server = Server(*[ok(openai_reply()) for _ in range(5)])
    budget = Budget(0.0001)
    llm = openai_llm(server, budget, max_tokens=100)
    with pytest.raises(BudgetExceeded):
        for _ in range(5):
            llm.complete("sys", "user", Draft)
    assert budget.calls == 1 and len(server.requests) == 1
    assert budget.spent_usd <= budget.max_cost_usd


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (openai_reply(refusal="I can't help with that"), "refused"),
        (openai_reply(finish="length"), "max_tokens"),
        (openai_reply(content={"question": "only one field"}), "doesn't match Draft"),
    ],
)
def test_bad_replies_raise_but_are_still_billed(reply, message):
    budget = Budget(1.0)
    with pytest.raises(LLMError, match=message):
        openai_llm(Server(ok(reply)), budget).complete("sys", "user", Draft)
    assert budget.calls == 1


def test_anthropic_without_the_tool_call_raises():
    with pytest.raises(LLMError, match="no 'draft' tool call"):
        anthropic_llm(Server(ok(anthropic_reply(name="other"))), Budget(1.0)).complete(
            "sys", "user", Draft, name="draft"
        )


def test_an_unpriced_model_fails_before_any_call():
    with pytest.raises(ValueError, match="no price"):
        LLM("openai", "gpt-unknown", budget=Budget(1.0), client=object())


def test_prices_can_be_overridden_from_the_environment(monkeypatch):
    monkeypatch.setenv("EVAL_PRICE_GPT_4O_MINI", "1.0,2.0")
    assert price("gpt-4o-mini") == (1.0, 2.0)
    assert cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(3.0)


def test_estimate_tokens_overcounts():
    text = "word " * 400  # about 400 tokens
    assert estimate_tokens(text) > 400


def test_strict_schema_requires_everything_and_drops_defaults():
    class Inner(BaseModel):
        a: int = 1

    class Outer(BaseModel):
        inner: Inner
        tags: list[str] = []
        score: int = Field(default=3, ge=1, le=5)

    schema = strict_schema(Outer)
    assert schema["required"] == ["inner", "tags", "score"]
    assert schema["additionalProperties"] is False
    assert "default" not in json.dumps(schema)
    inner = schema["$defs"]["Inner"]
    assert inner["required"] == ["a"] and inner["additionalProperties"] is False
    assert schema["properties"]["score"]["minimum"] == 1


def test_load_env_sets_without_overriding(tmp_path, monkeypatch):
    monkeypatch.delenv("EVAL_TEST_A", raising=False)
    monkeypatch.setenv("EVAL_TEST_B", "kept")
    env = tmp_path / ".env"
    env.write_text("# comment\nEVAL_TEST_A='from file'\nEVAL_TEST_B=ignored\n")
    load_env(env)
    import os

    assert os.environ["EVAL_TEST_A"] == "from file"
    assert os.environ["EVAL_TEST_B"] == "kept"
