"""LLM client abstraction: real provider clients or the fixture player.

Agents call ``await client.chat(agent, messages, tools=..., output_schema=...,
producer_provider=...)`` and stay oblivious to providers; routing happens here
(via llm/router.py). MOCK_LLM=1 swaps in the fixture player so nothing leaves
the process. ``complete`` is the legacy prompt-string path, kept only until
its last callers move to ``chat``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from orchestrator.config import get_settings
from orchestrator.llm.mock import LLMResponse, MockLLMClient, ToolCall
from orchestrator.llm.router import route
from orchestrator.llm.structured import StructuredOutputError


class LLMClient(Protocol):
    async def chat(
        self,
        agent: str,
        messages: list[BaseMessage],
        *,
        tools: list[dict] | None = None,
        output_schema: type[BaseModel] | None = None,
        producer_provider: str | None = None,
    ) -> LLMResponse: ...

    def complete(
        self, agent: str, prompt: str, *, producer_provider: str | None = None
    ) -> LLMResponse: ...


def _usage(message: AIMessage) -> dict[str, int]:
    usage = message.usage_metadata or {}
    return {
        "prompt_tokens": int(usage.get("input_tokens", 0)),
        "completion_tokens": int(usage.get("output_tokens", 0)),
    }


class RealLLMClient:
    """Routes each call to OpenAI or Anthropic via the langchain chat models."""

    @staticmethod
    @lru_cache(maxsize=8)
    def _chat_model(provider: str, model: str):
        settings = get_settings()
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model, api_key=settings.openai_api_key, temperature=0)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=model, api_key=settings.anthropic_api_key, temperature=0)
        raise ValueError(f"Unknown provider {provider!r}")

    async def chat(
        self,
        agent: str,
        messages: list[BaseMessage],
        *,
        tools: list[dict] | None = None,
        output_schema: type[BaseModel] | None = None,
        producer_provider: str | None = None,
    ) -> LLMResponse:
        if tools and output_schema is not None:
            raise ValueError("tools and output_schema are mutually exclusive")
        choice = route(agent, producer_provider)
        model = self._chat_model(choice.provider, choice.model)
        model_id = f"{choice.provider}:{choice.model}"

        if output_schema is not None:
            # Native structured outputs: the provider constrains decoding to the
            # schema; strict mode exists on OpenAI only (Anthropic is always exact).
            options: dict = {"method": "json_schema", "include_raw": True}
            if choice.provider == "openai":
                options["strict"] = True
            result = await model.with_structured_output(output_schema, **options).ainvoke(messages)
            raw, parsed, error = result["raw"], result["parsed"], result["parsing_error"]
            if error is not None or parsed is None:
                raise StructuredOutputError(
                    f"{model_id} did not return a valid {output_schema.__name__}: {error or raw.text!r}"
                )
            return LLMResponse(text=parsed.model_dump_json(), parsed=parsed, model=model_id, **_usage(raw))

        runnable = model.bind_tools(tools) if tools else model
        message = await runnable.ainvoke(messages)
        return LLMResponse(
            text=message.text,
            tool_calls=tuple(
                ToolCall(id=call["id"] or f"call-{index}", name=call["name"], arguments=dict(call["args"]))
                for index, call in enumerate(message.tool_calls)
            ),
            model=model_id,
            **_usage(message),
        )

    def complete(
        self, agent: str, prompt: str, *, producer_provider: str | None = None
    ) -> LLMResponse:
        choice = route(agent, producer_provider)
        message = self._chat_model(choice.provider, choice.model).invoke(prompt)
        return LLMResponse(
            text=message.text, model=f"{choice.provider}:{choice.model}", **_usage(message)
        )


def get_llm_client() -> LLMClient:
    settings = get_settings()
    if settings.mock_llm:
        return MockLLMClient(settings.llm_fixtures_dir)
    return RealLLMClient()
