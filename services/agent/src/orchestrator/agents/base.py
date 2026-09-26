"""Base agent: named LLM caller."""

from __future__ import annotations

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from orchestrator.llm.clients import LLMClient
from orchestrator.llm.mock import LLMResponse


class BaseAgent:
    name: str = "agent"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def chat(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[dict] | None = None,
        output_schema: type[BaseModel] | None = None,
        producer_provider: str | None = None,
    ) -> LLMResponse:
        return await self.llm.chat(
            self.name, messages, tools=tools, output_schema=output_schema, producer_provider=producer_provider
        )
