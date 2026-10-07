"""Structured completions from OpenAI or Anthropic, priced and kept under a budget.

The reply's shape is a Pydantic model. OpenAI receives it as a strict JSON-schema
response format; Anthropic receives it as the input schema of a tool it is made to
call. Either way the reply is validated into the model, and the call's token usage
is priced and recorded in the ``Budget``, which refuses a call before it is sent if
its worst case could pass the cap. The SDKs retry 429s and 5xx responses themselves
(``max_retries``); a failed attempt isn't billed.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from budget import Budget

Provider = Literal["openai", "anthropic"]

# USD per 1M tokens (input, output). Copied from
# services/agent/src/orchestrator/llm/pricing.py; keep the two in sync. One model's
# prices can be overridden with EVAL_PRICE_<MODEL>="<input>,<output>", where <MODEL>
# is the id in upper case with "-" and "." as "_" (EVAL_PRICE_GPT_4O_MINI="0.15,0.60").
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),  # legacy
    "claude-haiku-4-5": (1.00, 5.00),
}

KEY_ENV: dict[str, str] = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class LLMError(RuntimeError):
    """The model refused, ran out of output tokens, or returned the wrong shape."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cost_usd: float


def price(model: str) -> tuple[float, float]:
    env = "EVAL_PRICE_" + model.upper().replace("-", "_").replace(".", "_")
    if os.environ.get(env):
        prices = tuple(float(p) for p in os.environ[env].split(","))
        if len(prices) != 2:
            raise ValueError(f'{env} must be "<input>,<output>"')
        return prices  # type: ignore[return-value]
    if model not in PRICES_PER_MTOK:
        raise ValueError(f"no price for {model!r}: add it to PRICES_PER_MTOK or set {env}")
    return PRICES_PER_MTOK[model]


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = price(model)
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def estimate_tokens(text: str) -> int:
    """A deliberately high token count for budgeting: English runs near 4 characters
    per token, so counting 3 overestimates."""
    return math.ceil(len(text) / 3) + 20


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema in the form OpenAI's strict mode accepts: every object
    lists all its properties as required and forbids others, and nothing has a default."""

    def fix(node: Any) -> None:
        if isinstance(node, list):
            for value in node:
                fix(value)
        elif isinstance(node, dict):
            node.pop("default", None)
            if isinstance(node.get("properties"), dict):
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
                for value in node["properties"].values():
                    fix(value)
            for key, value in node.items():
                if key != "properties":
                    fix(value)

    schema = copy.deepcopy(model.model_json_schema())
    fix(schema)
    return schema


def load_env(path: Path) -> None:
    """Set ``KEY=VALUE`` lines from a dotenv file, without overriding the environment."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def make_client(provider: Provider, *, max_retries: int = 4, http_client: Any = None) -> Any:
    key = os.environ.get(KEY_ENV[provider])
    if not key:
        raise LLMError(f"{KEY_ENV[provider]} is not set (in the environment or eval/.env)")
    if provider == "openai":
        import openai

        return openai.OpenAI(api_key=key, max_retries=max_retries, http_client=http_client)
    import anthropic

    return anthropic.Anthropic(api_key=key, max_retries=max_retries, http_client=http_client)


class LLM:
    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        budget: Budget,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        price(model)  # fail on an unpriced model before anything is spent
        self.provider = provider
        self.model = model
        self.budget = budget
        self.max_tokens = max_tokens
        self.temperature = temperature  # OpenAI only: this Anthropic SDK takes none
        self.client = client if client is not None else make_client(provider)

    def complete[M: BaseModel](
        self, system: str, user: str, schema: type[M], *, name: str = "answer"
    ) -> tuple[M, Usage]:
        """One structured completion; raises ``BudgetExceeded`` before the call if its
        worst case could pass the cap, and ``LLMError`` if the reply isn't a ``schema``."""
        json_schema = strict_schema(schema)
        prompt_chars = system + user + json.dumps(json_schema)
        worst = cost_usd(self.model, estimate_tokens(prompt_chars), self.max_tokens)
        self.budget.check(worst)
        if self.provider == "openai":
            data, usage = self._openai(system, user, json_schema, name)
        else:
            data, usage = self._anthropic(system, user, json_schema, name)
        try:
            return schema.model_validate(data), usage
        except ValidationError as exc:
            raise LLMError(f"reply doesn't match {schema.__name__}: {exc}") from exc

    def _record(self, input_tokens: int, output_tokens: int) -> Usage:
        usage = Usage(
            input_tokens, output_tokens, cost_usd(self.model, input_tokens, output_tokens)
        )
        self.budget.record(usage.cost_usd)
        return usage

    def _openai(self, system: str, user: str, json_schema: dict, name: str) -> tuple[Any, Usage]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "schema": json_schema, "strict": True},
            },
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        usage = self._record(response.usage.prompt_tokens, response.usage.completion_tokens)
        choice = response.choices[0]
        if choice.message.refusal:
            raise LLMError(f"the model refused: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise LLMError(f"the reply hit max_tokens ({self.max_tokens})")
        try:
            return json.loads(choice.message.content), usage
        except (TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"the reply isn't JSON: {exc}") from exc

    def _anthropic(self, system: str, user: str, json_schema: dict, name: str) -> tuple[Any, Usage]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {"name": name, "description": "Record the answer.", "input_schema": json_schema}
            ],
            tool_choice={"type": "tool", "name": name},
        )
        usage = self._record(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason == "max_tokens":
            raise LLMError(f"the reply hit max_tokens ({self.max_tokens})")
        for block in response.content:
            if block.type == "tool_use" and block.name == name:
                return block.input, usage
        raise LLMError(f"the reply has no {name!r} tool call")
