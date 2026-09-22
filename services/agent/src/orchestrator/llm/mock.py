"""Fixture-playback LLM client, active when MOCK_LLM=1.

Recorded provider responses live as JSON files in the fixtures directory
(default: tests/fixtures/llm). ``chat`` renders the message list to text with
``render_messages`` and looks a call up by (agent, rendered text):

  1. ``<fixtures_dir>/<key>.json`` where key = sha256("{agent}::{text}")[:16]
     — an exact recorded call (written by scripts/record_fixtures.py)
  2. any ``match`` fixture for that agent whose needles all occur in the text;
     the one with the most needles wins, filename order breaks ties — used to
     script multi-turn behaviour (a later turn's fixture matches on a tool
     result line that only exists once the tool has run)
  3. ``<fixtures_dir>/<agent>.json`` — an agent-level default response

Fixture file format (v2)::

    {
      "agent": "research",
      "prompt": "...",                     # informational (exact fixtures)
      "match": ["Gather facts about Chroma", "-> web_search returned:"],  # optional
      "response": {
        "text": "...",                     # may be empty when tool_calls are present
        "tool_calls": [                    # optional; ids default to "mock-<key>-<n>"
          {"name": "web_search", "arguments": {"query": "Chroma vector database"}}
        ],
        "model": "gpt-4o",
        "prompt_tokens": 123,
        "completion_tokens": 45
      }
    }

With ``output_schema`` the fixture's text is validated against the schema and
returned as ``LLMResponse.parsed``; invalid text raises ``StructuredOutputError``
exactly as the real client does for a provider parsing failure.

``complete`` (prompt string in, first matching fixture wins) is the legacy sync
path and goes away with the last of its callers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

from orchestrator.llm.messages import render_messages
from orchestrator.llm.structured import StructuredOutputError


@dataclass(frozen=True)
class ToolCall:
    id: str  # provider id or "mock-…"; echoed back as ToolMessage.tool_call_id
    name: str
    arguments: dict  # already a dict on every provider


@dataclass(frozen=True)
class LLMResponse:
    text: str = ""  # assistant text; for output_schema calls: the JSON the model produced
    tool_calls: tuple[ToolCall, ...] = ()  # empty ⇒ the model is done
    parsed: Any = None  # validated output_schema instance when one was requested
    model: str = "mock"
    prompt_tokens: int = 0
    completion_tokens: int = 0


class FixtureNotFoundError(LookupError):
    pass


def fixture_key(agent: str, text: str) -> str:
    return hashlib.sha256(f"{agent}::{text}".encode()).hexdigest()[:16]


def _to_response(payload: dict, key: str) -> LLMResponse:
    response = payload.get("response", {})
    tool_calls = tuple(
        ToolCall(
            id=str(call.get("id") or f"mock-{key[:8]}-{index}"),
            name=call["name"],
            arguments=dict(call.get("arguments", {})),
        )
        for index, call in enumerate(response.get("tool_calls", []))
    )
    return LLMResponse(
        text=response.get("text", ""),
        tool_calls=tool_calls,
        model=response.get("model", "mock"),
        prompt_tokens=int(response.get("prompt_tokens", 0)),
        completion_tokens=int(response.get("completion_tokens", 0)),
    )


class MockLLMClient:
    def __init__(self, fixtures_dir: Path | str):
        self.fixtures_dir = Path(fixtures_dir)

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
        rendered = render_messages(messages)
        key = fixture_key(agent, rendered)
        exact = self.fixtures_dir / f"{key}.json"
        if exact.exists():
            payload = json.loads(exact.read_text(encoding="utf-8"))
        elif matches := self._matches(agent, rendered):
            payload = max(matches, key=lambda found: found[0])[1]  # first maximal ⇒ filename order on ties
        else:
            payload = self._default(agent, key)
        response = _to_response(payload, key)
        if output_schema is not None:
            try:
                response = replace(response, parsed=output_schema.model_validate_json(response.text))
            except ValidationError as exc:
                raise StructuredOutputError(str(exc)) from exc
        return response

    def complete(
        self, agent: str, prompt: str, *, producer_provider: str | None = None
    ) -> LLMResponse:
        key = fixture_key(agent, prompt)
        exact = self.fixtures_dir / f"{key}.json"
        if exact.exists():
            return _to_response(json.loads(exact.read_text(encoding="utf-8")), key)
        if matches := self._matches(agent, prompt):
            return _to_response(matches[0][1], key)
        return _to_response(self._default(agent, key), key)

    def _matches(self, agent: str, text: str) -> list[tuple[int, dict]]:
        """(needle count, payload) for every match-fixture of *agent* whose needles all occur in *text*.

        Filename order, so callers can take the first or the best.
        """
        found: list[tuple[int, dict]] = []
        for path in sorted(self.fixtures_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            match = payload.get("match")
            if payload.get("agent") != agent or not match:
                continue
            needles = [match] if isinstance(match, str) else match
            if all(needle in text for needle in needles):
                found.append((len(needles), payload))
        return found

    def _default(self, agent: str, key: str) -> dict:
        default = self.fixtures_dir / f"{agent}.json"
        if default.exists():
            return json.loads(default.read_text(encoding="utf-8"))
        raise FixtureNotFoundError(
            f"No LLM fixture for agent={agent!r} (key={key}) in {self.fixtures_dir}. "
            f"Add {key}.json for this exact call, a match-fixture, or {agent}.json as an agent default."
        )
