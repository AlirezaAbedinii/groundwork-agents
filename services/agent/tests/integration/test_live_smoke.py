"""Live smoke tests: real provider round trips for the native paths.

Run with `pytest -m live` and real API keys. Each test skips without its
provider's key, and `make test` and CI deselect them. Every test pins a small
model itself (gpt-4o-mini or claude-haiku-4-5), so nothing here reaches a
larger model whatever the MODEL_* settings say, and each checks its own token
budget. They run one after another on one event loop, as a Celery worker runs
its tasks: the providers' HTTP clients are shared process-wide and belong to
the loop that first used them.
"""

from dataclasses import replace

import pytest

from orchestrator.agents.specialists import ResearchSpecialist
from orchestrator.config import get_settings
from orchestrator.llm import clients
from orchestrator.llm.clients import RealLLMClient
from orchestrator.llm.pricing import cost_usd
from orchestrator.llm.router import ModelChoice
from orchestrator.planning.decomposer import decompose
from orchestrator.planning.schemas import ExecutionPlan
from orchestrator.tools import web_search
from orchestrator.tools.base import InMemoryInvocationStore, ToolContext
from orchestrator.tools.registry import ToolRegistry
from orchestrator.workers.loop import run_in_worker_loop

pytestmark = pytest.mark.live

needs_openai = pytest.mark.skipif(not get_settings().openai_api_key, reason="OPENAI_API_KEY not configured")
needs_anthropic = pytest.mark.skipif(
    not get_settings().anthropic_api_key, reason="ANTHROPIC_API_KEY not configured"
)

REQUEST = (
    "Research the three most popular open-source vector databases, compare their "
    "GitHub activity, and write a one-page recommendation memo."
)
TOKEN_BUDGET = 5_000  # per test


class MeteredClient:
    """The real client pinned to one small model, keeping each response's usage."""

    def __init__(self, monkeypatch, provider: str, model: str):
        monkeypatch.setattr(clients, "route", lambda agent, producer_provider=None: ModelChoice(provider, model))
        self.inner = RealLLMClient()
        self.responses = []

    async def chat(self, agent, messages, **kwargs):
        response = await self.inner.chat(agent, messages, **kwargs)
        self.responses.append(response)
        return response

    def check_usage(self) -> None:
        tokens = sum(r.prompt_tokens + r.completion_tokens for r in self.responses)
        cost = sum(cost_usd(r.model, r.prompt_tokens, r.completion_tokens) for r in self.responses)
        print(f"\n{len(self.responses)} call(s), {tokens} tokens, ${cost:.4f}")
        assert tokens < TOKEN_BUDGET


def _check_plan(plan: ExecutionPlan) -> None:
    assert isinstance(plan, ExecutionPlan)
    assert len(plan.subtasks) >= 2
    assert 0.0 <= plan.confidence <= 1.0
    assert plan.topological_waves()  # acyclic


@needs_openai
def test_openai_plans_through_strict_json_schema(monkeypatch):
    llm = MeteredClient(monkeypatch, "openai", "gpt-4o-mini")

    _check_plan(run_in_worker_loop(decompose(llm, REQUEST)))
    llm.check_usage()


@needs_anthropic
def test_anthropic_plans_through_json_schema(monkeypatch):
    llm = MeteredClient(monkeypatch, "anthropic", "claude-haiku-4-5")

    _check_plan(run_in_worker_loop(decompose(llm, REQUEST)))
    llm.check_usage()


@needs_openai
def test_openai_tool_call_round_trips_through_the_loop(monkeypatch):
    llm = MeteredClient(monkeypatch, "openai", "gpt-4o-mini")
    canned = web_search.WebSearchOutput(results=[web_search.SearchResult(
        title="Chroma", url="https://www.trychroma.com", snippet="Chroma is an open-source embedding database.",
    )])
    store = InMemoryInvocationStore()
    registry = ToolRegistry(store)
    registry.register(replace(web_search.SPEC, handler=lambda args, ctx: canned))  # no real search
    spec = {
        "id": "s1",
        "description": "Use web_search to find out what the Chroma vector database is, then answer in one sentence.",
        "expected_output_format": "one sentence ending with the source URL",
    }
    ctx = ToolContext(task_id="live-smoke", specialist="research", subtask_id="s1")

    result = run_in_worker_loop(ResearchSpecialist(llm, registry).execute(spec, {}, None, ctx))

    # the model asked natively, the tool ran, and its answer came after the tool result
    assert "web_search" in result.tool_calls
    assert any(r.tool_name == "web_search" and r.status == "success" for r in store.records)
    assert len(llm.responses) >= 2
    assert result.output
    llm.check_usage()
