"""Plans and review verdicts arrive as native structured output (LLMResponse.parsed).

The schema-level checks Pydantic runs after decoding (the plan's DAG rules, the
verdict's 1-5 range) still reject bad output; those errors are retried once
with the error in the prompt.
"""

import json

import pytest

from orchestrator.agents.reviewer import REVIEW_MARKER, Reviewer, ReviewVerdict
from orchestrator.llm.messages import render_messages
from orchestrator.llm.mock import LLMResponse, MockLLMClient
from orchestrator.planning.decomposer import PLAN_MARKER, PlanValidationError, decompose
from orchestrator.planning.schemas import ExecutionPlan

pytestmark = pytest.mark.anyio


class RecordingLLM:
    """Forwards chat calls to *inner* and keeps what each one was asked."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []

    async def chat(self, agent, messages, **kwargs):
        self.calls.append({"agent": agent, "prompt": render_messages(messages), **kwargs})
        return await self.inner.chat(agent, messages, **kwargs)


class CannedLLM:
    """Always returns the same response."""

    def __init__(self, response: LLMResponse):
        self.response = response

    async def chat(self, agent, messages, **kwargs):
        return self.response


def _fixture(directory, name: str, agent: str, text: str, match: list[str]) -> None:
    payload = {"agent": agent, "match": match, "response": {"text": text}}
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def _plan(subtasks: list[dict], confidence: float = 0.9) -> dict:
    return {"task_summary": "compare two databases", "subtasks": subtasks, "confidence": confidence}


def _subtask(sid: str, depends_on: list[str]) -> dict:
    return {"id": sid, "description": f"step {sid}", "specialist": "research", "depends_on": depends_on}


CYCLIC = _plan([_subtask("s1", ["s2"]), _subtask("s2", ["s1"])])


async def test_a_cyclic_plan_is_retried_once_with_the_error_then_fails(tmp_path):
    _fixture(tmp_path, "plan", "supervisor", json.dumps(CYCLIC), match=[PLAN_MARKER])
    # the retry prompt carries the marker and the rejection, so this fixture wins it
    _fixture(tmp_path, "plan_retry", "supervisor", json.dumps(CYCLIC),
             match=[PLAN_MARKER, "Your previous plan was invalid"])
    llm = RecordingLLM(MockLLMClient(tmp_path))

    with pytest.raises(PlanValidationError, match=r"(?s)^Plan invalid after retry: .*Dependency cycle"):
        await decompose(llm, "compare Chroma and Qdrant")

    assert [call["output_schema"] for call in llm.calls] == [ExecutionPlan, ExecutionPlan]
    first, retry = (call["prompt"] for call in llm.calls)
    assert "Your previous plan was invalid" not in first
    assert "Your previous plan was invalid" in retry
    assert "Dependency cycle among subtasks: ['s1', 's2']" in retry  # the validation error itself
    assert "Respond with ONLY" not in first + retry  # the schema travels natively, not in the prompt


async def test_a_valid_plan_comes_back_parsed_without_reading_the_text():
    plan = ExecutionPlan.model_validate(_plan([_subtask("s1", []), _subtask("s2", ["s1"])]))
    # text that no parser could read: decompose must use the provider-parsed object
    llm = RecordingLLM(CannedLLM(LLMResponse(text="<not json>", parsed=plan)))

    result = await decompose(llm, "compare Chroma and Qdrant")

    assert result is plan
    (call,) = llm.calls
    assert call["agent"] == "supervisor"
    assert call["output_schema"] is ExecutionPlan
    assert not call.get("tools")


async def test_an_out_of_range_review_score_is_retried_then_accepted(tmp_path):
    _fixture(tmp_path, "review", "reviewer", json.dumps({"score": 7, "feedback": "flawless"}),
             match=[REVIEW_MARKER])
    _fixture(tmp_path, "review_retry", "reviewer", json.dumps({"score": 4, "feedback": "cite sources"}),
             match=[REVIEW_MARKER, "Your previous verdict was invalid"])
    llm = RecordingLLM(MockLLMClient(tmp_path))

    verdict = await Reviewer(llm).review(
        "write a memo", "markdown", "MEMO-TEXT", producer_provider="openai"
    )

    assert verdict == ReviewVerdict(score=4, feedback="cite sources")
    assert len(llm.calls) == 2
    assert all(call["output_schema"] is ReviewVerdict for call in llm.calls)
    # the retry keeps the reviewer on the other provider
    assert all(call["producer_provider"] == "openai" for call in llm.calls)
    assert "less than or equal to 5" in llm.calls[1]["prompt"]
