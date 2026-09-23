"""Task decomposition engine: request → validated ExecutionPlan.

The plan comes back as native structured output constrained to the
ExecutionPlan schema. The checks only Pydantic can run (unique ids, known
dependencies, no cycles) still apply, so a plan that fails them is retried
exactly once with the validation error appended to the prompt.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from orchestrator.llm.clients import LLMClient
from orchestrator.llm.router import SPECIALISTS
from orchestrator.llm.structured import StructuredOutputError
from orchestrator.planning.schemas import ExecutionPlan

# Stable marker; mock fixtures and tests match on it.
PLAN_MARKER = "Create an execution plan"

PLAN_PROMPT = """You are the supervisor of a multi-agent system. {marker} for the task below.

Task: {request}

Available specialists: {specialists} —
research (web research, source gathering), analysis (data extraction/computation),
writing (drafts, summaries, memos), code (writes and runs code).

Decompose the task into subtasks.

Rules: subtask ids are unique; depends_on may only reference other subtask ids and
must form no cycles; make independent subtasks so they can run in parallel.
"""

RETRY_SUFFIX = """

Your previous plan was invalid and has been rejected. Validation error:
{error}

Produce a corrected plan that fixes this error.
"""


class PlanValidationError(ValueError):
    pass


async def decompose(llm: LLMClient, request: str, memories: str | None = None) -> ExecutionPlan:
    prompt = PLAN_PROMPT.format(marker=PLAN_MARKER, request=request, specialists=", ".join(SPECIALISTS))
    if memories:
        prompt = f"{prompt}\n{memories}\n"
    try:
        return await _plan(llm, prompt)
    except StructuredOutputError as first_error:
        try:
            return await _plan(llm, prompt + RETRY_SUFFIX.format(error=first_error))
        except StructuredOutputError as second_error:
            raise PlanValidationError(
                f"Plan invalid after retry: {second_error}"
            ) from second_error


async def _plan(llm: LLMClient, prompt: str) -> ExecutionPlan:
    response = await llm.chat("supervisor", [HumanMessage(content=prompt)], output_schema=ExecutionPlan)
    return response.parsed
