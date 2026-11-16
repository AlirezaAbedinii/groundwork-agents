"""Task decomposition engine: request → validated ExecutionPlan.

The planner is shown what this deployment can do, as a table of each specialist's
tools read from the live registry (planning/capabilities.py), and asked for the
fewest subtasks that do the work. The plan comes back as native structured output
constrained to the ExecutionPlan schema. Two kinds of check follow: the ones only
Pydantic can run (unique ids, known dependencies, no cycles), and the plan lint
(every tool exists and belongs to its subtask's specialist, a tool-bound subtask
names its tools, the plan is within the size limit). A plan that fails either is
retried exactly once with the errors appended to the prompt. With
``plan_lint="off"`` the lint's findings are reported but reject nothing, and
without ``lint_plans`` (the rules of a task from before the lint, orchestrator/rules.py)
the lint does not run.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from orchestrator.config import get_settings
from orchestrator.llm.clients import LLMClient
from orchestrator.llm.structured import StructuredOutputError
from orchestrator.planning.capabilities import check, manifest, render_manifest
from orchestrator.planning.schemas import ExecutionPlan
from orchestrator.tools.registry import ToolRegistry

# Stable marker; mock fixtures and tests match on it.
PLAN_MARKER = "Create an execution plan"

PLAN_PROMPT = """You are the supervisor of a multi-agent system. {marker} for the task below.

Task: {request}

Each specialist can call only its own tools. This table lists every tool in this
deployment and who owns it; a tool that is not in the table does not exist here.

{manifest}

Every specialist is shown the task exactly as written above, together with its own
subtask.

Rules:
1. Use the fewest subtasks that do the work, and never more than {max_subtasks}.
2. If one specialist owns every tool the task needs, plan exactly one subtask for it.
3. Add a subtask only for work that needs another specialist's tools. Do not add a
   subtask that only reviews, checks, finalises or analyses another subtask's output.
4. In each subtask's tools, list the tools it will call. Give a subtask only to a
   specialist that owns all of them. A research, analysis or code subtask lists at
   least one tool; a writing subtask may list none and work from its inputs.
5. Copy file names, input values and constraints from the task into the description
   of the subtask that needs them, word for word.
6. Subtask ids are unique. depends_on may only reference other subtask ids and must
   form no cycles. Subtasks that do not depend on each other run in parallel.
"""

RETRY_SUFFIX = """

Your previous plan was invalid and has been rejected. Validation error:
{error}

Produce a corrected plan that fixes this error.
"""


class PlanValidationError(ValueError):
    """No acceptable plan after the retry; ``lint_errors`` are the lint's findings, if any."""

    def __init__(self, message: str, lint_errors: list[str] | None = None):
        super().__init__(message)
        self.lint_errors = lint_errors or []


class _LintFailure(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class Planned:
    plan: ExecutionPlan
    # every lint finding while planning: a rejected first plan's, and with plan_lint="off"
    # the accepted plan's (reported, not enforced); empty when the first plan was clean
    lint_errors: list[str]


async def decompose(
    llm: LLMClient, request: str, registry: ToolRegistry, memories: str | None = None, *, lint_plans: bool = True
) -> Planned:
    settings = get_settings()
    prompt = PLAN_PROMPT.format(
        marker=PLAN_MARKER,
        request=request,
        manifest=render_manifest(manifest(registry)),
        max_subtasks=settings.max_plan_subtasks,
    )
    if memories:
        prompt = f"{prompt}\n{memories}\n"
    found: list[str] = []

    async def attempt(text: str) -> ExecutionPlan:
        plan = await _plan(llm, text)
        if not lint_plans:
            return plan
        errors, rejected = check(plan, registry)
        found.extend(errors)
        if rejected:
            raise _LintFailure(errors)
        return plan

    try:
        return Planned(await attempt(prompt), found)
    except (StructuredOutputError, _LintFailure) as first_error:
        try:
            return Planned(await attempt(prompt + RETRY_SUFFIX.format(error=first_error)), found)
        except (StructuredOutputError, _LintFailure) as second_error:
            raise PlanValidationError(f"Plan invalid after retry: {second_error}", found) from second_error


async def _plan(llm: LLMClient, prompt: str) -> ExecutionPlan:
    response = await llm.chat("supervisor", [HumanMessage(content=prompt)], output_schema=ExecutionPlan)
    return response.parsed
