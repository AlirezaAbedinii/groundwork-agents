"""Supervisor agent: task decomposition and final synthesis.

Synthesis is shown a manifest of the run: each subtask, the outcome of each tool call
it made (from the task's invocation log), what it reported, and the files the plan must
deliver with whether each was written. It is told to use only those reports, so the
final answer adds nothing a subtask did not find.
"""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.messages import HumanMessage

from orchestrator.agents.base import BaseAgent
from orchestrator.planning.decomposer import Planned, decompose
from orchestrator.planning.schemas import ExecutionPlan
from orchestrator.tools.base import InvocationRecord
from orchestrator.tools.file_io import WRITE_SPEC, written_paths
from orchestrator.tools.registry import ToolRegistry

# Stable marker; mock fixtures and tests match on it.
SYNTH_MARKER = "Synthesize the final deliverable"

# What synthesis says, word for word, where a subtask found the documentation silent.
NOT_COVERED = "the documentation does not cover this"

SYNTH_PROMPT = """{marker} for the task below from what its subtasks reported.

Task: {request}

What each subtask did and reported:

{subtasks}

Files the task must deliver:
{deliverables}

Write the final answer for the user from the reports above, under these rules:
1. Use only what the subtasks reported. Add no fact, number, command, setting, example
   or recommendation of your own, even one you believe is correct.
2. Where a subtask reports that the documentation does not cover something, or that it
   is not documented, say "{not_covered}" for that point and give no answer to it
   from anywhere else.
3. Name each file listed above by its exact path, and say a file was written only if it
   is marked written.
4. If the reports leave part of the task undone, say which part.

Respond with the final answer only.
"""

_OUTCOMES = {"success": "succeeded", "failure": "failed", "rejected": "refused", "rate_limited": "refused (limit)"}


def _tool_outcomes(records: list[InvocationRecord]) -> str:
    """Each distinct call outcome once, in first-seen order, with a count when it repeats."""
    counts: dict[str, int] = {}
    for record in records:
        path = record.arguments.get("path") if record.tool_name == WRITE_SPEC.name else None
        label = f"{record.tool_name} `{path}`" if isinstance(path, str) else record.tool_name
        outcome = f"{label} {_OUTCOMES.get(record.status, record.status)}"
        counts[outcome] = counts.get(outcome, 0) + 1
    return "; ".join(f"{outcome} x{n}" if n > 1 else outcome for outcome, n in counts.items()) or "none"


def synthesis_manifest(
    plan: ExecutionPlan, results: dict[str, dict], invocations: Iterable[InvocationRecord]
) -> tuple[str, str]:
    """The subtasks section and the deliverables section of the synthesis prompt."""
    invocations = list(invocations)
    by_subtask: dict[str, list[InvocationRecord]] = {}
    for record in invocations:
        by_subtask.setdefault(record.subtask_id or "", []).append(record)
    blocks = []
    for subtask in plan.subtasks:
        result = results.get(subtask.id) or {}
        completed = result.get("status") == "completed"
        reporter = "a human reviewer, who took over" if result.get("human_provided") else subtask.specialist
        report = (result.get("output") or "(empty)") if completed else "(no completed output)"
        blocks.append(
            f"[{subtask.id}] {subtask.specialist}: {subtask.description}\n"
            f"Tool calls: {_tool_outcomes(by_subtask.get(subtask.id, []))}\n"
            f"Reported by {reporter}:\n{report}"
        )
    written = written_paths(invocations)
    deliverables = "\n".join(
        f"- `{d.path}` (by {d.producer}): {'written' if d.path in written else 'NOT written'}"
        for d in plan.deliverables
    ) or "(none)"
    return "\n\n".join(blocks), deliverables


class Supervisor(BaseAgent):
    name = "supervisor"

    async def plan(
        self,
        request: str,
        registry: ToolRegistry,
        memories: str | None = None,
        *,
        lint_plans: bool = True,
        check_deliverables: bool = True,
    ) -> Planned:
        """A plan for ``request`` from the tools in ``registry``, with what the lint found."""
        return await decompose(
            self.llm, request, registry, memories=memories,
            lint_plans=lint_plans, check_deliverables=check_deliverables,
        )

    async def synthesize(
        self, request: str, plan: ExecutionPlan, results: dict[str, dict], invocations: Iterable[InvocationRecord]
    ) -> str:
        """The final answer from the subtasks' reports and the task's invocation log."""
        subtasks, deliverables = synthesis_manifest(plan, results, invocations)
        prompt = SYNTH_PROMPT.format(
            marker=SYNTH_MARKER, request=request, subtasks=subtasks, deliverables=deliverables, not_covered=NOT_COVERED
        )
        return (await self.chat([HumanMessage(content=prompt)])).text
