"""What the planner can assign: the capability manifest, and the lint a plan must pass.

Both read the live tool registry, so they describe this deployment: a tool left out
of the registry (``code_exec`` without a sandbox, ``web_search`` without a search
provider, the ``rag_*`` tools without MCP) is absent from the manifest, and a plan
that names it fails the lint. A specialist is only ever offered the tools it owns,
so a subtask given to the wrong specialist would not fail on its own: it would be
answered from the model's memory. The lint catches that before anything runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from orchestrator.config import get_settings
from orchestrator.llm.router import SPECIALISTS
from orchestrator.planning.schemas import ExecutionPlan
from orchestrator.tools.base import ToolSpec
from orchestrator.tools.registry import ToolRegistry

# Specialists whose work is tool calls. A writing subtask may work from its inputs alone.
TOOL_BOUND = ("research", "analysis", "code")


@dataclass(frozen=True)
class ToolEntry:
    name: str
    summary: str  # the description's first sentence
    limit: int  # calls per task


@dataclass(frozen=True)
class Capability:
    specialist: str
    tools: tuple[ToolEntry, ...]


def summary(description: str) -> str:
    """The first sentence of a tool description, without its full stop."""
    first = " ".join(description.split()).split(". ", 1)[0]
    return first.removesuffix(".")


def _available(registry: ToolRegistry) -> dict[str, ToolSpec]:
    """Every registered tool some specialist owns, by name."""
    return {spec.name: spec for specialist in SPECIALISTS for spec in registry.tools_for(specialist)}


def manifest(registry: ToolRegistry) -> list[Capability]:
    """Each specialist with the tools it owns in this deployment, in a stable order."""
    return [
        Capability(
            specialist=specialist,
            tools=tuple(
                ToolEntry(name=spec.name, summary=summary(spec.description), limit=spec.rate_limit)
                for spec in sorted(registry.tools_for(specialist), key=lambda spec: spec.name)
            ),
        )
        for specialist in SPECIALISTS
    ]


def render_manifest(capabilities: list[Capability]) -> str:
    """The manifest as a table, one row per specialist and tool."""
    rows = ["| Specialist | Tool | What it does | Calls per task |", "|---|---|---|---|"]
    for capability in capabilities:
        if not capability.tools:
            rows.append(f"| {capability.specialist} | (none) | Works only from its inputs | |")
        for tool in capability.tools:
            rows.append(f"| {capability.specialist} | {tool.name} | {tool.summary} | {tool.limit} |")
    return "\n".join(rows)


def lint(plan: ExecutionPlan, registry: ToolRegistry, max_subtasks: int) -> list[str]:
    """What is wrong with ``plan`` for this registry, one message per problem; empty if nothing.

    Pure: it reads the plan and the registry and changes neither. Each message says
    what to change, because the messages go back to the planner on its retry.
    """
    available = _available(registry)
    errors: list[str] = []
    if len(plan.subtasks) > max_subtasks:
        errors.append(
            f"The plan has {len(plan.subtasks)} subtasks; at most {max_subtasks} are allowed. "
            "Give work that one specialist's tools can do to a single subtask."
        )
    for subtask in plan.subtasks:
        owned = sorted(spec.name for spec in registry.tools_for(subtask.specialist))
        if not subtask.tools and subtask.specialist in TOOL_BOUND:
            errors.append(
                f"{subtask.id}: list the tools this {subtask.specialist} subtask will call "
                f"({subtask.specialist} owns: {', '.join(owned) or 'no tools'})."
            )
        for tool in dict.fromkeys(subtask.tools):  # each name once, in the plan's order
            if tool not in available:
                errors.append(
                    f"{subtask.id}: there is no tool {tool!r} in this deployment "
                    f"(available: {', '.join(sorted(available))})."
                )
            elif subtask.specialist not in available[tool].owners:
                owners = sorted(owner for owner in available[tool].owners if owner in SPECIALISTS)
                errors.append(
                    f"{subtask.id}: {tool!r} belongs to {' or '.join(owners)}, not {subtask.specialist}; "
                    "give this subtask to a specialist that owns all of its tools."
                )
    return errors


def check(plan: ExecutionPlan, registry: ToolRegistry) -> tuple[list[str], bool]:
    """The lint's findings under ``MAX_PLAN_SUBTASKS``, and whether ``PLAN_LINT`` makes them reject the plan."""
    settings = get_settings()
    errors = lint(plan, registry, settings.max_plan_subtasks)
    return errors, bool(errors) and settings.plan_lint == "enforce"
