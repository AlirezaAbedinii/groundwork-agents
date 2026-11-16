"""What the planner can assign: the capability manifest.

The manifest reads the live tool registry, so it describes this deployment: a tool
left out of the registry (``code_exec`` without a sandbox, ``web_search`` without a
search provider, the ``rag_*`` tools without MCP) is absent from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from orchestrator.llm.router import SPECIALISTS
from orchestrator.tools.registry import ToolRegistry


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

