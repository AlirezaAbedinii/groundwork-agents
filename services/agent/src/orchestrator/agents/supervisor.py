"""Supervisor agent: task decomposition and final synthesis."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from orchestrator.agents.base import BaseAgent
from orchestrator.planning.decomposer import Planned, decompose
from orchestrator.tools.registry import ToolRegistry

# Stable marker; mock fixtures and tests match on it.
SYNTH_MARKER = "Synthesize the final deliverable"

SYNTH_PROMPT = """{marker} for the task below from the completed subtask outputs.

Task: {request}

Completed subtask outputs:
{outputs}

Write the final deliverable for the user. Respond with the deliverable text only.
"""


class Supervisor(BaseAgent):
    name = "supervisor"

    async def plan(
        self, request: str, registry: ToolRegistry, memories: str | None = None, *, lint_plans: bool = True
    ) -> Planned:
        """A plan for ``request`` from the tools in ``registry``, with what the lint found."""
        return await decompose(self.llm, request, registry, memories=memories, lint_plans=lint_plans)

    async def synthesize(self, request: str, outputs: dict[str, str]) -> str:
        rendered = "\n".join(f"[{sid}]\n{text}\n" for sid, text in sorted(outputs.items()))
        prompt = SYNTH_PROMPT.format(marker=SYNTH_MARKER, request=request, outputs=rendered)
        return (await self.chat([HumanMessage(content=prompt)])).text
