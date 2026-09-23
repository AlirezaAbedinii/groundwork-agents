"""Supervisor agent: task decomposition and final synthesis."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from orchestrator.agents.base import BaseAgent
from orchestrator.planning.decomposer import decompose
from orchestrator.planning.schemas import ExecutionPlan

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

    async def plan(self, request: str, memories: str | None = None) -> ExecutionPlan:
        return await decompose(self.llm, request, memories=memories)

    async def synthesize(self, request: str, outputs: dict[str, str]) -> str:
        rendered = "\n".join(f"[{sid}]\n{text}\n" for sid, text in sorted(outputs.items()))
        prompt = SYNTH_PROMPT.format(marker=SYNTH_MARKER, request=request, outputs=rendered)
        return (await self.chat([HumanMessage(content=prompt)])).text
