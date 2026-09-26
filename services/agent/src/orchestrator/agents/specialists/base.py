"""Specialist agents: a bounded loop over native tool calls.

Each turn the model either replies in plain text (the deliverable) or asks for
one or more tool calls through the provider's tool-calling API. Every call of a
turn is answered with exactly one ToolMessage before the model's next turn:

1. The human-approval gate rules on each sensitive call before any call of the
   turn runs. It can pause the whole run (``interrupt()``), so it is asked one
   call at a time.
2. The remaining calls run concurrently, each in a worker thread, because the
   registry and the tools are synchronous.
3. The answers go back in call order. A ToolError is an answer the model sees
   (``status="error"``), not a failed attempt.

An attempt fails only when the turn budget runs out (``max_tool_iterations``; a
batch of calls is one turn) or when something other than a ToolError escapes.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from orchestrator.agents.base import BaseAgent
from orchestrator.config import get_settings
from orchestrator.llm.clients import LLMClient
from orchestrator.llm.mock import ToolCall
from orchestrator.tools.base import ToolContext, ToolError
from orchestrator.tools.registry import ToolRegistry

# Stable prompt marker; mock fixtures and tests match on it.
FEEDBACK_MARKER = "Reviewer feedback:"

SYSTEM_PROMPT = """You are the {name} specialist in a multi-agent system. {role}
Use the tools provided when they help. When you are finished, reply with the
deliverable as plain text and no tool calls."""

TASK_PROMPT = """Subtask: {description}
Expected output format: {expected_format}
Inputs from completed subtasks:
{inputs}{feedback_block}"""

# gate(tool_name, arguments, iteration, transcript) -> {"action", "payload", "notes"}
Gate = Callable[[str, dict, int, list[str]], dict | None]


class SpecialistError(RuntimeError):
    pass


@dataclass
class SpecialistResult:
    output: str
    tool_calls: list[str] = field(default_factory=list)


def _answer(call: ToolCall, content: str, *, error: bool = False) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=call.id, name=call.name, status="error" if error else "success")


class SpecialistAgent(BaseAgent):
    ROLE = ""

    def __init__(self, llm: LLMClient, registry: ToolRegistry):
        super().__init__(llm)
        self.registry = registry

    def _opening(self, spec: dict, inputs: dict[str, str], feedback: str | None) -> list[BaseMessage]:
        task = TASK_PROMPT.format(
            description=spec["description"],
            expected_format=spec.get("expected_output_format", "plain text"),
            inputs="\n".join(f"[{sid}] {text}" for sid, text in sorted(inputs.items())) or "(none)",
            feedback_block=f"\n\n{FEEDBACK_MARKER} {feedback}" if feedback else "",
        )
        return [SystemMessage(content=SYSTEM_PROMPT.format(name=self.name, role=self.ROLE)), HumanMessage(content=task)]

    async def execute(
        self, spec: dict, inputs: dict[str, str], feedback: str | None, ctx: ToolContext, gate: Gate | None = None
    ) -> SpecialistResult:
        """Run the loop. ``gate`` is consulted before every sensitive call and returns
        a human decision: approve (run it), modify (run with ``payload["arguments"]``),
        reject (answer with the denial), or take_over (``payload["output"]`` is the
        result). Its ``transcript`` is a readable history for the approval context;
        the model never sees it."""
        messages = self._opening(spec, inputs, feedback)
        tools = self.registry.tool_definitions_for(self.name)
        budget = get_settings().max_tool_iterations
        transcript: list[str] = []
        executed: list[str] = []
        for iteration in range(budget):
            response = await self.chat(messages, tools=tools)
            if not response.tool_calls:
                return SpecialistResult(output=response.text, tool_calls=executed)
            messages.append(AIMessage(content=response.text, tool_calls=[
                {"id": call.id, "name": call.name, "args": call.arguments, "type": "tool_call"}
                for call in response.tool_calls
            ]))

            # 1. The gate rules on every sensitive call before anything of this turn runs.
            arguments = {call.id: call.arguments for call in response.tool_calls}
            decided: dict[str, tuple[ToolMessage, str | None]] = {}  # answered by a human, with the name to list
            for call in response.tool_calls:
                if gate is None or not self.registry.is_sensitive_call(call.name, call.arguments):
                    continue
                decision = gate(call.name, call.arguments, iteration, list(transcript)) or {}
                action, payload = decision.get("action", "approve"), decision.get("payload") or {}
                if action == "reject":
                    denial = f"denied by human reviewer: {decision.get('notes') or 'not permitted'}"
                    decided[call.id] = (_answer(call, denial, error=True), None)
                elif action == "take_over":
                    human = _answer(call, f"(human-provided) {payload.get('output', '')}")
                    decided[call.id] = (human, f"{call.name}[human]")
                elif action == "modify":
                    arguments[call.id] = payload.get("arguments", call.arguments)

            # 2. The other calls run concurrently, each in a worker thread.
            to_run = [call for call in response.tool_calls if call.id not in decided]
            outcomes = await asyncio.gather(
                *(asyncio.to_thread(self.registry.invoke, call.name, arguments[call.id], ctx) for call in to_run),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException) and not isinstance(outcome, ToolError):
                    raise outcome  # not the model's to handle: the attempt fails
            ran = {call.id: outcome for call, outcome in zip(to_run, outcomes)}

            # 3. Exactly one answer per call, in call order, before the next turn.
            for call in response.tool_calls:
                if call.id in decided:
                    answer, listed = decided[call.id]
                else:
                    outcome = ran[call.id]
                    if isinstance(outcome, ToolError):
                        answer = _answer(call, str(outcome), error=True)
                    else:
                        answer = _answer(call, json.dumps(outcome)[:2000])
                    listed = call.name
                messages.append(answer)
                if listed:
                    executed.append(listed)
                verb = "failed" if answer.status == "error" else "returned"
                transcript.append(f"-> {call.name} {verb}: {answer.content}")
        raise SpecialistError(f"{self.name} exceeded {budget} tool iterations")
