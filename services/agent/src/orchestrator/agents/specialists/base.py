"""Specialist agents: a bounded loop over native tool calls.

Each turn the model either replies in plain text (the deliverable) or asks for
one or more tool calls through the provider's tool-calling API. Every call of a
turn is answered with exactly one ToolMessage before the model's next turn:

0. A call identical to one that reached the registry earlier in this attempt
   (same tool, same arguments), or to an earlier call of the same turn, is
   answered with that answer again, marked as a repeat. It reaches neither the
   gate nor the registry. Answers a human gave are not reused across turns: a
   repeated sensitive call goes back to the reviewer. A call of a tool that
   ``changes_state`` makes every earlier answer stale once it has run.
1. The human-approval gate rules on each sensitive call before any call of the
   turn runs. It can pause the whole run (``interrupt()``), so it is asked one
   call at a time.
2. The remaining calls run concurrently, each in a worker thread, because the
   registry and the tools are synchronous.
3. The answers go back in call order, each result as JSON within its tool's
   ``result_budget_chars`` (tools/serialize.py). A ToolError is an answer the
   model sees (``status="error"``), not a failed attempt. A tool that answers
   with its rate limit is spent for the task: the answer says so, and the tool
   is not offered again in this attempt.

Each call that reached the registry or the human gate leaves a receipt: the tool,
whether it returned a result (``success``), failed (``error``) or was denied by a
human (``denied``), and the first RECEIPT_CHARS characters of its answer, with
credential-shaped text replaced (``redact``): the reviewer may run on another
provider than the one that called the tool. A repeat leaves none, since nothing
ran; the receipt of the call it repeats stands. The graph checks the receipts
before review, and the reviewer reads them.

Before the last turn the model is told that it is the last. An attempt fails
only when the turn budget runs out (``max_tool_iterations``; a batch of calls is
one turn), with a digest of its last tool exchanges for the retry, or when
something other than a ToolError escapes.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from orchestrator.agents.base import BaseAgent
from orchestrator.config import get_settings
from orchestrator.llm.clients import LLMClient
from orchestrator.llm.mock import ToolCall
from orchestrator.tools.base import (
    RateLimitExceededError,
    ToolContext,
    ToolError,
    ToolPermissionError,
    UnknownToolError,
)
from orchestrator.tools.registry import ToolRegistry
from orchestrator.tools.serialize import render_result

# Stable prompt markers; mock fixtures and tests match on them.
REQUEST_MARKER = "User's request (context; do only your subtask):"
FEEDBACK_MARKER = "Reviewer feedback:"
REPEAT_PREFIX = "Identical to an earlier call in this attempt; result unchanged:"
UNAVAILABLE = "this tool is unavailable for the rest of this task"
LAST_TURN = "This is your last turn. Reply with the deliverable. State plainly anything you could not do."

# How much of each answer a receipt keeps
RECEIPT_CHARS = 200

# Credential-shaped text a receipt never shows. A pattern list, not a guarantee: it covers
# named secrets, passwords in URLs, bearer tokens and common provider key formats.
REDACTED = "[redacted]"
_SECRET_NAME = (
    r"(?:secret|secret[_-]?key|password|passwd|api[_-]?key|access[_-]?key|private[_-]?key|account[_-]?key"
    r"|(?:access|auth|refresh|session|api)[_-]?token)"
)
SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"), REDACTED),
    # scheme://user:password@host keeps the user
    (re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]{0,20}://[^\s/:@]+):[^\s/@]+@"), rf"\1:{REDACTED}@"),
    (re.compile(r"(?i)\b(bearer\s+)(?=[\w.~+/=-]*\d)[\w.~+/=-]{8,}"), rf"\1{REDACTED}"),
    # NAME=value, NAME: value and "NAME": "value", with or without a prefix (OPENAI_API_KEY, db_password)
    (re.compile(rf"(?i)(?<![\w-])([\w-]*{_SECRET_NAME}[\"']?\s*[:=]\s*[\"']?)[^\s\"',;&}}]+"), rf"\1{REDACTED}"),
    # provider key formats that appear without a name
    (re.compile(r"\b(?:sk-[\w-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|AKIA[0-9A-Z]{16}"
                r"|xox[abposr]-[A-Za-z0-9-]{10,}|AIza[\w-]{30,}|eyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,})"), REDACTED),
]

# What a failed attempt's retry is told: its last tool exchanges, in at most this many characters.
DIGEST_LINES = 3
DIGEST_CHARS = 600

SYSTEM_PROMPT = """You are the {name} specialist in a multi-agent system. {role}
Use the tools provided when they help. When you are finished, reply with the
deliverable as plain text and no tool calls."""

# The request comes first and verbatim: the subtask description is the planner's
# paraphrase of it, and whatever detail the paraphrase drops is otherwise lost.
TASK_PROMPT = REQUEST_MARKER + """ {request}

Subtask: {description}
Expected output format: {expected_format}
Inputs from completed subtasks:
{inputs}{feedback_block}"""

# gate(tool_name, arguments, iteration, transcript) -> {"action", "payload", "notes"}
Gate = Callable[[str, dict, int, list[str]], dict | None]


class SpecialistError(RuntimeError):
    """An attempt that ran out of turns; ``digest`` is what its last tool exchanges were."""

    def __init__(self, message: str, digest: str = ""):
        super().__init__(message)
        self.digest = digest


@dataclass
class SpecialistResult:
    output: str
    tool_calls: list[str] = field(default_factory=list)
    # one per call that ran or that a human ruled on, in call order: {"tool", "status", "digest"}
    receipts: list[dict] = field(default_factory=list)


def redact(text: str) -> str:
    """``text`` with its credential-shaped parts replaced by [redacted]."""
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def receipt(tool: str, status: str, answer: str) -> dict:
    """What a call left behind: ``status`` is success, error or denied; ``digest`` is the
    start of the answer on one line, where the outcome and an error's reason are. One line,
    so a tool's text cannot start a line of the review prompt; redacted before the cut, so
    a key that straddles it is still found."""
    return {"tool": tool, "status": status, "digest": redact(" ".join(answer.split()))[:RECEIPT_CHARS]}


def _answer(call: ToolCall, content: str, *, error: bool = False) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=call.id, name=call.name, status="error" if error else "success")


def _call_key(name: str, arguments: dict) -> str:
    return name + json.dumps(arguments, sort_keys=True)


def _refused(outcome: object) -> bool:
    """Whether the registry turned the call away before it ran. Invalid arguments count as run:
    treating them so can only cost a run later, never a stale answer."""
    return isinstance(outcome, UnknownToolError | ToolPermissionError | RateLimitExceededError)


def _digest(transcript: list[str]) -> str:
    """The last DIGEST_LINES transcript lines, one line each, together at most DIGEST_CHARS.

    Each line gets an equal share and keeps its start, where the tool, the
    outcome and the start of an error message are."""
    lines = [" ".join(line.split()) for line in transcript[-DIGEST_LINES:]]
    share = (DIGEST_CHARS - (len(lines) - 1)) // max(len(lines), 1)  # less the newlines between them
    return "\n".join(line if len(line) <= share else line[: share - 1] + "…" for line in lines)


class SpecialistAgent(BaseAgent):
    ROLE = ""

    def __init__(self, llm: LLMClient, registry: ToolRegistry):
        super().__init__(llm)
        self.registry = registry

    def _opening(
        self, spec: dict, inputs: dict[str, str], feedback: str | None, request: str = ""
    ) -> list[BaseMessage]:
        task = TASK_PROMPT.format(
            request=request.strip() or "(none)",
            description=spec["description"],
            expected_format=spec.get("expected_output_format", "plain text"),
            inputs="\n".join(f"[{sid}] {text}" for sid, text in sorted(inputs.items())) or "(none)",
            feedback_block=f"\n\n{FEEDBACK_MARKER} {feedback}" if feedback else "",
        )
        return [SystemMessage(content=SYSTEM_PROMPT.format(name=self.name, role=self.ROLE)), HumanMessage(content=task)]

    async def execute(
        self,
        spec: dict,
        inputs: dict[str, str],
        feedback: str | None,
        ctx: ToolContext,
        gate: Gate | None = None,
        *,
        request: str = "",
    ) -> SpecialistResult:
        """Run the loop. ``request`` is the task's request as the user wrote it, shown
        above the subtask as context. ``gate`` is consulted before every sensitive call
        and returns a human decision: approve (run it), modify (run with
        ``payload["arguments"]``), reject (answer with the denial), or take_over
        (``payload["output"]`` is the result). Its ``transcript`` is a readable history
        for the approval context; the model never sees it."""
        messages = self._opening(spec, inputs, feedback, request)
        budget = get_settings().max_tool_iterations
        transcript: list[str] = []
        executed: list[str] = []
        receipts: list[dict] = []
        answered: dict[str, ToolMessage] = {}  # each call that reached the registry, by the arguments it ran with
        withdrawn: set[str] = set()  # tools that reached their rate limit
        for iteration in range(budget):
            if iteration == budget - 1:
                messages.append(HumanMessage(content=LAST_TURN))
            # read each turn, so the calls left on each tool are current
            tools = self.registry.tool_definitions_for(self.name, ctx.task_id)
            offered = [tool for tool in tools if tool["function"]["name"] not in withdrawn]
            response = await self.chat(messages, tools=offered)
            if not response.tool_calls:
                return SpecialistResult(output=response.text, tool_calls=executed, receipts=receipts)
            messages.append(AIMessage(content=response.text, tool_calls=[
                {"id": call.id, "name": call.name, "args": call.arguments, "type": "tool_call"}
                for call in response.tool_calls
            ]))

            # 0. Repeats of an answered call, or of an earlier call of this turn, are answered from it.
            repeats: dict[str, str] = {}  # call id -> key of the call it repeats
            firsts: set[str] = set()
            for call in response.tool_calls:
                key = _call_key(call.name, call.arguments)
                if key in answered or key in firsts:
                    repeats[call.id] = key
                else:
                    firsts.add(key)
            fresh = [call for call in response.tool_calls if call.id not in repeats]

            # 1. The gate rules on every sensitive call before anything of this turn runs.
            arguments = {call.id: call.arguments for call in response.tool_calls}
            decided: dict[str, tuple[ToolMessage, str | None]] = {}  # answered by a human, with the name to list
            denied: set[str] = set()  # call ids a human rejected
            for call in fresh:
                if gate is None or not self.registry.is_sensitive_call(call.name, call.arguments):
                    continue
                decision = gate(call.name, call.arguments, iteration, list(transcript)) or {}
                action, payload = decision.get("action", "approve"), decision.get("payload") or {}
                if action == "reject":
                    denial = f"denied by human reviewer: {decision.get('notes') or 'not permitted'}"
                    decided[call.id] = (_answer(call, denial, error=True), None)
                    denied.add(call.id)
                elif action == "take_over":
                    human = _answer(call, f"(human-provided) {payload.get('output', '')}")
                    decided[call.id] = (human, f"{call.name}[human]")
                elif action == "modify":
                    arguments[call.id] = payload.get("arguments", call.arguments)

            # 2. The other calls run concurrently, each in a worker thread.
            to_run = [call for call in fresh if call.id not in decided]
            outcomes = await asyncio.gather(
                *(asyncio.to_thread(self.registry.invoke, call.name, arguments[call.id], ctx) for call in to_run),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException) and not isinstance(outcome, ToolError):
                    raise outcome  # not the model's to handle: the attempt fails
            ran = {call.id: outcome for call, outcome in zip(to_run, outcomes)}

            # A call that changes state and ran makes every earlier answer stale. The calls of
            # a turn run at the same time, so a turn's answers are kept only if none of its
            # calls changed state, or the one that did ran alone (then an identical call
            # straight after it is still answered from it).
            ran_here = [call for call in to_run if not _refused(ran[call.id])]
            changed = any(self.registry.get(call.name).changes_state for call in ran_here)
            if changed:
                answered.clear()
            keep = not changed or len(ran_here) == 1

            # 3. Exactly one answer per call, in call order, before the next turn.
            this_turn: dict[str, ToolMessage] = {}  # this turn's answers, by the arguments the model asked with
            for call in response.tool_calls:
                if call.id in repeats:
                    key = repeats[call.id]
                    earlier = this_turn[key] if key in this_turn else answered[key]
                    answer = _answer(call, f"{REPEAT_PREFIX} {earlier.content}", error=earlier.status == "error")
                    listed = None  # nothing ran
                elif call.id in decided:
                    answer, listed = decided[call.id]
                    receipts.append(receipt(call.name, "denied" if call.id in denied else answer.status, answer.content))
                else:
                    outcome = ran[call.id]
                    if isinstance(outcome, RateLimitExceededError):
                        withdrawn.add(call.name)
                        answer = _answer(call, f"{outcome}; {UNAVAILABLE}.", error=True)
                    elif isinstance(outcome, ToolError):
                        answer = _answer(call, str(outcome), error=True)
                    else:
                        answer = _answer(call, render_result(outcome, self.registry.get(call.name).result_budget_chars))
                    listed = call.name
                    receipts.append(receipt(call.name, answer.status, answer.content))
                    if keep:
                        answered[_call_key(call.name, arguments[call.id])] = answer
                if call.id not in repeats:
                    this_turn.setdefault(_call_key(call.name, call.arguments), answer)
                messages.append(answer)
                if listed:
                    executed.append(listed)
                verb = "failed" if answer.status == "error" else "returned"
                transcript.append(f"-> {call.name} {verb}: {answer.content}")
        raise SpecialistError(f"{self.name} exceeded {budget} tool iterations", digest=_digest(transcript))
