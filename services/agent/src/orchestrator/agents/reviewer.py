"""Reviewer agent: scores specialist outputs 1–5 with feedback.

Routed to a different provider than the producing agent (llm/router.py).

With evidence (REVIEW_EVIDENCE, the default) the reviewer also sees the user's
request, the tools the subtask lists, which inputs it was given, and the tool
receipts of the attempt, and it is told that a claim with no receipt behind it
is unsupported. Before any of that, ``required_tools_silent`` decides without a
model whether the attempt has anything to review at all: a subtask that lists
tools, none of which returned a result, is sent back with ``no_result_feedback``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from orchestrator.agents.base import BaseAgent
from orchestrator.config import get_settings
from orchestrator.llm.structured import StructuredOutputError

# Stable marker; mock fixtures and tests match on it.
REVIEW_MARKER = "Review the following specialist output"
REQUEST_HEADING = "User's request (context; review only this subtask):"
RECEIPTS_HEADING = "Tool receipts from this attempt (tool, status, start of the result):"
# What a subtask is told when none of its tools returned a result; tests and fixtures match on it.
NO_RESULT = "None of your required tools returned a result in this attempt"
# How many of the errors the feedback quotes, the most recent last
QUOTED_ERRORS = 3

# Without evidence the blocks are empty and the prompt is the one from before them.
REVIEW_PROMPT = """{marker} for quality before it is returned to the supervisor.
{request_block}
Subtask: {description}
Expected output format: {expected_format}
{evidence_block}
Specialist output:
---
{output}
---

Score the output 1-5 (5 = excellent, meets the subtask and format; 1 = unusable).
{rules_block}"""

REQUEST_BLOCK = """
{heading}
{request}
"""

EVIDENCE_BLOCK = """Tools listed for this subtask: {tools}
Inputs from completed subtasks: {inputs}

{heading}
{receipts}
"""

RULES_BLOCK = """Rules:
1. A claim with no receipt behind it is unsupported, unless it could have come from the
   request or the inputs. An output that states unsupported numbers, names or facts as
   findings scores below {threshold}: name them in your feedback. Saying plainly that
   something could not be found or retrieved is not a claim.
2. When the user's request restricts the sources, do not credit facts from your own
   knowledge, even ones you believe are true.
"""

RETRY_SUFFIX = """

Your previous verdict was invalid ({error}). Score the output again, from 1 to 5.
"""


class ReviewVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    feedback: str = Field("", description="What is wrong or missing, if anything")


@dataclass(frozen=True)
class Evidence:
    """What the reviewer is shown besides the subtask and its output."""

    request: str
    tools: list[str] = field(default_factory=list)  # the tools the subtask lists
    receipts: list[dict] = field(default_factory=list)  # the attempt's, as the specialist loop left them
    inputs: list[str] = field(default_factory=list)  # ids of the completed subtasks whose outputs it was given


def required_tools_silent(tools: list[str], receipts: list[dict]) -> bool:
    """Whether a subtask lists tools and none of them returned a result in the attempt.

    A call a human denied counts as answered: the subtask cannot do better by trying
    again, and the reviewer judges what it made of the denial. A subtask that lists
    no tools (writing from its inputs) is never silent."""
    listed = set(tools)
    return bool(listed) and not any(
        r["tool"] in listed and r["status"] in ("success", "denied") for r in receipts
    )


def no_result_feedback(tools: list[str], receipts: list[dict]) -> str:
    """The rework feedback for a silent attempt: the tools it lists and the errors they returned."""
    listed = list(dict.fromkeys(tools))
    errors = [r for r in receipts if r["tool"] in listed and r["status"] == "error"]
    names = ", ".join(listed)
    if not errors:
        one = len(listed) == 1
        called = f"you did not call {names}" if one else f"you called none of them ({names})"
        return (
            f"{NO_RESULT}: {called}. Use {'it' if one else 'them'}; if {'it cannot' if one else 'they cannot'} "
            "give what the subtask needs, say so plainly and do not estimate it."
        )
    quoted = errors[-QUOTED_ERRORS:]
    count = f"the last {len(quoted)} of {len(errors)}" if len(errors) > len(quoted) else str(len(errors))
    lines = "\n".join(f"- {r['tool']}: {r['digest']}" for r in quoted)
    return (
        f"{NO_RESULT} ({names}). Errors ({count}):\n{lines}\n"
        "Correct the calls and try again; if the data cannot be retrieved, say so plainly and do not estimate it."
    )


def _receipt_lines(receipts: list[dict]) -> str:
    if not receipts:
        return "- none: no tool was called"
    return "\n".join(f"- {r['tool']} {r['status']}: {r['digest']}" for r in receipts)


def review_prompt(description: str, expected_format: str, output: str, evidence: Evidence | None = None) -> str:
    """The reviewer's prompt; without ``evidence`` it is the prompt from before evidence."""
    request_block = evidence_block = rules_block = ""
    if evidence is not None:
        request_block = REQUEST_BLOCK.format(heading=REQUEST_HEADING, request=evidence.request.strip() or "(none)")
        evidence_block = EVIDENCE_BLOCK.format(
            tools=", ".join(evidence.tools) or "none",
            inputs=(", ".join(evidence.inputs) + " (their text is not shown)") if evidence.inputs else "none",
            heading=RECEIPTS_HEADING,
            receipts=_receipt_lines(evidence.receipts),
        )
        rules_block = RULES_BLOCK.format(threshold=get_settings().review_score_threshold)
    return REVIEW_PROMPT.format(
        marker=REVIEW_MARKER,
        request_block=request_block,
        description=description,
        expected_format=expected_format,
        evidence_block=evidence_block,
        output=output,
        rules_block=rules_block,
    )


class Reviewer(BaseAgent):
    name = "reviewer"

    async def review(
        self,
        description: str,
        expected_format: str,
        output: str,
        *,
        evidence: Evidence | None = None,
        producer_provider: str | None = None,
    ) -> ReviewVerdict:
        prompt = review_prompt(description, expected_format, output, evidence)
        try:
            return await self._verdict(prompt, producer_provider)
        except StructuredOutputError as error:
            return await self._verdict(prompt + RETRY_SUFFIX.format(error=error), producer_provider)

    async def _verdict(self, prompt: str, producer_provider: str | None) -> ReviewVerdict:
        response = await self.chat(
            [HumanMessage(content=prompt)], output_schema=ReviewVerdict, producer_provider=producer_provider
        )
        return response.parsed
