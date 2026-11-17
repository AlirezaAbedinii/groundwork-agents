"""Reviewer agent: scores specialist outputs 1–5 with feedback.

Routed to a different provider than the producing agent (llm/router.py).

Before the reviewer, ``required_tools_silent`` decides without a model whether the
attempt has anything to review at all: a subtask that lists tools, none of which
returned a result, is sent back with ``no_result_feedback``.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from orchestrator.agents.base import BaseAgent
from orchestrator.llm.structured import StructuredOutputError

# Stable marker; mock fixtures and tests match on it.
REVIEW_MARKER = "Review the following specialist output"
# What a subtask is told when none of its tools returned a result; tests and fixtures match on it.
NO_RESULT = "None of your required tools returned a result in this attempt"
# How many of the errors the feedback quotes, the most recent last
QUOTED_ERRORS = 3

REVIEW_PROMPT = """{marker} for quality before it is returned to the supervisor.

Subtask: {description}
Expected output format: {expected_format}

Specialist output:
---
{output}
---

Score the output 1-5 (5 = excellent, meets the subtask and format; 1 = unusable).
"""

RETRY_SUFFIX = """

Your previous verdict was invalid ({error}). Score the output again, from 1 to 5.
"""


class ReviewVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    feedback: str = Field("", description="What is wrong or missing, if anything")


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


class Reviewer(BaseAgent):
    name = "reviewer"

    async def review(
        self,
        description: str,
        expected_format: str,
        output: str,
        *,
        producer_provider: str | None = None,
    ) -> ReviewVerdict:
        prompt = REVIEW_PROMPT.format(
            marker=REVIEW_MARKER,
            description=description,
            expected_format=expected_format,
            output=output,
        )
        try:
            return await self._verdict(prompt, producer_provider)
        except StructuredOutputError as error:
            return await self._verdict(prompt + RETRY_SUFFIX.format(error=error), producer_provider)

    async def _verdict(self, prompt: str, producer_provider: str | None) -> ReviewVerdict:
        response = await self.chat(
            [HumanMessage(content=prompt)], output_schema=ReviewVerdict, producer_provider=producer_provider
        )
        return response.parsed
