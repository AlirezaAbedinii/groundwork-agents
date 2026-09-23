"""Reviewer agent: scores specialist outputs 1–5 with feedback.

Routed to a different provider than the producing agent (llm/router.py).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from orchestrator.agents.base import BaseAgent
from orchestrator.llm.structured import StructuredOutputError

# Stable marker; mock fixtures and tests match on it.
REVIEW_MARKER = "Review the following specialist output"

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
