"""Post-completion memory extraction: task → episode / facts / preferences."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from orchestrator.llm.clients import LLMClient
from orchestrator.llm.structured import StructuredOutputError
from orchestrator.memory.longterm import LongTermMemory

# Stable marker; mock fixtures match on it.
EXTRACT_MARKER = "Extract long-term memories"

EXTRACT_PROMPT = """{marker} from this completed task so future planning improves.

Task request: {request}

Subtask outputs:
{outputs}

Tools used per subtask: {tools}

Final deliverable:
{final}
"""

RETRY_SUFFIX = """

Your previous reply was invalid ({error}). Extract the memories again.
"""


class ExtractedMemories(BaseModel):
    episode: str = Field(description="One paragraph: what was asked, the approach that worked, the tools used")
    facts: list[str] = Field(default_factory=list, description="Domain facts discovered; empty when none apply")
    preferences: list[str] = Field(
        default_factory=list, description="User preferences observed; empty when none apply"
    )


async def extract_memories(
    llm: LLMClient, *, request: str, outputs: dict[str, str], tools_used: dict[str, list[str]], final_output: str
) -> ExtractedMemories:
    prompt = EXTRACT_PROMPT.format(
        marker=EXTRACT_MARKER,
        request=request,
        outputs="\n".join(f"[{sid}] {text}" for sid, text in sorted(outputs.items())) or "(none)",
        tools={sid: calls for sid, calls in sorted(tools_used.items())},
        final=final_output,
    )
    try:
        return await _extract(llm, prompt)
    except StructuredOutputError as error:
        return await _extract(llm, prompt + RETRY_SUFFIX.format(error=error))


async def _extract(llm: LLMClient, prompt: str) -> ExtractedMemories:
    response = await llm.chat("memory", [HumanMessage(content=prompt)], output_schema=ExtractedMemories)
    return response.parsed


def store_extracted(
    longterm: LongTermMemory,
    extracted: ExtractedMemories,
    *,
    user_id: str,
    task_id: str,
    events=None,
) -> list[tuple[str, str]]:
    """Persist extracted memories; returns (kind, memory_id) pairs."""
    stored: list[tuple[str, str]] = []
    batches = (
        ("episodes", [extracted.episode] if extracted.episode.strip() else []),
        ("facts", extracted.facts),
        ("preferences", extracted.preferences),
    )
    for kind, texts in batches:
        for text in texts:
            memory_id = longterm.add(kind, text, user_id=user_id, task_id=task_id)
            if events is not None:
                events.record(
                    user_id=user_id, memory_id=memory_id, kind=kind, action="created", task_id=task_id
                )
            stored.append((kind, memory_id))
    return stored
