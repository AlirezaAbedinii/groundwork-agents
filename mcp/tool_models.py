"""What the MCP tools return: structured results, sized for the agents that read them.

Every tool returns one of these models, so the SDK publishes an output schema for
it and sends the result as ``structured_content`` next to the text. The agent
loop shows the model at most RESULT_BUDGET_CHARS of a tool's JSON, so ``search``
sends snippets instead of whole chunks and ``ask`` leaves out the contexts its
answer came from.
"""

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field

# The agent loop cuts every tool result to this many characters of JSON
# (services/agent/src/orchestrator/agents/specialists/base.py:140).
RESULT_BUDGET_CHARS = 2000
SNIPPET_CHARS = 300
# Measured on the Ferry corpus, worst case: 3 hits of 300 chars come to 1588 chars,
# 5 hits to 2541 (each hit carries ~160 chars of metadata, and json.dumps turns
# every non-ASCII character into six). ADR 0004 records the trade-off against
# 5 hits of 150 chars, to re-check with the Phase 3 evals.
DEFAULT_TOP_K = 3

# The RAG index stores "" for a chunk with no section heading (Chroma rejects None).
Heading = Annotated[str | None, BeforeValidator(lambda heading: heading or None)]


class SearchHit(BaseModel):
    """One retrieved chunk, cut to a snippet."""

    chunk_id: str
    source_file: str
    section_heading: Heading
    score: float = Field(description="Retrieval score; higher is more relevant.")
    snippet: str = Field(description=f"The start of the chunk, at most {SNIPPET_CHARS} chars.")


class SearchResult(BaseModel):
    """Result of ``search``: the best-matching chunks, best first."""

    query: str
    mode: str
    hits: list[SearchHit]


class Citation(BaseModel):
    """One ``[n]`` citation in an answer, mapped to the chunk it cites."""

    index: int = Field(description="The n in [n], as written in the answer.")
    chunk_id: str
    source_file: str
    section_heading: Heading
    supported: bool | None = Field(
        description="Whether a verifier judged that the chunk supports the claim; "
        "null when verification did not run."
    )


class AskResult(BaseModel):
    """Result of ``ask``: a grounded answer, without the contexts behind it."""

    answer: str
    refused: bool = Field(description="True when the service declined to answer.")
    confidence: float = Field(description="Composite confidence in [0, 1].")
    citations: list[Citation]
    cost_usd: float = Field(description="What the retrieval service spent on this answer.")


class Source(BaseModel):
    """One indexed document."""

    source_file: str
    chunks: int


class SourcesResult(BaseModel):
    """Result of ``list_sources``: every indexed document and its chunk count."""

    sources: list[Source]
    total_chunks: int
