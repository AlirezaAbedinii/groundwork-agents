"""What the MCP tools return: structured results, sized for the agents that read them.

Every tool returns one of these models, so the SDK publishes an output schema for
it and sends the result as ``structured_content`` next to the text. The agent
loop shows the model at most 4000 characters of a search's JSON
(services/agent/src/orchestrator/tools/defaults.py), and wraps the result in
``{"content": …}`` first. ``search`` sizes its result to RESULT_BUDGET_CHARS,
which leaves room for that, so it arrives whole; ``ask`` leaves out the contexts
its answer came from.

``fit_hits`` does the sizing; it takes and returns plain dicts. Sizes are
characters of ``json.dumps`` output with its defaults, as the agent loop counts
them: each non-ASCII character is six.
"""

import json
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field

RESULT_BUDGET_CHARS = 3600
DEFAULT_TOP_K = 3
MAX_TOP_K = 8
# The query is echoed back; a longer one is echoed cut, so it can't crowd out the hits.
QUERY_ECHO_CHARS = 200

# RAG chunk metadata reports "" for a chunk with no section heading.
Heading = Annotated[str | None, BeforeValidator(lambda heading: heading or None)]


class SearchHit(BaseModel):
    """One retrieved chunk: the whole text, or as much of its start as fits."""

    chunk_id: str
    source_file: str
    section_heading: Heading
    score: float = Field(description="Retrieval score; higher is more relevant.")
    complete: bool = Field(description="True when text is the whole chunk.")
    text: str = Field(description="The chunk's text when complete; otherwise its start.")


class SearchResult(BaseModel):
    """Result of ``search``: the best-matching chunks, best first."""

    query: str = Field(description=f"The query; its first {QUERY_ECHO_CHARS} chars if longer.")
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


def _chars(value: Any) -> int:
    """Characters of ``value`` as JSON."""
    return len(json.dumps(value))


def _text_chars(text: str) -> int:
    """Characters ``text`` takes inside a JSON string."""
    return _chars(text) - 2


def fit_hits(query: str, mode: str, hits: list[dict], budget: int = RESULT_BUDGET_CHARS) -> dict:
    """A search result of at most ``budget`` characters: ``{"query", "mode", "hits"}``.

    ``hits`` are the RAG API's (chunk_id, source_file, section_heading, score,
    text), best first. Hits take whole texts in rank order while the result stays
    within the budget; from the first that doesn't fit, each takes as much of its
    start as fits an equal share of what is left (a share a hit doesn't use passes
    to the hits after it). Every hit keeps its metadata. Only metadata too long to
    fit at all drops hits, from the end.
    """
    echo = query if len(query) <= QUERY_ECHO_CHARS else query[:QUERY_ECHO_CHARS] + "…"
    shaped = [
        {
            "chunk_id": hit["chunk_id"],
            "source_file": hit["source_file"],
            "section_heading": hit["section_heading"] or None,
            "score": hit["score"],
            "complete": False,
            "text": "",
        }
        for hit in hits
    ]
    result = {"query": echo, "mode": mode, "hits": shaped}
    while shaped and _chars(result) > budget:
        shaped.pop()
    room = budget - _chars(result)
    whole = True
    for rank, out in enumerate(shaped):
        text = hits[rank]["text"]
        whole = whole and _text_chars(text) <= room
        out["text"] = text if whole else _prefix(text, room // (len(shaped) - rank))
        out["complete"] = out["text"] == text
        room -= _text_chars(out["text"]) - out["complete"]  # true is a character shorter than false
    return result


def _prefix(text: str, limit: int) -> str:
    """The longest start of ``text`` within ``limit`` characters inside a JSON string."""
    return text[: _most(len(text), lambda n: _text_chars(text[:n]) <= limit)]


def _most(high: int, fits) -> int:
    """The largest n in [0, high] with ``fits(n)``; ``fits`` is true up to some n, then false."""
    low = 0
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    return low
