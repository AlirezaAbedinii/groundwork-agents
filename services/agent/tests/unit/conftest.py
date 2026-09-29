"""A stand-in for the RAG MCP server: in-process, recording exactly what reaches it."""

from typing import Annotated, Literal

import pytest
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field


class Hit(BaseModel):
    source_file: str
    snippet: str


class Hits(BaseModel):
    query: str
    hits: list[Hit]


class Answer(BaseModel):
    answer: str
    refused: bool


class Sources(BaseModel):
    sources: list[str]
    total_chunks: int


class RagStub:
    """An MCPServer shaped like the RAG one, plus tools no policy allows.

    ``calls`` keeps each call's (tool, arguments) exactly as they arrived on the
    wire, so tests see what the adapter sent, including the keys it left out.
    Set ``fail`` to make every tool answer with that error.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail: str | None = None
        self.server = server = MCPServer("rag-stub")

        # A misleading annotation on purpose: the agent's policy decides sensitivity, never the server.
        @server.tool(description="Search the team's docs.", annotations=ToolAnnotations(destructiveHint=True))
        async def search(
            ctx: Context,
            query: Annotated[str, Field(min_length=1, max_length=2000)],
            top_k: Annotated[int, Field(ge=1, le=20)] = 3,
            mode: Literal["dense", "hybrid"] | None = None,
        ) -> Hits:
            self._received(ctx)
            return Hits(query=query, hits=[Hit(source_file="04-error-codes.md", snippet=f"{query}: rate limited")])

        @server.tool(description="Answer from the team's docs.")
        async def ask(
            ctx: Context,
            question: Annotated[str, Field(min_length=1, max_length=2000)],
            mode: Literal["dense", "hybrid"] | None = None,
            top_k: Annotated[int, Field(ge=1, le=20)] | None = None,
        ) -> Answer:
            self._received(ctx)
            return Answer(answer=f"Answer to {question}", refused=False)

        @server.tool(description="List the indexed documents.")
        async def list_sources(ctx: Context) -> Sources:
            self._received(ctx)
            return Sources(sources=["04-error-codes.md", "05-rate-limits.md"], total_chunks=8)

        @server.tool(description="Plain-text version string.", structured_output=False)
        async def version(ctx: Context) -> str:
            self._received(ctx)
            return "rag-stub 1.0"

        # Offered by the server, allowed by no policy; its list argument is outside
        # the schema subset, so discovery must not even build a model for it.
        @server.tool(description="Delete every document.", annotations=ToolAnnotations(destructiveHint=True))
        async def delete_everything(ctx: Context, paths: list[str]) -> Sources:
            self._received(ctx)
            return Sources(sources=[], total_chunks=0)

    def _received(self, ctx: Context) -> None:
        params = ctx.request_context.params
        self.calls.append((params["name"], params.get("arguments") or {}))
        if self.fail is not None:
            raise ToolError(self.fail)


@pytest.fixture()
def rag_stub() -> RagStub:
    return RagStub()
