"""The retrieval service as MCP tools: search, ask and list_sources.

A thin layer over the RAG HTTP API. Each tool call sends one request
(POST /v1/search, POST /v1/ask, GET /v1/documents) and returns a model from
tool_models.py, sized for the agents that read it. Every failure the server can
foresee becomes a ToolError, whose message the model sees; the SDK hides the
details of any other exception.

    python server.py                    # Streamable HTTP on 127.0.0.1:8001/mcp
    python server.py --transport stdio  # for desktop MCP clients

RAG_API_URL points at the RAG API (default http://localhost:8000).
"""

import argparse
import os
from typing import Annotated, Any, Literal

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from tool_models import (
    DEFAULT_TOP_K,
    SNIPPET_CHARS,
    AskResult,
    SearchHit,
    SearchResult,
    SourcesResult,
)

DEFAULT_RAG_URL = "http://localhost:8000"
TIMEOUT_S = 30.0
# Hints for other MCP clients; the agents' own policy never trusts them.
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

Mode = Annotated[
    Literal["dense", "hybrid"] | None,
    Field(
        description="hybrid: keywords and meaning, reranked; dense: meaning only. "
        "Leave it out to use the service's default."
    ),
]
TopK = Annotated[int, Field(ge=1, le=20, description="How many passages to use.")]


def build_server(rag_url: str | None = None, *, http: httpx.AsyncClient | None = None) -> MCPServer:
    """The MCP server over the RAG API at ``rag_url`` (default: $RAG_API_URL).

    ``http`` is a client for that API that the caller owns and closes (tests
    inject one); without it, every call opens and closes its own.
    """
    rag_url = rag_url or os.environ.get("RAG_API_URL", DEFAULT_RAG_URL)
    server = MCPServer("groundwork-rag", instructions="Read-only search over the team's docs.")

    async def rag(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        """One request to the RAG API; failures become errors the model can read."""
        # None means "not set": leave it out so the RAG API applies its default.
        payload = None if body is None else {k: v for k, v in body.items() if v is not None}
        try:
            if http is not None:
                response = await http.request(method, path, json=payload)
            else:
                async with httpx.AsyncClient(base_url=rag_url, timeout=TIMEOUT_S) as client:
                    response = await client.request(method, path, json=payload)
        except httpx.TransportError as exc:  # connect, timeout, dropped connection
            raise ToolError(
                f"retrieval service unreachable at {rag_url} ({type(exc).__name__})"
            ) from exc
        if not response.is_success:
            raise ToolError(
                f"retrieval service returned {response.status_code}: {_detail(response)}"
            )
        return response.json()

    @server.tool(
        annotations=READ_ONLY,
        description=(
            "Search the team's indexed documentation for passages that match a query. "
            "Returns the best hits first, each with its source file, section, relevance "
            f"score and a snippet (the passage's first {SNIPPET_CHARS} characters). Fast "
            "and cheap (no LLM call): use it to find where something is documented, check "
            "exact wording or collect sources to cite. Use ask instead for one answer that "
            "combines several passages."
        ),
    )
    async def search(
        query: Annotated[str, Field(min_length=1, max_length=2000, description="What to find.")],
        top_k: TopK = DEFAULT_TOP_K,
        mode: Mode = None,
    ) -> SearchResult:
        found = await rag("POST", "/v1/search", {"query": query, "top_k": top_k, "mode": mode})
        # Each hit keeps the start of its text as a snippet; the full text stays behind.
        hits = [SearchHit(**hit, snippet=hit["text"][:SNIPPET_CHARS]) for hit in found["hits"]]
        return SearchResult(query=found["query"], mode=found["mode"], hits=hits)

    @server.tool(
        annotations=READ_ONLY,
        description=(
            "Ask a question and get an answer written from the team's indexed "
            "documentation, with [n] citations to the passages it used and a confidence "
            "score. Slower than search and it spends the retrieval service's LLM budget, "
            "so use it when you need an explanation; use search to find or quote passages. "
            "refused=true means the documentation doesn't cover the question: that is an "
            "answer, not an error."
        ),
    )
    async def ask(
        question: Annotated[str, Field(min_length=1, max_length=2000, description="The question.")],
        mode: Mode = None,
        top_k: TopK | None = None,
    ) -> AskResult:
        answered = await rag(
            "POST", "/v1/ask", {"question": question, "mode": mode, "top_k": top_k}
        )
        # AskResult keeps only its own fields, so the retrieved contexts (and usage,
        # timings, ...) stay behind; the citations name the passages used.
        return AskResult.model_validate(answered)

    @server.tool(
        annotations=READ_ONLY,
        description=(
            "List the documents in the team's index, with how many chunks each has. "
            "Use it to see what the documentation covers before searching."
        ),
    )
    async def list_sources() -> SourcesResult:
        listed = await rag("GET", "/v1/documents")
        return SourcesResult(sources=listed["documents"], total_chunks=listed["total_chunks"])

    return server


def _detail(response: httpx.Response) -> str:
    """The error's ``detail`` from a JSON body (FastAPI's shape), else the body text."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])
    return response.text or response.reason_phrase


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the RAG service as MCP tools.")
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="http",
        help="http: Streamable HTTP on /mcp (default); stdio: for desktop MCP clients",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address")
    parser.add_argument("--port", type=int, default=8001, help="HTTP port")
    args = parser.parse_args(argv)

    server = build_server()
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
