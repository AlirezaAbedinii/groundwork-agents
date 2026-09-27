"""A fake RAG API, so the MCP server can be tested without the retrieval service.

``FakeRag`` is an ``httpx.MockTransport`` handler. It serves the three routes the
server calls (``POST /v1/search``, ``POST /v1/ask``, ``GET /v1/documents``) with
bodies shaped like the real API's, records every request, and can be told to
fail. The ``rag_http`` fixture wraps it in the client that tests inject with
``build_server(RAG_URL, http=rag_http)``.
"""

import json

import httpx
import pytest

RAG_URL = "http://rag.test"

# Ferry-like chunks, best match first: (chunk_id, source_file, section_heading).
# A None heading is text above a file's first heading; the RAG API sends null.
CHUNKS = [
    ("a3f9c2e17b4d8e01", "04-error-codes.md", "Request errors"),
    ("5be80d6a91c2f473", "05-rate-limits.md", "What happens when you exceed the limit"),
    ("0c7d2e94a1b3f586", "03-configuration.md", "Environment variable mapping"),
    ("e61a4b8f2c9d0735", "04-error-codes.md", None),
    ("9d2b7c5e0f1a4836", "06-architecture.md", "Ferry — Architecture & Glossary"),
    ("47f0e3a9c8b21d65", "01-overview.md", "What Ferry is"),
]

# Chunk counts of the sample corpus under the default fixed-size chunker.
SOURCES = {
    "01-overview.md": 5,
    "02-quickstart.md": 7,
    "03-configuration.md": 6,
    "04-error-codes.md": 4,
    "05-rate-limits.md": 4,
    "06-architecture.md": 5,
    "README.md": 6,
}

ANSWER = (
    "FERRY-429 means the client exceeded its submission quota [1]. The response "
    "carries a Retry-After header in seconds, and the server does not retry for "
    "you [1]. Clients retry according to their ferry.retry.* settings [2]."
)

# services/rag/src/rag/generation/prompts.py: REFUSAL_MESSAGE
REFUSAL = (
    "I don't know based on the provided documentation. "
    "The retrieved context did not contain enough information to answer this question."
)

_TEXT = (
    "FERRY-429 — Rate Limit Exceeded. You have exceeded your submission quota. The "
    "response includes a Retry-After header (in seconds). The server does not retry "
    "for you — the CLI/client retries according to your ferry.retry.* settings. "
)


def chunk_text(chars: int) -> str:
    """Chunk text of exactly ``chars`` characters, non-ASCII dashes included."""
    return (_TEXT * (chars // len(_TEXT) + 1))[:chars]


class FakeRag:
    """Canned RAG API responses; every request is kept in ``requests``.

    Knobs, all off by default:
      chunk_chars  length of every chunk text (the RAG's chunk_size is 800)
      refuse       /v1/ask answers with the refusal instead of an answer
      fail         (status, body): every route answers with that error
      raises       the transport raises this, e.g. httpx.ConnectError("refused")
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.chunk_chars = 800
        self.refuse = False
        self.fail: tuple[int, dict | str] | None = None
        self.raises: Exception | None = None

    def sent(self) -> list[tuple[str, str, dict | None]]:
        """(method, path, JSON body or None) of every request received."""
        return [
            (r.method, r.url.path, json.loads(r.content) if r.content else None)
            for r in self.requests
        ]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        if self.fail is not None:
            status, body = self.fail
            if isinstance(body, str):
                return httpx.Response(status, text=body)
            return httpx.Response(status, json=body)
        route = (request.method, request.url.path)
        if route == ("POST", "/v1/search"):
            return httpx.Response(200, json=self._search(json.loads(request.content)))
        if route == ("POST", "/v1/ask"):
            return httpx.Response(200, json=self._ask(json.loads(request.content)))
        if route == ("GET", "/v1/documents"):
            return httpx.Response(200, json=self._documents())
        return httpx.Response(404, json={"detail": "Not Found"})

    def _chunks(self, top_k: int | None) -> list[tuple[int, str, str, str | None]]:
        # Without top_k the RAG API falls back to its configured default (10).
        return [(i, *chunk) for i, chunk in enumerate(CHUNKS[: top_k or 10])]

    def _search(self, body: dict) -> dict:
        return {
            "query": body["query"],
            "mode": body.get("mode") or "hybrid",
            "hits": [
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text(self.chunk_chars),
                    "score": 1 / (61 + i),  # RRF-sized, full float precision
                    "source_file": source_file,
                    "section_heading": heading,
                }
                for i, chunk_id, source_file, heading in self._chunks(body.get("top_k"))
            ],
            "timings_ms": {"embed": 3.2, "dense": 1.4, "sparse": 0.6, "total_ms": 41.8},
        }

    def _ask(self, body: dict) -> dict:
        contexts = [
            {
                "chunk_id": chunk_id,
                "text": chunk_text(self.chunk_chars),
                "score": 1 / (61 + i),
                "metadata": {
                    "source_file": source_file,
                    "section_heading": heading or "",
                    "page": -1,
                    "strategy": "fixed",
                    "ordinal": i,
                },
            }
            for i, chunk_id, source_file, heading in self._chunks(body.get("top_k") or 5)
        ]
        citations = [
            {
                "index": n,
                "resolved": True,
                "chunk_id": ctx["chunk_id"],
                "source_file": ctx["metadata"]["source_file"],
                "section_heading": ctx["metadata"]["section_heading"] or None,
                "supported": True,
            }
            for n, ctx in enumerate(contexts[:2], start=1)
        ]
        common = {"question": body["question"], "mode": body.get("mode") or "hybrid"}
        if self.refuse:
            # The RAG pipeline refuses before generating: contexts, no citations,
            # no tokens, and only the (here local, free) embedding cost.
            return common | {
                "answer": REFUSAL,
                "refused": True,
                "confidence": 0.12,
                "confidence_breakdown": {"retrieval": 0.12},
                "citations": [],
                "contexts": contexts,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost_usd": 0.0,
                "timings_ms": {"embed": 3.1, "dense": 1.3, "total_ms": 5.0},
            }
        return common | {
            "answer": ANSWER,
            "refused": False,
            "confidence": 0.8123456789012345,
            "confidence_breakdown": {
                "retrieval": 0.74,
                "citation_coverage": 1.0,
                "completeness": 1.0,
                "verified": True,
            },
            "citations": citations,
            "contexts": contexts,
            "usage": {"prompt_tokens": 1450, "completion_tokens": 72, "total_tokens": 1522},
            "cost_usd": 0.000261,
            "timings_ms": {"embed": 3.1, "dense": 1.3, "generate": 812.5, "total_ms": 860.3},
        }

    def _documents(self) -> dict:
        return {
            "documents": [{"source_file": s, "chunks": n} for s, n in SOURCES.items()],
            "total_chunks": sum(SOURCES.values()),
        }


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def fake_rag() -> FakeRag:
    return FakeRag()


@pytest.fixture
async def rag_http(fake_rag: FakeRag):
    """The client the server under test uses; its requests go to ``fake_rag``."""
    transport = httpx.MockTransport(fake_rag.handler)
    async with httpx.AsyncClient(base_url=RAG_URL, transport=transport) as client:
        yield client
