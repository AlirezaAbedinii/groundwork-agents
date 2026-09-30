"""Phase 5 tests for the FastAPI service (LLM mocked via a fake pipeline)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag.api.main import create_app
from rag.config import ConfigError, Settings
from rag.generation.citations import Citation
from rag.generation.prompts import REFUSAL_MESSAGE
from rag.indexing.vector_store import ScoredChunk
from rag.observability.metrics import TokenUsage
from rag.observability.trace_store import TraceStore
from rag.pipeline import AnswerResult


def _answered(question: str) -> AnswerResult:
    ctx = ScoredChunk(
        "c1", "Ferry retries failed jobs.", 0.88, {"source_file": "04-error-codes.md"}
    )
    return AnswerResult(
        question=question,
        answer="Ferry retries failed jobs automatically [1].",
        mode="dense",
        refused=False,
        retrieval_confidence=0.88,
        confidence=0.88,
        confidence_breakdown={
            "retrieval": 0.88,
            "citation_coverage": 1.0,
            "completeness": 1.0,
            "verified": True,
        },
        citations=[
            Citation(
                index=1,
                resolved=True,
                chunk_id="c1",
                source_file="04-error-codes.md",
                supported=True,
            )
        ],
        contexts=[ctx],
        usage=TokenUsage(prompt_tokens=120, completion_tokens=30),
        cost_usd=0.000045,
        timings_ms={"embed": 12.0, "dense": 3.0, "generate": 400.0, "total_ms": 415.0},
    )


def _refused(question: str) -> AnswerResult:
    return AnswerResult(
        question=question,
        answer=REFUSAL_MESSAGE,
        mode="dense",
        refused=True,
        retrieval_confidence=0.05,
        timings_ms={"embed": 12.0, "dense": 3.0, "total_ms": 15.0},
    )


class FakePipeline:
    def __init__(self, mode: str, result_builder) -> None:
        self.mode = mode
        self._build = result_builder

    def answer(self, question: str, top_k=None) -> AnswerResult:
        result = self._build(question)
        result.mode = self.mode
        return result


@pytest.fixture
def client(tmp_path):
    """App wired with a fake pipeline + a real (temp) trace store."""
    settings = Settings(_env_file=None)
    app = create_app(
        settings,
        pipeline_factory=lambda mode: FakePipeline(mode, _answered),
        trace_store=TraceStore(tmp_path / "traces.sqlite"),
    )
    return TestClient(app)


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_happy_path_returns_documented_schema(client: TestClient) -> None:
    resp = client.post("/v1/ask", json={"question": "How does Ferry handle failures?"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["answer"].endswith("[1].")
    assert body["refused"] is False
    assert body["mode"] == "hybrid"  # no mode in the request -> settings.default_mode
    assert body["confidence"] == pytest.approx(0.88)
    assert body["confidence_breakdown"]["citation_coverage"] == 1.0
    assert body["confidence_breakdown"]["verified"] is True
    # Citations resolve to retrieved chunks, with the verification verdict.
    assert body["citations"] == [
        {
            "index": 1,
            "resolved": True,
            "chunk_id": "c1",
            "source_file": "04-error-codes.md",
            "section_heading": None,
            "supported": True,
        }
    ]
    assert body["contexts"][0]["chunk_id"] == "c1"
    assert body["contexts"][0]["score"] == pytest.approx(0.88)
    # Latency/cost metadata fields (the acceptance criterion).
    assert body["cost_usd"] == pytest.approx(0.000045)
    assert body["timings_ms"]["total_ms"] == pytest.approx(415.0)
    assert body["usage"]["total_tokens"] == 150


def test_ask_refusal_shape(tmp_path) -> None:
    settings = Settings(_env_file=None)
    app = create_app(
        settings,
        pipeline_factory=lambda mode: FakePipeline(mode, _refused),
        trace_store=TraceStore(tmp_path / "t.sqlite"),
    )
    resp = TestClient(app).post("/v1/ask", json={"question": "What is the meaning of life?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is True
    assert body["answer"] == REFUSAL_MESSAGE
    assert body["citations"] == []
    assert body["usage"]["total_tokens"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing question
        {"question": ""},  # empty question
        {"question": "ok", "mode": "sparse"},  # invalid mode literal
        {"question": "ok", "top_k": 0},  # top_k below bound
        {"question": "ok", "top_k": 999},  # top_k above bound
    ],
)
def test_ask_bad_input_is_422(client: TestClient, payload: dict) -> None:
    assert client.post("/v1/ask", json=payload).status_code == 422


def test_ask_hybrid_before_seeding_is_503(tmp_path) -> None:
    """A factory failing with FileNotFoundError (unseeded BM25) -> actionable 503."""

    def factory(mode: str):
        if mode == "hybrid":
            raise FileNotFoundError("BM25 index not found. Ingest first: python scripts/seed.py")
        return FakePipeline(mode, _answered)

    app = create_app(
        Settings(_env_file=None),
        pipeline_factory=factory,
        trace_store=TraceStore(tmp_path / "t.sqlite"),
    )
    resp = TestClient(app).post("/v1/ask", json={"question": "anything", "mode": "hybrid"})
    assert resp.status_code == 503
    assert "seed" in resp.json()["detail"]


def test_ask_logs_a_trace_per_request(tmp_path) -> None:
    store = TraceStore(tmp_path / "traces.sqlite")
    app = create_app(
        Settings(_env_file=None),
        pipeline_factory=lambda mode: FakePipeline(mode, _answered),
        trace_store=store,
    )
    client = TestClient(app)
    client.post("/v1/ask", json={"question": "q one"})
    client.post("/v1/ask", json={"question": "q two"})
    assert store.count() == 2


def test_openapi_docs_are_exposed(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    for path in ("/v1/ask", "/v1/search", "/v1/ingest", "/v1/documents", "/v1/stats"):
        assert path in schema["paths"]


# --- V1 endpoints -----------------------------------------------------------
class FakeStore:
    def list_sources(self) -> dict[str, int]:
        return {"01-overview.md": 5, "04-error-codes.md": 4}

    def count(self) -> int:
        return 9


class FakeIndexSummary:
    files = 7
    chunks_indexed = 37
    total_chunks_in_store = 37
    embedding_cost_usd = 0.00012
    bm25_chunks = 37
    chunks_skipped_duplicates = 2
    timings_ms = {"load_chunk": 3.0, "embed": 120.0, "store": 15.0, "total_ms": 138.0}


def _v1_app(tmp_path):
    return create_app(
        Settings(_env_file=None),
        pipeline_factory=lambda mode: FakePipeline(mode, _answered),
        trace_store=TraceStore(tmp_path / "traces.sqlite"),
        indexer=lambda path: FakeIndexSummary(),
        store_factory=lambda: FakeStore(),
    )


def test_stats_aggregates_over_logged_requests(tmp_path) -> None:
    client = TestClient(_v1_app(tmp_path))
    for _ in range(3):
        client.post("/v1/ask", json={"question": "q"})

    stats = client.get("/v1/stats").json()
    assert stats["requests"] == 3
    assert stats["refused"] == 0
    assert stats["total_cost_usd"] == pytest.approx(3 * 0.000045)
    # Per-stage percentiles present, including the generate stage.
    assert stats["latency_ms"]["generate"]["p95"] == pytest.approx(400.0)
    assert stats["latency_ms"]["total_ms"]["n"] == 3


def test_stats_empty_is_zeroed(tmp_path) -> None:
    client = TestClient(_v1_app(tmp_path))
    stats = client.get("/v1/stats").json()
    assert stats["requests"] == 0
    assert stats["refusal_rate"] == 0.0
    assert stats["latency_ms"] == {}


def test_documents_lists_indexed_sources(tmp_path) -> None:
    client = TestClient(_v1_app(tmp_path))
    body = client.get("/v1/documents").json()
    assert body["total_chunks"] == 9
    assert {"source_file": "01-overview.md", "chunks": 5} in body["documents"]


def test_documents_builds_one_store_and_reuses_it(tmp_path) -> None:
    built: list[FakeStore] = []

    def store_factory() -> FakeStore:
        built.append(FakeStore())
        return built[-1]

    client = TestClient(
        create_app(
            Settings(_env_file=None),
            trace_store=TraceStore(tmp_path / "traces.sqlite"),
            store_factory=store_factory,
        )
    )
    for _ in range(3):
        assert client.get("/v1/documents").status_code == 200

    assert len(built) == 1  # the real store holds a connection pool


def test_ingest_defaults_to_sample_corpus(tmp_path) -> None:
    client = TestClient(_v1_app(tmp_path))
    resp = client.post("/v1/ingest", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks_indexed"] == 37
    assert body["embedding_cost_usd"] == pytest.approx(0.00012)
    assert "embed" in body["timings_ms"]


def test_ingest_missing_path_is_400(tmp_path) -> None:
    client = TestClient(_v1_app(tmp_path))
    resp = client.post("/v1/ingest", json={"path": "/no/such/dir"})
    assert resp.status_code == 400
    assert "does not exist" in resp.json()["detail"]


# --- /v1/search: retrieval only ----------------------------------------------
class FakeRetriever:
    """Canned chunks, best first; records every retrieve() call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k=None, stopwatch=None) -> list[ScoredChunk]:
        self.calls.append((query, top_k))
        with stopwatch.time("dense"):
            return [
                ScoredChunk(
                    "c1",
                    "FERRY-429 — Rate Limit Exceeded.",
                    0.91,
                    {"source_file": "04-error-codes.md", "section_heading": "Request errors"},
                ),
                # The index stores "" for a chunk above the first heading.
                ScoredChunk(
                    "c2",
                    "Every error has a stable FERRY-NNN code.",
                    0.74,
                    {"source_file": "04-error-codes.md", "section_heading": ""},
                ),
            ]


def _no_pipeline(mode: str):
    raise AssertionError("/v1/search must not build a pipeline: that needs a chat client")


def _search_app(tmp_path, retriever_factory, pipeline_factory=_no_pipeline):
    return create_app(
        Settings(_env_file=None),
        pipeline_factory=pipeline_factory,
        retriever_factory=retriever_factory,
        trace_store=TraceStore(tmp_path / "traces.sqlite"),
    )


def test_search_returns_ranked_hits_and_logs_no_trace(tmp_path) -> None:
    app = _search_app(tmp_path, lambda mode: FakeRetriever())
    resp = TestClient(app).post("/v1/search", json={"query": "FERRY-429"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "FERRY-429"
    assert body["mode"] == "hybrid"  # no mode in the request -> settings.default_mode
    assert body["hits"] == [
        {
            "chunk_id": "c1",
            "text": "FERRY-429 — Rate Limit Exceeded.",
            "score": 0.91,
            "source_file": "04-error-codes.md",
            "section_heading": "Request errors",
        },
        {
            "chunk_id": "c2",
            "text": "Every error has a stable FERRY-NNN code.",
            "score": 0.74,
            "source_file": "04-error-codes.md",
            "section_heading": None,
        },
    ]
    assert set(body["timings_ms"]) == {"dense", "total_ms"}
    assert app.state.trace_store.count() == 0  # /v1/stats covers answered questions only


def test_search_passes_mode_and_top_k_to_one_retriever_per_mode(tmp_path) -> None:
    built: dict[str, FakeRetriever] = {}

    def factory(mode: str) -> FakeRetriever:
        assert mode not in built, "a mode's retriever is built once and reused"
        built[mode] = FakeRetriever()
        return built[mode]

    client = TestClient(_search_app(tmp_path, factory))
    client.post("/v1/search", json={"query": "a", "mode": "dense", "top_k": 7})
    client.post("/v1/search", json={"query": "b", "mode": "dense"})

    assert list(built) == ["dense"]
    # No top_k in the request -> None, so the retriever applies its own default.
    assert built["dense"].calls == [("a", 7), ("b", None)]


def test_search_works_where_ask_lacks_a_generation_key(tmp_path) -> None:
    def keyless_pipeline(mode: str):
        raise ConfigError("Missing required configuration: OPENAI_API_KEY.")

    client = TestClient(
        _search_app(tmp_path, lambda mode: FakeRetriever(), pipeline_factory=keyless_pipeline)
    )

    assert client.post("/v1/search", json={"query": "FERRY-429"}).status_code == 200
    assert client.post("/v1/ask", json={"question": "What is FERRY-429?"}).status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing query
        {"query": ""},  # empty query
        {"query": "x" * 2001},  # query above the length bound
        {"query": "ok", "mode": "sparse"},  # invalid mode literal
        {"query": "ok", "top_k": 0},  # top_k below bound
        {"query": "ok", "top_k": 51},  # top_k above bound
    ],
)
def test_search_bad_input_is_422(tmp_path, payload: dict) -> None:
    retriever = FakeRetriever()
    client = TestClient(_search_app(tmp_path, lambda mode: retriever))
    assert client.post("/v1/search", json=payload).status_code == 422
    assert retriever.calls == []


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (NotImplementedError("mode not available"), 501),
        (ConfigError("Missing required configuration: OPENAI_API_KEY."), 503),
        (FileNotFoundError("BM25 index not found. Ingest first: python scripts/seed.py"), 503),
        (ImportError("No module named 'psycopg'"), 503),
    ],
)
def test_search_maps_missing_pieces_exactly_like_ask(tmp_path, error, status) -> None:
    def broken(mode: str):
        raise error

    client = TestClient(_search_app(tmp_path, broken, pipeline_factory=broken))
    ask = client.post("/v1/ask", json={"question": "q"})
    search = client.post("/v1/search", json={"query": "q"})

    assert search.status_code == ask.status_code == status
    assert search.json() == ask.json()


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("Collection 'ferry_docs' is not indexed; seed first."),
        ConfigError("Collection 'ferry_docs' was embedded with 'a', but EMBEDDING_MODEL is 'b'."),
    ],
)
def test_store_errors_while_serving_are_503_too(tmp_path, error) -> None:
    """A dense retriever only meets its collection at query time, not when built."""

    class RaisingRetriever:
        def retrieve(self, query, top_k=None, stopwatch=None):
            raise error

    class RaisingPipeline:
        def answer(self, question, top_k=None):
            raise error

    client = TestClient(
        _search_app(
            tmp_path, lambda mode: RaisingRetriever(), pipeline_factory=lambda m: RaisingPipeline()
        )
    )
    search = client.post("/v1/search", json={"query": "q"})
    ask = client.post("/v1/ask", json={"question": "q"})

    assert search.status_code == ask.status_code == 503
    assert search.json() == ask.json() == {"detail": str(error)}


def test_search_hybrid_before_seeding_is_503_then_works_once_seeded(tmp_path) -> None:
    seeded = False

    def factory(mode: str) -> FakeRetriever:
        if mode == "hybrid" and not seeded:
            raise FileNotFoundError("BM25 index not found. Ingest first: python scripts/seed.py")
        return FakeRetriever()

    client = TestClient(_search_app(tmp_path, factory))
    resp = client.post("/v1/search", json={"query": "q", "mode": "hybrid"})
    assert resp.status_code == 503
    assert "seed" in resp.json()["detail"]

    seeded = True  # a failed build isn't cached: the next request builds again
    assert client.post("/v1/search", json={"query": "q", "mode": "hybrid"}).status_code == 200
