"""Traces to Application Insights: one span per request, carrying the pipeline's outcome.

The app gets its own tracer provider with an in-memory exporter in place of the
batched Azure Monitor one, so nothing is global and nothing leaves the process.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from rag.api.main import create_app
from rag.config import Settings
from rag.indexing.vector_store import ScoredChunk
from rag.observability import telemetry
from rag.observability.trace_store import TraceStore
from rag.pipeline import AnswerResult


class FakePipeline:
    def __init__(self, mode: str, refused: bool) -> None:
        self.mode, self.refused = mode, refused

    def answer(self, question: str, top_k=None) -> AnswerResult:
        if self.refused:
            return AnswerResult(
                question=question, answer="", mode=self.mode, refused=True,
                retrieval_confidence=0.05, refused_by="gate",
                timings_ms={"embed": 12.0, "dense": 3.0, "total_ms": 15.0},
            )
        return AnswerResult(
            question=question, answer="Ferry retries failed jobs [1].", mode=self.mode,
            refused=False, retrieval_confidence=0.88, cost_usd=0.000045,
            timings_ms={"embed": 12.0, "dense": 3.0, "generate": 400.0, "total_ms": 415.0},
        )


class FakeRetriever:
    def retrieve(self, query: str, top_k=None, stopwatch=None) -> list[ScoredChunk]:
        with stopwatch.time("dense"):
            return [ScoredChunk("c1", "FERRY-429 — Rate Limit Exceeded.", 0.91, {})]


def _app(tmp_path, *, refused: bool = False, provider: TracerProvider | None = None):
    return create_app(
        Settings(_env_file=None),
        pipeline_factory=lambda mode: FakePipeline(mode, refused),
        retriever_factory=lambda mode: FakeRetriever(),
        trace_store=TraceStore(tmp_path / "traces.sqlite"),
        tracer_provider=provider,
    )


@pytest.fixture
def traced():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_an_answer_puts_its_outcome_on_the_request_span(tmp_path, traced):
    provider, exporter = traced
    client = TestClient(_app(tmp_path, provider=provider))
    response = client.post("/v1/ask", json={"question": "What is FERRY-429?", "mode": "dense"})
    assert response.status_code == 200

    (span,) = exporter.get_finished_spans()  # one span per request: no operation spans
    assert span.kind is SpanKind.SERVER
    assert span.name == "POST /v1/ask"
    attributes = dict(span.attributes)
    assert attributes["rag.mode"] == "dense"
    assert attributes["rag.refused"] is False
    assert "rag.refused_by" not in attributes  # None: left out
    assert attributes["rag.cost_usd"] == 0.000045
    assert attributes["rag.top_k"] == Settings(_env_file=None).top_k
    assert attributes["rag.timings_ms.generate"] == 400.0
    assert attributes["rag.timings_ms.total_ms"] == 415.0


def test_a_refusal_and_a_search_say_what_they_did(tmp_path, traced):
    provider, exporter = traced
    client = TestClient(_app(tmp_path, refused=True, provider=provider))
    client.post("/v1/ask", json={"question": "Why is the sky blue?", "mode": "hybrid", "top_k": 3})
    client.post("/v1/search", json={"query": "FERRY-429", "mode": "hybrid"})

    ask, search = (dict(span.attributes) for span in exporter.get_finished_spans())
    assert (ask["rag.refused"], ask["rag.refused_by"], ask["rag.top_k"]) == (True, "gate", 3)
    assert search["rag.mode"] == "hybrid"
    assert search["rag.top_k"] == Settings(_env_file=None).rerank_top_k  # hybrid's default
    assert "rag.timings_ms.dense" in search and "rag.refused" not in search


def test_health_probes_are_not_traced(tmp_path, traced):
    provider, exporter = traced
    client = TestClient(_app(tmp_path, provider=provider))
    assert client.get("/health").status_code == 200
    client.get("/v1/config")
    assert [span.name for span in exporter.get_finished_spans()] == ["GET /v1/config"]


def test_without_a_connection_string_nothing_is_traced(tmp_path, traced):
    _, exporter = traced  # a provider exists in the process, but the app isn't given it
    assert telemetry.setup_telemetry(Settings(_env_file=None)) is None
    client = TestClient(_app(tmp_path))
    assert client.post("/v1/ask", json={"question": "What is FERRY-429?"}).status_code == 200
    assert exporter.get_finished_spans() == ()
    telemetry.annotate(mode="dense")  # no request span: a no-op


def test_a_connection_string_builds_one_provider_named_for_the_service(monkeypatch):
    exported_to: list[tuple] = []

    class FakeAzureMonitorExporter(InMemorySpanExporter):
        def __init__(self, connection_string: str, tracer_provider) -> None:
            super().__init__()
            exported_to.append((connection_string, tracer_provider))

    import azure.monitor.opentelemetry.exporter as azure_exporter

    monkeypatch.setattr(azure_exporter, "AzureMonitorTraceExporter", FakeAzureMonitorExporter)
    telemetry._provider.cache_clear()
    settings = Settings(
        _env_file=None,
        applicationinsights_connection_string="InstrumentationKey=test",
        app_version="sha-abc1234",
    )
    provider = telemetry.setup_telemetry(settings)
    try:
        assert telemetry.setup_telemetry(settings) is provider  # once per process
        # the exporter reads this provider's resource, not a global one's
        assert exported_to == [("InstrumentationKey=test", provider)]
        assert provider.resource.attributes["service.name"] == "groundwork-rag"
        assert provider.resource.attributes["service.version"] == "sha-abc1234"
    finally:
        provider.shutdown()
        telemetry._provider.cache_clear()
