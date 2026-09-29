"""Custom Postgres span exporter: row mapping, parent links, status rules.

Uses a local TracerProvider (never the global one) with the exporter's test
sink, so nothing touches the database or other tests' tracing state.
"""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from orchestrator.observability.tracing import PostgresSpanExporter


@pytest.fixture()
def capture():
    rows: list[dict] = []
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(PostgresSpanExporter(sink=rows.extend)))
    return provider.get_tracer("test"), rows


def test_parent_links_attributes_and_duration(capture):
    tracer, rows = capture
    with tracer.start_as_current_span("plan") as root:
        root.set_attribute("orchestrator.task_id", "t1")
        root.set_attribute("orchestrator.kind", "planning")
        root.set_attribute("confidence", 0.9)
        with tracer.start_as_current_span("llm:supervisor") as child:
            child.set_attribute("orchestrator.task_id", "t1")
            child.set_attribute("orchestrator.kind", "llm")
            child.set_attribute("orchestrator.agent", "supervisor")
            child.set_attribute("cost_usd", 0.004)

    child_row, root_row = rows  # children end (and export) first
    assert root_row["name"] == "plan"
    assert root_row["kind"] == "planning"
    assert root_row["task_id"] == "t1"
    assert root_row["parent_id"] is None
    assert len(root_row["id"]) == 16 and len(root_row["trace_id"]) == 32
    assert root_row["duration_ms"] >= 0
    assert root_row["attributes"] == {"confidence": 0.9}  # orchestrator.* keys extracted

    assert child_row["parent_id"] == root_row["id"]
    assert child_row["trace_id"] == root_row["trace_id"]
    assert child_row["agent"] == "supervisor"
    assert child_row["attributes"]["cost_usd"] == 0.004


def test_status_defaults_exception_and_custom_override(capture):
    tracer, rows = capture
    with tracer.start_as_current_span("ok") as span:
        span.set_attribute("orchestrator.task_id", "t1")

    with pytest.raises(RuntimeError):
        with tracer.start_as_current_span("boom") as span:
            span.set_attribute("orchestrator.task_id", "t1")
            raise RuntimeError("kaput")

    with tracer.start_as_current_span("gate") as span:
        span.set_attribute("orchestrator.task_id", "t1")
        span.set_attribute("orchestrator.status", "escalated")

    by_name = {row["name"]: row for row in rows}
    assert by_name["ok"]["status"] == "success"
    assert by_name["boom"]["status"] == "failure"
    assert by_name["gate"]["status"] == "escalated"  # custom attr wins


def test_spans_without_task_id_are_not_exported(capture):
    tracer, rows = capture
    with tracer.start_as_current_span("framework-noise"):
        pass
    with tracer.start_as_current_span("task-scoped") as span:
        span.set_attribute("orchestrator.task_id", "t1")

    assert [row["name"] for row in rows] == ["task-scoped"]


def test_a_task_run_is_its_own_trace_even_inside_a_request_span(capture, monkeypatch):
    # FastAPI (0.142+) opens a span for every request, and inline runs execute as
    # background tasks inside it; a run's root span must still be a root.
    from orchestrator.observability import tracing

    tracer, rows = capture
    monkeypatch.setattr(tracing, "_tracer", lambda: tracer)

    with tracer.start_as_current_span("POST /approvals/{approval_id}/resolve"):
        with tracing.task_run_span("t1", "task:resume"):
            with tracing.child_span("tool:web_search", kind="tool"):
                pass

    by_name = {row["name"]: row for row in rows}  # the request span has no task id: not exported
    assert set(by_name) == {"task:resume", "tool:web_search"}
    assert by_name["task:resume"]["parent_id"] is None
    assert by_name["tool:web_search"]["parent_id"] == by_name["task:resume"]["id"]
