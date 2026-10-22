"""OpenTelemetry traces to Azure Monitor (Application Insights), when configured.

Without ``APPLICATIONINSIGHTS_CONNECTION_STRING`` nothing is set up, and the service
keeps only its own ``Stopwatch`` timings and trace store. With it, ``create_app``
hands FastAPI (0.142+) a tracer provider with a batched Azure Monitor exporter:
FastAPI opens a server span per request, and the query routes add the pipeline's
outcome to it (``annotate``). The provider is the app's own, not the global one, and
is built once per process; it exports what its batch holds when the process exits.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from ..config import ConfigError, Settings

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

SERVICE_NAME = "groundwork-rag"


def setup_telemetry(settings: Settings) -> TracerProvider | None:
    """The process's tracer provider, or None when no connection string is set."""
    if not settings.applicationinsights_connection_string:
        return None
    return _provider(settings.applicationinsights_connection_string, settings.app_version)


@lru_cache
def _provider(connection_string: str, version: str) -> TracerProvider:
    try:
        from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise ConfigError(
            "APPLICATIONINSIGHTS_CONNECTION_STRING needs the telemetry extra. "
            'Install: pip install -e ".[telemetry]"'
        ) from exc
    provider = TracerProvider(
        resource=Resource.create({"service.name": SERVICE_NAME, "service.version": version})
    )
    # The exporter reads the resource from this provider, since it isn't the global one.
    exporter = AzureMonitorTraceExporter(
        connection_string=connection_string, tracer_provider=provider
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def fastapi_telemetry(provider: TracerProvider | None) -> dict:
    """``FastAPI(telemetry=...)``: request spans on ``provider`` only, none without it."""
    if provider is None:
        return {"tracing": False}
    return {
        "tracer_provider": provider,
        # One span per request, so the route's current span is the request span.
        "operation_spans": False,
        # Container Apps probes /health every few seconds; their spans would only be noise.
        "exclude": lambda scope: scope.get("path") == "/health",
    }


def annotate(**attributes: Any) -> None:
    """Set ``rag.<name>`` attributes on the current request span; a no-op without one.

    ``timings_ms`` (a stage -> ms dict) becomes one ``rag.timings_ms.<stage>`` each;
    None values are left out, since span attributes can't hold them.
    """
    span = trace.get_current_span()
    if not span.is_recording():
        return
    for name, value in attributes.items():
        if name == "timings_ms":
            for stage, ms in value.items():
                span.set_attribute(f"rag.timings_ms.{stage}", ms)
        elif value is not None:
            span.set_attribute(f"rag.{name}", value)
