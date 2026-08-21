from __future__ import annotations

import os
from typing import Any, Optional

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    trace = None
    TracerProvider = Any
    SpanExporter = Any

_tracer_provider: Optional[Any] = None
_initialized = False


def init_tracer_provider(exporter: Optional[Any] = None) -> Any:
    """Initialize or retrieve the global TracerProvider idempotently."""
    global _tracer_provider, _initialized
    if not HAS_OPENTELEMETRY:
        return None
    if _tracer_provider is not None and _initialized:
        return _tracer_provider

    service_name = os.getenv("OTEL_SERVICE_NAME", "semantic-plagiarism-detector")
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)

    # Use provided custom exporter (e.g., in-memory for tests) or OTLP if endpoint is configured
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if exporter:
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider.add_span_processor(SimpleSpanProcessor(exporter))
    elif otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    trace.set_tracer_provider(provider)
    _tracer_provider = provider
    _initialized = True
    return provider


def get_tracer(name: str = "semantic-plagiarism.detector") -> Any:
    """Get an OpenTelemetry tracer instance lazily."""
    if not HAS_OPENTELEMETRY:
        return None
    if not _tracer_provider:
        init_tracer_provider()
    return trace.get_tracer(name)


def inject_traceparent(carrier: dict[str, str]) -> None:
    """Inject current trace context into carrier dict (e.g., HTTP headers)."""
    if HAS_OPENTELEMETRY:
        TraceContextTextMapPropagator().inject(carrier=carrier)


def extract_traceparent(carrier: dict[str, str]) -> Any:
    """Extract trace context from carrier dict."""
    if HAS_OPENTELEMETRY:
        return TraceContextTextMapPropagator().extract(carrier=carrier)
    return None
