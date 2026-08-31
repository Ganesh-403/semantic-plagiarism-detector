import pytest

opentelemetry = pytest.importorskip("opentelemetry")
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.api.app import app


@pytest.fixture(autouse=True)
def memory_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Store the original provider
    original_provider = trace.get_tracer_provider()

    # Set the new test provider
    trace.set_tracer_provider(provider)

    yield exporter

    # Restore the original provider
    trace.set_tracer_provider(original_provider)


def test_otel_middleware_records_exception(memory_exporter):
    """
    Test that the otel_tracing_middleware catches an exception from a route,
    records it in the span, sets http.status_code to 500, and re-raises.
    """
    client = TestClient(app, raise_server_exceptions=True)

    @app.get("/_test_error")
    async def test_error():
        raise ValueError("Intentional error for testing")

    # The exception should be propagated back to the TestClient
    with pytest.raises(ValueError, match="Intentional error"):
        client.get("/_test_error")

    spans = memory_exporter.get_finished_spans()
    assert len(spans) > 0, "No spans were exported"

    # Find our HTTP span
    http_span = next(
        (span for span in spans if span.name == "HTTP GET /_test_error"), None
    )
    assert http_span is not None, "Could not find HTTP span for the route"

    # Check that status code is 500
    attributes = http_span.attributes
    assert attributes.get("http.status_code") == 500

    # Check exception events
    events = http_span.events
    assert len(events) > 0, "No events recorded on the span"

    exception_event = next((e for e in events if e.name == "exception"), None)
    assert exception_event is not None, "No exception event recorded"

    assert exception_event.attributes.get("exception.type") == "ValueError"
    assert "Intentional error for testing" in exception_event.attributes.get(
        "exception.message", ""
    )


def test_otel_middleware_extracts_user_id_from_bearer_token(memory_exporter):
    """Test that otel_tracing_middleware extracts user ID from Authorization Bearer token."""
    from src.security.jwt_utils import create_access_token

    token = create_access_token(sub="alice_user_123")
    client = TestClient(app)

    @app.get("/_test_user_id")
    async def test_user():
        return {"status": "ok"}

    client.get("/_test_user_id", headers={"Authorization": f"Bearer {token}"})

    spans = memory_exporter.get_finished_spans()
    http_span = next((s for s in spans if s.name == "HTTP GET /_test_user_id"), None)
    assert http_span is not None
    assert http_span.attributes.get("user.id") == "alice_user_123"


def test_otel_middleware_anonymous_user_id_fallback(memory_exporter):
    """Test that otel_tracing_middleware defaults user.id to 'anonymous' when unauthenticated."""
    client = TestClient(app)

    @app.get("/_test_anon_user")
    async def test_anon():
        return {"status": "ok"}

    client.get("/_test_anon_user")

    spans = memory_exporter.get_finished_spans()
    http_span = next((s for s in spans if s.name == "HTTP GET /_test_anon_user"), None)
    assert http_span is not None
    assert http_span.attributes.get("user.id") == "anonymous"


def test_otel_middleware_groups_by_route_template(memory_exporter):
    """Test that otel_tracing_middleware uses the route template (generalized FastAPI path) for the span name to avoid cardinality explosion."""
    client = TestClient(app)

    @app.get("/_test_users/{user_id}")
    async def get_test_user(user_id: int):
        return {"user_id": user_id}

    client.get("/_test_users/123")
    client.get("/_test_users/456")

    spans = memory_exporter.get_finished_spans()
    user_spans = [
        s for s in spans if "/_test_users/" in s.name or "/_test_users/{" in s.name
    ]
    assert len(user_spans) == 2, "Should have 2 user spans recorded"
    for span in user_spans:
        assert span.name == "HTTP GET /_test_users/{user_id}"
        assert span.attributes.get("http.route") == "/_test_users/{user_id}"


def test_otel_middleware_injects_trace_id_in_global_exception_handler(
    memory_exporter, monkeypatch
):
    """Test that global_exception_handler injects the OpenTelemetry Trace ID into the error response payload when a trace is active."""
    monkeypatch.setenv("APP_ENVIRONMENT", "production")
    client = TestClient(app, raise_server_exceptions=False)

    @app.get("/_test_unhandled_error")
    async def raise_error():
        raise RuntimeError("Test unhandled exception for trace ID injection")

    response = client.get("/_test_unhandled_error")
    assert response.status_code == 500
    body = response.json()
    assert body["error"] is True
    assert "trace_id" in body
    assert len(body["trace_id"]) == 32  # standard 32-character hex string for trace ID

    # Ensure it matches the span trace ID
    spans = memory_exporter.get_finished_spans()
    span = next((s for s in spans if "/_test_unhandled_error" in s.name), None)
    assert span is not None
    span_trace_id = trace.format_trace_id(span.get_span_context().trace_id)
    assert body["trace_id"] == span_trace_id
