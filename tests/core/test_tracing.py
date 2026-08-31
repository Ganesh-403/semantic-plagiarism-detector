import pytest

opentelemetry = pytest.importorskip("opentelemetry")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.core.processing import run_full_pipeline
from src.utils.tracing import get_tracer, init_tracer_provider


@pytest.fixture
def memory_exporter():
    exporter = InMemorySpanExporter()
    init_tracer_provider(exporter=exporter)
    return exporter


def test_pipeline_spans_hierarchy(memory_exporter):
    memory_exporter.clear()

    # Run mock pipeline
    tracer = get_tracer()
    with tracer.start_as_current_span("root_test"):
        run_full_pipeline([{"content": "test text"}])

    spans = memory_exporter.get_finished_spans()
    span_names = [span.name for span in spans]

    assert "run_full_pipeline" in span_names
    assert "pipeline.parse" in span_names
    assert "pipeline.chunk" in span_names
    assert "pipeline.embed" in span_names
    assert "pipeline.faiss_search" in span_names
