# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
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
