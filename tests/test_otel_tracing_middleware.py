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

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request, status
from fastapi.testclient import TestClient

# Import target custom middleware
from src.api.app import otel_tracing_middleware

# ----------------------------------------------------------------------
# 1. Setup a dummy FastAPI application and inject standard mock routes
# ----------------------------------------------------------------------
app = FastAPI()

# Add the otel tracing middleware
app.middleware("http")(otel_tracing_middleware)


# Simulated authentication injection dependency mock layer
@app.middleware("http")
async def mock_auth_injector(request: Request, call_next):
    # Mimic your authentication layer by stamping a user profile ID onto request states
    request.state.user_id = "user_98765_test"
    response = await call_next(request)
    return response


# Standard operational endpoint route configuration instance
@app.get("/api/v1/documents/{document_id}")
async def get_document_mock(document_id: str):
    return {"status": "success", "id": document_id}


# ----------------------------------------------------------------------
# 2. Pytest Suite containing TestClient validation assertions
# ----------------------------------------------------------------------
@patch("src.api.app.get_tracer")
def test_otel_tracing_middleware_intercepts_and_records_attributes(mock_get_tracer):
    """
    Verifies that the OpenTelemetry tracing middleware captures incoming request vectors,
    extracts stamped user authentication profiles, and tracks final HTTP status codes.
    """
    # Create mock handles mimicking OpenTelemetry Span life cycle structures
    mock_span = MagicMock()
    mock_tracer = MagicMock()

    # Configure the mock context manager to return our mock span handle safely
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    mock_get_tracer.return_value = mock_tracer

    # Programmatic shortcut verification block testing tracking logic explicitly:
    with TestClient(app) as client:
        # Fire a mock GET request targeting the route pattern profile
        response = client.get("/api/v1/documents/doc_abc123")

        # Verify the base application endpoint executed successfully
        assert response.status_code == status.HTTP_200_OK

    # ------------------------------------------------------------------
    # 3. Acceptance Criteria Assertions: Verify Span Attribute Binding
    # ------------------------------------------------------------------

    # Assert that set_attribute was invoked across our span targets
    mock_span.set_attribute.assert_any_call("http.method", "GET")
    mock_span.set_attribute.assert_any_call("user.id", "user_98765_test")
    mock_span.set_attribute.assert_any_call("http.status_code", 200)

    # Check that route path is recorded (either via request.url.path or route.path)
    mock_span.set_attribute.assert_any_call(
        "http.route", "/api/v1/documents/{document_id}"
    )
