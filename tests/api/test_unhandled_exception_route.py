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

"""
tests/api/test_unhandled_exception_route.py
----------------------------------------------
End-to-end coverage for the global exception handler, via a real,
fully-routed request rather than a direct handler call.

tests/api/test_app.py already covers global_exception_handler's payload
shape by calling `asyncio.run(global_exception_handler(mock_request, exc))`
directly -- but that never exercises FastAPI's actual exception-dispatch
path (routing, the app's global auth dependency, and the
`@app.exception_handler(Exception)` registration all doing their real
jobs). This module closes that gap with a dedicated test-only route that
deliberately raises `RuntimeError("Boom")`, hit via `TestClient`, so the
full request/response cycle -- not just the handler function in
isolation -- is verified.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.middleware import verify_bearer_token

TEST_RUNTIME_ERROR_PATH = "/api/v1/_test/raise-runtime-error"


# Registered once, directly on the real `app` singleton, at import time.
# This never ships to production traffic: the route only exists because
# this test module imported and executed this decorator, which only
# happens under pytest. `include_in_schema=False` also keeps it out of
# the generated OpenAPI docs.
@app.get(TEST_RUNTIME_ERROR_PATH, include_in_schema=False)
def _raise_runtime_error_for_testing():
    """Deliberately raise an unhandled exception so the real
    global_exception_handler dispatch path can be exercised end-to-end."""
    raise RuntimeError("Boom")


client = TestClient(app)


@pytest.fixture(autouse=True)
def _bypass_auth_for_this_module():
    """Override the app's global auth dependency for the duration of this
    module's tests, so they isolate global_exception_handler's behavior
    rather than authentication. Uses FastAPI's documented
    dependency_overrides mechanism, which correctly intercepts
    verify_bearer_token wherever it's wired in -- including as a
    top-level `dependencies=[...]` entry on the FastAPI() app itself, not
    just per-route Depends()."""
    app.dependency_overrides[verify_bearer_token] = lambda: "test-token"
    yield
    app.dependency_overrides.pop(verify_bearer_token, None)


def test_unhandled_runtime_error_returns_500_with_masked_message_in_production(
    monkeypatch,
):
    """Acceptance criteria: a route that raises RuntimeError("Boom") must
    return HTTP 500 with the standardized error body, and "Boom" must
    never be exposed anywhere in the response when
    APP_ENVIRONMENT=production."""
    monkeypatch.setenv("APP_ENVIRONMENT", "production")

    response = client.get(TEST_RUNTIME_ERROR_PATH)

    assert response.status_code == 500

    body = response.json()
    assert body["error"] is True
    assert body["code"] == 500
    assert body["message"] == "An internal server error occurred."
    assert "timestamp" in body

    # Belt-and-suspenders: check the raw response text too, not just the
    # "message" field, in case "Boom" ever leaked into another key.
    assert "Boom" not in response.text


def test_unhandled_runtime_error_surfaces_message_outside_production(monkeypatch):
    """Sanity check proving the masking assertion above is meaningful: the
    real exception message IS surfaced outside production, so the
    previous test's masking isn't just an artifact of the message always
    being replaced."""
    monkeypatch.setenv("APP_ENVIRONMENT", "development")

    response = client.get(TEST_RUNTIME_ERROR_PATH)

    assert response.status_code == 500
    body = response.json()
    assert body["error"] is True
    assert body["message"] == "Boom"


import sqlite3

TEST_SQLITE_LOCKED_PATH = "/api/v1/_test/raise-sqlite-locked"


@app.get(TEST_SQLITE_LOCKED_PATH, include_in_schema=False)
def _raise_sqlite_locked_for_testing():
    """Deliberately raise a sqlite3.OperationalError database is locked unhandled exception."""
    raise sqlite3.OperationalError("database is locked")


def test_sqlite_locked_returns_503_service_unavailable():
    """Verify that a sqlite3.OperationalError with 'locked' returns 503 Service Unavailable and a 'Service busy, please retry' message."""
    response = client.get(TEST_SQLITE_LOCKED_PATH)
    assert response.status_code == 503
    body = response.json()
    assert body["error"] is True
    assert body["code"] == 503
    assert body["message"] == "Service busy, please retry"
    assert "timestamp" in body


def test_unhandled_runtime_error_injects_request_id(monkeypatch):
    """Verify that global_exception_handler injects the X-Request-ID header value into the error response payload."""
    monkeypatch.setenv("APP_ENVIRONMENT", "production")

    response = client.get(
        TEST_RUNTIME_ERROR_PATH, headers={"X-Request-ID": "test-req-id-123"}
    )
    assert response.status_code == 500
    body = response.json()
    assert body["request_id"] == "test-req-id-123"
