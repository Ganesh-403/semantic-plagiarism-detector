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

"""Tests for RFC 7807 Problem Details compliance in API error responses (Issue #2922)."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import get_current_user
from src.api.middleware import verify_bearer_token

client = TestClient(app)


@pytest.fixture(autouse=True)
def _bypass_auth():
    app.dependency_overrides[verify_bearer_token] = lambda: "test-token"
    app.dependency_overrides[get_current_user] = lambda: {
        "username": "admin",
        "role": "admin",
        "scopes": ["read", "write"],
    }
    yield
    app.dependency_overrides.pop(verify_bearer_token, None)
    app.dependency_overrides.pop(get_current_user, None)


def test_rfc7807_404_not_found():
    """Verify 404 responses conform to RFC 7807 Problem Details schema."""
    response = client.get("/api/v1/nonexistent-endpoint-abc")
    assert response.status_code == 404
    assert response.headers.get("content-type") == "application/problem+json"

    data = response.json()
    assert data["type"] == "about:blank"
    assert data["title"] == "Not Found"
    assert data["status"] == 404
    assert data["detail"] == "API endpoint or resource not found"
    assert data["instance"] == "/api/v1/nonexistent-endpoint-abc"


def test_rfc7807_422_validation_error():
    """Verify 422 validation errors conform to RFC 7807 Problem Details schema."""
    response = client.post(
        "/api/v1/scan",
        headers={
            "Authorization": "Bearer dummy-token",
            "Content-Type": "multipart/form-data; boundary=----WebKitFormBoundaryTest",
        },
    )
    assert response.status_code == 422
    assert response.headers.get("content-type") == "application/problem+json"

    data = response.json()
    assert data["type"] == "about:blank"
    assert data["title"] == "Unprocessable Entity"
    assert data["status"] == 422
    assert data["detail"] == "Validation failed."
    assert "details" in data or "invalid_params" in data


def test_rfc7807_400_value_error():
    """Verify 400 ValueError responses conform to RFC 7807 Problem Details schema."""
    from fastapi import APIRouter

    test_router = APIRouter()

    @test_router.get("/api/v1/_test/rfc7807-value-error")
    def trigger_value_error():
        raise ValueError("Invalid query parameter provided.")

    app.include_router(test_router)

    response = client.get("/api/v1/_test/rfc7807-value-error")
    assert response.status_code == 400
    assert response.headers.get("content-type") == "application/problem+json"

    data = response.json()
    assert data["type"] == "about:blank"
    assert data["title"] == "Bad Request"
    assert data["status"] == 400
    assert data["detail"] == "Invalid query parameter provided."
    assert data["instance"] == "/api/v1/_test/rfc7807-value-error"
