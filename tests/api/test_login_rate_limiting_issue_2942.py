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
test_login_rate_limiting_issue_2942.py
---------------------------------------
Unit test suite for Issue #2942:
Validates that slowapi rate limiting is strictly applied to the login endpoint (/auth/login and /api/v1/auth/login),
limiting requests to 5 per minute per IP address and returning HTTP 429 Too Many Requests when exceeded.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_login_endpoint_exists_and_rate_limited(client):
    """Verify login endpoints allow up to 5 requests per minute and return 429 on 6th attempt."""
    # Reset limiter storage if available to ensure clean test state
    if hasattr(app.state, "limiter") and hasattr(app.state.limiter, "reset"):
        app.state.limiter.reset()

    headers = {"X-Forwarded-For": "198.51.100.42"}

    # First 5 requests should succeed (200 OK)
    for i in range(5):
        response = client.post("/auth/login", headers=headers)
        assert response.status_code == 200, f"Request {i+1} should succeed"

    # 6th request should fail with 429 Too Many Requests
    response = client.post("/auth/login", headers=headers)
    assert (
        response.status_code == 429
    ), "6th request within minute should be rate-limited"
    data = response.json()
    assert data.get("status") == 429 or "Rate limit exceeded" in str(data)


def test_api_v1_login_endpoint_rate_limited(client):
    """Verify /api/v1/auth/login endpoint is also protected by slowapi rate limiting."""
    headers = {"X-Forwarded-For": "198.51.100.43"}

    # First 5 requests succeed
    for i in range(5):
        response = client.post("/api/v1/auth/login", headers=headers)
        assert response.status_code == 200

    # 6th request returns 429
    response = client.post("/api/v1/auth/login", headers=headers)
    assert response.status_code == 429
