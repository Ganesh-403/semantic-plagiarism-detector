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
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


@pytest.mark.unit
def test_openapi_schema_responses():
    """Verify that OpenAPI schema defines response models for API endpoints."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    openapi = response.json()
    schemas = openapi.get("components", {}).get("schemas", {})

    # Check that Pydantic response models exist in OpenAPI schemas
    expected_schemas = [
        "LoginResponse",
        "HealthCheckResponse",
        "HealthzResponse",
        "SimilarityCheckResponse",
        "MatchedDocument",
        "FlaggedChunkMatch",
        "ClearDataResponse",
        "ErrorResponse",
    ]
    for schema_name in expected_schemas:
        assert (
            schema_name in schemas
        ), f"Schema {schema_name} missing from OpenAPI specification"

    paths = openapi.get("paths", {})

    # Verify /api/v1/auth/login response codes
    login_responses = (
        paths.get("/api/v1/auth/login", {}).get("post", {}).get("responses", {})
    )
    assert "200" in login_responses
    assert "400" in login_responses
    assert "401" in login_responses
    assert "500" in login_responses

    # Verify /api/v1/scan response codes
    scan_responses = paths.get("/api/v1/scan", {}).get("post", {}).get("responses", {})
    assert "200" in scan_responses
    assert "400" in scan_responses
    assert "422" in scan_responses
    assert "500" in scan_responses

    # Verify /api/v1/clear response codes
    clear_responses = (
        paths.get("/api/v1/clear", {}).get("post", {}).get("responses", {})
    )
    assert "200" in clear_responses
    assert "403" in clear_responses
    assert "500" in clear_responses
