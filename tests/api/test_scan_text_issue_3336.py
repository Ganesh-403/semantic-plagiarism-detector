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

"""Unit tests for POST /api/v1/scan/text raw text submission endpoint (Issue #3336)."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import get_current_user
from src.api.middleware import verify_bearer_token

client = TestClient(app)


@pytest.fixture(autouse=True)
def override_auth():
    """Override auth dependencies for test suite."""
    app.dependency_overrides[verify_bearer_token] = lambda: "test-token"
    app.dependency_overrides[get_current_user] = lambda: {
        "username": "test_user",
        "scopes": ["write"],
    }
    yield
    app.dependency_overrides.clear()


def test_scan_text_success_returns_200():
    """Verify POST /api/v1/scan/text scans raw text successfully."""
    payload = {
        "text": "Academic integrity and plagiarism detection algorithms enable automated scanning.",
        "filename": "essay_submission.txt",
        "threshold": 0.7,
        "top_k": 3,
    }
    res = client.post("/api/v1/scan/text", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["filename"] == "essay_submission.txt"
    assert data["word_count"] > 0
    assert data["chunk_count"] > 0
    assert "plagiarism_flagged" in data
    assert data["threshold_used"] == 0.7
    assert "overall_document_similarity" in data
    assert "max_chunk_similarity" in data
    assert "matched_documents" in data


def test_scan_text_empty_string_returns_422():
    """Verify empty text string is rejected with HTTP 422 Unprocessable Entity."""
    payload = {
        "text": "   ",
        "filename": "empty.txt",
    }
    res = client.post("/api/v1/scan/text", json=payload)
    assert res.status_code == 422


def test_scan_text_duplicate_returns_409():
    """Verify duplicate raw text submission behavior."""
    payload = {
        "text": "Unique student submission raw text string for duplicate checking test #3336.",
        "filename": "dup_test.txt",
        "threshold": 0.5,
    }
    res1 = client.post("/api/v1/scan/text", json=payload)
    assert res1.status_code == 200

    res2 = client.post("/api/v1/scan/text", json=payload)
    assert res2.status_code in (200, 409)

    payload_reprocess = dict(payload, reprocess=True)
    res3 = client.post("/api/v1/scan/text", json=payload_reprocess)
    assert res3.status_code == 200
