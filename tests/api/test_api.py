"""
tests/test_api.py
------------------
Unit tests for the FastAPI REST API module (`src/api/app.py`).
Tests healthcheck, Bearer token authentication, and document scanning endpoint.
"""

import io
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app, get_expected_bearer_token

client = TestClient(app)


def test_health_check():
    """Verify that GET /health returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Semantic Plagiarism Detector API" in data["service"]


def test_scan_no_auth_header():
    """Verify that requests without Authorization header return 401/403 error."""
    response = client.post(
        "/api/v1/scan",
        files={"file": ("test.txt", b"Sample essay text for testing.")},
    )
    assert response.status_code in (401, 403)


def test_scan_invalid_bearer_token():
    """Verify that requests with an invalid Bearer token return 401 Unauthorized."""
    response = client.post(
        "/api/v1/scan",
        headers={"Authorization": "Bearer wrong-token-value"},
        files={"file": ("test.txt", b"Sample essay text for testing.")},
    )
    assert response.status_code == 401
    assert "Invalid or missing" in response.json()["detail"]


@patch("src.api.app.get_corpus_documents_with_embeddings")
@patch("src.api.app.embed_chunks")
def test_scan_valid_file_success(mock_embed, mock_corpus):
    """Verify successful document scan with valid Bearer token."""
    # Mock embedding output: 1 chunk x 384 dim vector
    mock_embed.return_value = np.ones((1, 384), dtype=np.float32)

    # Mock empty corpus
    mock_corpus.return_value = {}

    expected_token = get_expected_bearer_token()

    sample_content = b"Artificial Intelligence and Machine Learning are transforming modern higher education and academic integrity."

    response = client.post(
        "/api/v1/scan",
        headers={"Authorization": f"Bearer {expected_token}"},
        files={"file": ("essay.txt", io.BytesIO(sample_content), "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "essay.txt"
    assert data["chunk_count"] >= 1
    assert "plagiarism_flagged" in data
    assert "matched_documents" in data
    assert isinstance(data["matched_documents"], list)


@patch("src.api.app.get_corpus_documents_with_embeddings")
@patch("src.api.app.embed_chunks")
def test_scan_matching_corpus_flag(mock_embed, mock_corpus):
    """Verify scanning against matching corpus document returns plagiarism flag."""
    dummy_vec = np.ones((1, 384), dtype=np.float32)
    mock_embed.return_value = dummy_vec

    # Mock corpus document with identical embedding
    mock_corpus.return_value = {
        "existing_essay.txt": {
            "chunks": [
                "Artificial Intelligence and Machine Learning are transforming modern higher education."
            ],
            "embeddings": dummy_vec,
        }
    }

    expected_token = get_expected_bearer_token()
    sample_content = b"Artificial Intelligence and Machine Learning are transforming modern higher education."

    response = client.post(
        "/api/v1/scan?threshold=0.5",
        headers={"Authorization": f"Bearer {expected_token}"},
        files={"file": ("submitted.txt", io.BytesIO(sample_content), "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["plagiarism_flagged"] is True
    assert data["matched_documents_count"] == 1
    assert data["matched_documents"][0]["filename"] == "existing_essay.txt"
    assert data["matched_documents"][0]["max_chunk_similarity_score"] >= 0.90


def test_scan_empty_file_upload():
    """Verify that uploading an empty file returns 400 Bad Request."""
    expected_token = get_expected_bearer_token()
    response = client.post(
        "/api/v1/scan",
        headers={"Authorization": f"Bearer {expected_token}"},
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


def test_clear_all_documents_no_auth_header():
    """Verify that requests to POST /api/v1/clear without Authorization header return 401/403."""
    response = client.post("/api/v1/clear?username=admin")
    assert response.status_code in (401, 403)


def test_clear_all_documents_invalid_token():
    """Verify that requests to POST /api/v1/clear with invalid Bearer token return 401."""
    response = client.post(
        "/api/v1/clear?username=admin",
        headers={"Authorization": "Bearer wrong-token-value"},
    )
    assert response.status_code == 401


@patch("src.api.app.get_user_role")
def test_clear_all_documents_non_admin_forbidden(mock_get_role):
    """Verify that a non-administrator receives 403 Forbidden on POST /api/v1/clear."""
    mock_get_role.return_value = "teacher"

    expected_token = get_expected_bearer_token()
    response = client.post(
        "/api/v1/clear?username=teacher_user",
        headers={"Authorization": f"Bearer {expected_token}"},
    )

    assert response.status_code == 403
    assert "Forbidden" in response.json()["detail"]


@patch("src.api.app.get_user_role")
@patch("src.api.app.clear_all_data")
@patch("os.path.exists")
@patch("os.remove")
def test_clear_all_documents_admin_success(
    mock_remove, mock_exists, mock_clear_db, mock_get_role
):
    """Verify that an administrator can successfully clear all documents."""
    mock_get_role.return_value = "admin"
    mock_exists.return_value = True

    expected_token = get_expected_bearer_token()
    response = client.post(
        "/api/v1/clear?username=admin",
        headers={"Authorization": f"Bearer {expected_token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "cleared" in data["message"]

    mock_clear_db.assert_called_once()
    mock_exists.assert_called_once()
    mock_remove.assert_called_once()


@patch("src.api.app.get_user_role")
@patch("src.api.app.clear_all_data")
@patch("os.path.exists")
@patch("os.remove")
def test_clear_all_documents_already_empty(
    mock_remove, mock_exists, mock_clear_db, mock_get_role
):
    """Verify that clearing an already empty database behaves safely (index doesn't exist)."""
    mock_get_role.return_value = "admin"
    mock_exists.return_value = False

    expected_token = get_expected_bearer_token()
    response = client.post(
        "/api/v1/clear?username=admin",
        headers={"Authorization": f"Bearer {expected_token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    mock_clear_db.assert_called_once()
    mock_exists.assert_called_once()
    mock_remove.assert_not_called()
