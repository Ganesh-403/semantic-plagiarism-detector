import io
import json
import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.middleware import get_expected_bearer_token

client = TestClient(app)


@pytest.fixture(autouse=True)
def scoped_token(monkeypatch, mock_db):
    from src.api.middleware import get_valid_tokens
    monkeypatch.setenv("API_BEARER_TOKENS_MAPPING", json.dumps({"duplicate-test-token": ["write", "scan"]}))
    get_valid_tokens.cache_clear()
    yield
    get_valid_tokens.cache_clear()


@patch("src.api.routers.analysis.get_document_by_hash")
@patch("src.api.routers.analysis.calculate_file_sha256")
@patch("src.api.routers.analysis.get_corpus_documents_with_embeddings")
@patch("src.api.routers.analysis.embed_chunks")
def test_scan_duplicate_rejected(mock_embed, mock_corpus, mock_hash, mock_get_doc):
    """Verify that a duplicate upload returns 409 Conflict when reprocess=False."""
    mock_hash.return_value = "dummyhash"
    mock_get_doc.return_value = "existing_file.txt"

    expected_token = "duplicate-test-token"
    sample_content = b"Some duplicate text content."

    response = client.post(
        "/api/v1/scan",
        headers={"Authorization": f"Bearer {expected_token}"},
        files={"file": ("essay.txt", io.BytesIO(sample_content), "text/plain")},
    )

    assert response.status_code == 409
    assert response.json()["duplicate"] is True
    assert "already been uploaded" in response.json()["message"]


@patch("src.api.routers.analysis.get_document_by_hash")
@patch("src.api.routers.analysis.calculate_file_sha256")
@patch("src.api.routers.analysis.get_corpus_documents_with_embeddings")
@patch("src.api.routers.analysis.embed_chunks")
def test_scan_duplicate_reprocess(mock_embed, mock_corpus, mock_hash, mock_get_doc):
    """Verify that a duplicate upload with reprocess=True succeeds."""
    mock_hash.return_value = "dummyhash"
    mock_get_doc.return_value = "existing_file.txt"

    import numpy as np

    mock_embed.return_value = np.ones((1, 384), dtype=np.float32)
    mock_corpus.return_value = {}

    expected_token = "duplicate-test-token"
    sample_content = b"Some duplicate text content."

    response = client.post(
        "/api/v1/scan?reprocess=true",
        headers={"Authorization": f"Bearer {expected_token}"},
        files={"file": ("essay.txt", io.BytesIO(sample_content), "text/plain")},
    )

    assert response.status_code == 200
    assert "plagiarism_flagged" in response.json()


@patch("src.api.routers.analysis.get_document_by_hash")
@patch("src.api.routers.analysis.calculate_file_sha256")
def test_scan_async_duplicate_rejected(mock_hash, mock_get_doc):
    """Verify that async duplicate upload returns 409 Conflict when reprocess=False."""
    mock_hash.return_value = "dummyhash"
    mock_get_doc.return_value = "existing_file.txt"

    expected_token = "duplicate-test-token"
    sample_content = b"Some duplicate text content."

    response = client.post(
        "/api/v1/scan/async",
        headers={"Authorization": f"Bearer {expected_token}"},
        files={"file": ("essay.txt", io.BytesIO(sample_content), "text/plain")},
    )

    assert response.status_code == 409
    assert response.json()["duplicate"] is True
    assert "already been uploaded" in response.json()["message"]


@patch("src.api.routers.analysis.get_document_by_hash")
@patch("src.api.routers.analysis.calculate_file_sha256")
def test_scan_async_duplicate_reprocess(mock_hash, mock_get_doc):
    """Verify that async duplicate upload with reprocess=True succeeds."""
    mock_hash.return_value = "dummyhash"
    mock_get_doc.return_value = "existing_file.txt"

    expected_token = "duplicate-test-token"
    sample_content = b"Some duplicate text content."

    response = client.post(
        "/api/v1/scan/async?reprocess=true",
        headers={"Authorization": f"Bearer {expected_token}"},
        files={"file": ("essay.txt", io.BytesIO(sample_content), "text/plain")},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
