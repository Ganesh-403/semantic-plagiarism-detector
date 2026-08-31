"""
test_scan_jobs_redis_issue_3222.py
------------------------------------
Unit test suite for Issue #3222:
Validates that scan_jobs are persisted in Redis under `spd:v1:scan_jobs:{job_id}` with a 24-hour TTL,
allowing status checks across multiple Uvicorn workers, with fallback to in-memory dict when Redis is unavailable.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routers.analysis import scan_jobs, _get_scan_job, _set_scan_job
from src.utils.redis_cache import CacheNamespace, SCAN_JOBS_TTL


@pytest.fixture(autouse=True)
def clean_jobs():
    """Clear in-memory and mocked jobs before and after each test."""
    scan_jobs.clear()
    yield
    scan_jobs.clear()


def test_scan_job_redis_key_format():
    """Verify CacheNamespace.SCAN_JOBS builds key 'spd:v1:scan_jobs:{job_id}'."""
    job_id = "job_abc123def456"
    expected_key = "spd:v1:scan_jobs:job_abc123def456"
    assert CacheNamespace.SCAN_JOBS.build_key(job_id) == expected_key


def test_set_and_get_scan_job_with_redis_available():
    """Verify _set_scan_job and _get_scan_job store and retrieve from Redis."""
    job_id = "job_test_001"
    job_data = {
        "job_id": job_id,
        "status": "queued",
        "filename": "test_doc.pdf",
        "created_at": "2026-08-22T12:00:00Z",
        "completed_at": None,
        "result": None,
        "error": None,
    }

    mock_cache = MagicMock()
    mock_cache.is_available.return_value = True
    mock_cache.get_json.return_value = job_data

    with patch("src.api.routers.analysis.get_cache", return_value=mock_cache):
        _set_scan_job(job_id, job_data)
        mock_cache.set_json.assert_called_once_with(
            f"spd:v1:scan_jobs:{job_id}", job_data, ttl=SCAN_JOBS_TTL
        )

        retrieved = _get_scan_job(job_id)
        assert retrieved == job_data
        mock_cache.get_json.assert_called_once_with(f"spd:v1:scan_jobs:{job_id}")


def test_get_scan_job_cross_worker_retrieval():
    """Verify job stored in Redis is retrievable even if in-memory dict is empty (multi-worker simulation)."""
    job_id = "job_worker_b_check"
    job_data = {
        "job_id": job_id,
        "status": "completed",
        "filename": "worker_a_file.pdf",
        "created_at": "2026-08-22T12:00:00Z",
        "completed_at": "2026-08-22T12:00:05Z",
        "result": {"similarity": 0.85},
        "error": None,
    }

    # Worker B has empty in-memory dictionary
    assert job_id not in scan_jobs

    mock_cache = MagicMock()
    mock_cache.is_available.return_value = True
    mock_cache.get_json.return_value = job_data

    with patch("src.api.routers.analysis.get_cache", return_value=mock_cache):
        retrieved = _get_scan_job(job_id)
        assert retrieved == job_data
        assert retrieved["status"] == "completed"


def test_get_scan_job_fallback_when_redis_unavailable():
    """Verify fallback to in-memory scan_jobs dict when Redis is down."""
    job_id = "job_fallback_001"
    job_data = {
        "job_id": job_id,
        "status": "processing",
        "filename": "fallback.txt",
        "created_at": "2026-08-22T12:00:00Z",
    }
    scan_jobs[job_id] = job_data

    mock_cache = MagicMock()
    mock_cache.is_available.return_value = False

    with patch("src.api.routers.analysis.get_cache", return_value=mock_cache):
        retrieved = _get_scan_job(job_id)
        assert retrieved == job_data


def test_get_async_scan_status_endpoint_from_redis():
    """Verify /api/v1/scan/status/{job_id} retrieves status from Redis."""
    job_id = "job_endpoint_123"
    job_data = {
        "job_id": job_id,
        "status": "completed",
        "filename": "report.pdf",
        "created_at": "2026-08-22T12:00:00Z",
        "completed_at": "2026-08-22T12:00:02Z",
        "result": {
            "filename": "report.pdf",
            "word_count": 120,
            "chunk_count": 2,
            "plagiarism_flagged": False,
            "threshold_used": 0.59,
            "plagiarism_density": 0,
            "overall_document_similarity": 0.12,
            "max_chunk_similarity": 0.15,
            "matched_documents_count": 0,
            "matched_documents": [],
        },
        "error": None,
    }

    client = TestClient(app)

    mock_cache = MagicMock()
    mock_cache.is_available.return_value = True
    mock_cache.get_json.return_value = job_data

    with patch("src.api.routers.analysis.get_cache", return_value=mock_cache), \
         patch("src.api.middleware.verify_bearer_token", return_value="token123"), \
         patch("src.api.middleware.get_valid_tokens", return_value={"token123": ["read", "write"]}):
        response = client.get(
            f"/api/v1/scan/status/{job_id}",
            headers={"Authorization": "Bearer token123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "completed"
        assert data["filename"] == "report.pdf"


def test_get_async_scan_status_not_found():
    """Verify /api/v1/scan/status/{job_id} returns 404 when job does not exist in Redis or memory."""
    client = TestClient(app)

    mock_cache = MagicMock()
    mock_cache.is_available.return_value = True
    mock_cache.get_json.return_value = None

    with patch("src.api.routers.analysis.get_cache", return_value=mock_cache), \
         patch("src.api.middleware.verify_bearer_token", return_value="token123"), \
         patch("src.api.middleware.get_valid_tokens", return_value={"token123": ["read", "write"]}):
        response = client.get(
            "/api/v1/scan/status/non_existent_job",
            headers={"Authorization": "Bearer token123"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
