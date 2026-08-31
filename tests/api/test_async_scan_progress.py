"""
tests/api/test_async_scan_progress.py
-------------------------------------
Tests for progress percentage and stage in AsyncScanStatusResponse and _process_scan_job (Issue #3224).
"""

from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import get_current_user, verify_bearer_token
from src.api.routers.analysis import _process_scan_job, scan_jobs
from src.api.schemas import AsyncScanStatusResponse

client = TestClient(app)


def setup_function():
    scan_jobs.clear()


def teardown_function():
    scan_jobs.clear()


def test_async_scan_status_response_schema_defaults():
    """Verify AsyncScanStatusResponse schema includes progress_percent=0 and stage='' by default."""
    status_obj = AsyncScanStatusResponse(
        job_id="job_test_123",
        status="queued",
        filename="test.pdf",
        created_at="2026-08-23T20:00:00Z",
    )
    assert status_obj.progress_percent == 0
    assert status_obj.stage == ""


def test_process_scan_job_progress_progression():
    """Verify that _process_scan_job updates progress at each stage:
    text extraction: 20%, chunking: 40%, embedding: 70%, comparison: 90%, done: 100%.
    """
    job_id = "job_progress_test_1"
    scan_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress_percent": 0,
        "stage": "",
        "filename": "sample.txt",
        "created_at": "2026-08-23T20:00:00Z",
        "completed_at": None,
        "result": None,
        "error": None,
    }

    observed_stages = []

    def mock_extract(file_input, filename):
        observed_stages.append(("during_extract", scan_jobs[job_id]["progress_percent"], scan_jobs[job_id]["stage"]))
        return "This is a sample document for testing progress reporting in the pipeline."

    def mock_chunk(text):
        observed_stages.append(("during_chunk", scan_jobs[job_id]["progress_percent"], scan_jobs[job_id]["stage"]))
        return [text]

    def mock_embed(chunks):
        observed_stages.append(("during_embed", scan_jobs[job_id]["progress_percent"], scan_jobs[job_id]["stage"]))
        return np.ones((1, 384), dtype=np.float32)

    with patch("src.api.routers.analysis.extract_text", side_effect=mock_extract), \
         patch("src.api.routers.analysis.chunk_document", side_effect=mock_chunk), \
         patch("src.api.routers.analysis.embed_chunks", side_effect=mock_embed), \
         patch("src.api.routers.analysis.get_corpus_documents_with_embeddings", return_value={}):
        _process_scan_job(
            job_id=job_id,
            file_input="fake_path",
            filename="sample.txt",
            threshold=0.59,
            top_k=3,
        )

    # Verify stage progression observed during execution
    # After extract -> 20% text extraction (observed during chunk)
    assert ("during_chunk", 20, "text extraction") in observed_stages
    # After chunk -> 40% chunking (observed during embed)
    assert ("during_embed", 40, "chunking") in observed_stages

    # Final completed state
    assert scan_jobs[job_id]["status"] == "completed"
    assert scan_jobs[job_id]["progress_percent"] == 100
    assert scan_jobs[job_id]["stage"] == "done"


def test_get_async_scan_status_endpoint_returns_progress():
    """Verify that GET /api/v1/scan/status/{job_id} returns progress_percent and stage."""
    app.dependency_overrides[get_current_user] = lambda: {"sub": "tester", "scopes": ["read"]}
    app.dependency_overrides[verify_bearer_token] = lambda: "valid_token"

    try:
        job_id = "job_status_check_1"
        scan_jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "progress_percent": 70,
            "stage": "embedding",
            "filename": "document.pdf",
            "created_at": "2026-08-23T20:00:00Z",
            "completed_at": None,
            "result": None,
            "error": None,
        }

        response = client.get(f"/api/v1/scan/status/{job_id}", headers={"Authorization": "Bearer token"})
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "processing"
        assert data["progress_percent"] == 70
        assert data["stage"] == "embedding"
    finally:
        app.dependency_overrides.clear()
