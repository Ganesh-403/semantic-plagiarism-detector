"""tests/api/test_scan_stage_metrics.py
--------------------------------------
Tests for Prometheus histogram tracking scan pipeline stages (parsing, chunking, embedding, matrix comparison).
Issue #3478.
"""

import io
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dependencies import get_current_user, validate_content_type, verify_bearer_token
from src.api.routers.analysis import _process_scan_job, scan_jobs
from src.core.metrics import spd_scan_duration_seconds

client = TestClient(app)


def _get_stage_count(stage: str) -> float:
    label_child = spd_scan_duration_seconds.labels(stage=stage)
    return sum(b.get() for b in label_child._buckets)


def test_process_scan_job_records_all_stages():
    """Verify that _process_scan_job records durations for parsing, chunking, embedding, and matrix comparison."""
    job_id = "job_test_metrics_1"
    scan_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "filename": "sample.txt",
        "created_at": "2026-08-23T20:00:00Z",
        "completed_at": None,
        "result": None,
        "error": None,
    }

    stages = ["parsing", "chunking", "embedding", "matrix comparison"]
    counts_before = {stage: _get_stage_count(stage) for stage in stages}

    sample_text = (
        "This is a sample document for testing the plagiarism detection scan pipeline. "
        "It contains enough words to generate chunks and embeddings for verification."
    )

    with patch("src.api.routers.analysis.extract_text", return_value=sample_text), patch(
        "src.api.routers.analysis.embed_chunks",
        return_value=np.ones((2, 384), dtype=np.float32),
    ), patch(
        "src.api.routers.analysis.get_corpus_documents_with_embeddings",
        return_value={},
    ):
        _process_scan_job(
            job_id=job_id,
            file_input="fake_file_path",
            filename="sample.txt",
            threshold=0.59,
            top_k=3,
        )

    assert scan_jobs[job_id]["status"] == "completed"

    for stage in stages:
        count_after = _get_stage_count(stage)
        assert count_after > counts_before[stage], f"Stage {stage} was not observed"
        assert spd_scan_duration_seconds.labels(stage=stage)._sum.get() >= 0


def test_scan_endpoint_records_all_stages():
    """Verify that POST /api/v1/scan records durations for all pipeline stages in spd_scan_duration_seconds."""
    app.dependency_overrides[get_current_user] = lambda: {"sub": "tester", "scopes": ["write"]}
    app.dependency_overrides[validate_content_type] = lambda: None
    app.dependency_overrides[verify_bearer_token] = lambda: "valid_token"

    try:
        stages = ["parsing", "chunking", "embedding", "matrix comparison"]
        counts_before = {stage: _get_stage_count(stage) for stage in stages}

        sample_text = (
            "Artificial intelligence and natural language processing techniques are widely used "
            "to detect similarity across text corpora."
        )

        with patch(
            "src.api.routers.analysis.stream_upload_file_to_disk", return_value="dummy_path.txt"
        ), patch(
            "src.api.routers.analysis.extract_text", return_value=sample_text
        ), patch(
            "src.api.routers.analysis.embed_chunks",
            return_value=np.ones((1, 384), dtype=np.float32),
        ), patch(
            "src.api.routers.analysis.get_corpus_documents_with_embeddings",
            return_value={},
        ), patch(
            "src.api.routers.analysis.calculate_file_sha256", return_value="abc123hash"
        ), patch(
            "src.api.routers.analysis.get_document_by_hash", return_value=None
        ), patch(
            "os.path.exists", return_value=False
        ):
            file_payload = {"file": ("test_doc.txt", io.BytesIO(b"dummy binary data"), "text/plain")}
            response = client.post(
                "/api/v1/scan",
                files=file_payload,
                headers={"Authorization": "Bearer test_token"},
            )

            assert response.status_code == 200

        for stage in stages:
            count_after = _get_stage_count(stage)
            assert count_after > counts_before[stage], f"Stage {stage} was not observed in /api/v1/scan"
    finally:
        app.dependency_overrides.clear()


def test_metrics_endpoint_contains_spd_scan_duration_seconds():
    """Verify that GET /metrics includes spd_scan_duration_seconds with stage labels."""
    # Ensure all stages have at least one observation
    for stage in ["parsing", "chunking", "embedding", "matrix comparison"]:
        spd_scan_duration_seconds.labels(stage=stage).observe(0.01)

    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text

    assert "spd_scan_duration_seconds" in content
    assert 'stage="parsing"' in content
    assert 'stage="chunking"' in content
    assert 'stage="embedding"' in content
    assert 'stage="matrix comparison"' in content
