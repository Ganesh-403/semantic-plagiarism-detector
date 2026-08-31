"""
tests/integration/test_task_queue.py
------------------------------------
Integration tests for the distributed task queue (Issue #3146).

Verifies:
  - Job submission via the TaskQueue and the REST API.
  - Worker execution (ScanWorker pulls + executes + reports).
  - State transitions: PENDING → PROCESSING → COMPLETED.
  - Retry logic: FAILED jobs are re-queued up to max_retries.
  - Dead-letter handling: exhausted retries → DEAD_LETTER.
  - REST endpoints: POST /batch-scan, GET /{job_id}, GET /, dead-letter, retry.

All tests use an isolated temp SQLite database (via TASK_QUEUE_DB_PATH env var)
so they never touch the production data directory. The embedding pipeline
is mocked so tests run without the sentence-transformers model.
"""

from __future__ import annotations

import base64
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ── Fixture: isolated DB path ──────────────────────────────────

@pytest.fixture
def task_db_path(tmp_path, monkeypatch):
    """Point task_db at an isolated temp database for each test."""
    db_path = tmp_path / "task_queue_test.db"
    monkeypatch.setenv("TASK_QUEUE_DB_PATH", str(db_path))
    # Force re-creation of the connection pool.
    from src.db import task_db
    task_db._cleanup_all_connections()
    # Remove the thread-local so _get_connection creates a fresh one.
    if hasattr(task_db._connection_pool, "conn"):
        del task_db._connection_pool.conn
    task_db.reset_db(db_path)
    yield db_path
    task_db._cleanup_all_connections()
    if hasattr(task_db._connection_pool, "conn"):
        del task_db._connection_pool.conn


# ── Fixture: mocked embedding pipeline ─────────────────────────

@pytest.fixture
def mock_pipeline():
    """Mock the heavy ML functions so tests run without the model."""
    def mock_extract_text(file_bytes, filename, **kwargs):
        return f"Extracted text from {filename}. This is sample content for testing."

    def mock_chunk_documents(documents, **kwargs):
        return {name: [text[:100], text[100:]] for name, text in documents.items()}

    def mock_embed_chunks(chunks, **kwargs):
        import numpy as np
        return np.random.rand(len(chunks), 384).astype("float32")

    def mock_document_similarity_matrix(emb_matrix):
        import numpy as np
        n = emb_matrix.shape[0]
        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                mat[i][j] = mat[j][i] = 0.5
        return mat

    with patch("src.core.document_parser.extract_text", side_effect=mock_extract_text), \
         patch("src.core.text_chunking.chunk_documents", side_effect=mock_chunk_documents), \
         patch("src.core.embedding_model.embed_chunks", side_effect=mock_embed_chunks), \
         patch("src.core.similarity.document_similarity_matrix", side_effect=mock_document_similarity_matrix):
        yield


# ── Fixture: FastAPI test app with the tasks router ───────────

@pytest.fixture
def api_client():
    from src.api.endpoints.tasks import router as tasks_router
    app = FastAPI()
    app.include_router(tasks_router)
    yield TestClient(app)


# ════════════════════════════════════════════════════════════════
# 1. Task DB — state transitions
# ════════════════════════════════════════════════════════════════

class TestTaskDB:
    def test_create_job_returns_pending(self, task_db_path):
        from src.db import task_db
        job = task_db.create_job({"files": {"test.txt": "aGVsbG8="}})
        assert job["status"] == "PENDING"
        assert job["id"]
        assert job["payload"]["files"]["test.txt"] == "aGVsbG8="

    def test_get_job_returns_none_for_missing_id(self, task_db_path):
        from src.db import task_db
        assert task_db.get_job("nonexistent-uuid") is None

    def test_claim_next_job_flips_to_processing(self, task_db_path):
        from src.db import task_db
        task_db.create_job({"files": {"test.txt": "aGVsbG8="}})
        job = task_db.claim_next_job("worker-1")
        assert job is not None
        assert job["status"] == "PROCESSING"
        assert job["worker_id"] == "worker-1"
        assert job["started_at"] is not None

    def test_claim_next_job_returns_none_when_empty(self, task_db_path):
        from src.db import task_db
        assert task_db.claim_next_job("worker-1") is None

    def test_claim_next_job_fifo_order(self, task_db_path):
        from src.db import task_db
        j1 = task_db.create_job({"order": 1})
        j2 = task_db.create_job({"order": 2})
        claimed1 = task_db.claim_next_job("w1")
        assert claimed1["id"] == j1["id"]
        claimed2 = task_db.claim_next_job("w2")
        assert claimed2["id"] == j2["id"]

    def test_mark_completed_sets_result(self, task_db_path):
        from src.db import task_db
        job = task_db.create_job({"files": {"test.txt": "aGVsbG8="}})
        result = {"flagged_pairs": 1, "documents_processed": 1}
        task_db.mark_completed(job["id"], result)
        updated = task_db.get_job(job["id"])
        assert updated["status"] == "COMPLETED"
        assert updated["result"]["flagged_pairs"] == 1
        assert updated["completed_at"] is not None

    def test_mark_failed_requeues_until_max_retries(self, task_db_path):
        from src.db import task_db
        job = task_db.create_job({"files": {"test.txt": "aGVsbG8="}}, max_retries=2)
        # First failure → retry_count=1, status back to PENDING.
        task_db.mark_failed(job["id"], "error 1")
        j = task_db.get_job(job["id"])
        assert j["status"] == "PENDING"
        assert j["retry_count"] == 1
        assert j["error"] == "error 1"

        # Claim and fail again → retry_count=2, now dead-lettered.
        claimed = task_db.claim_next_job("w1")
        assert claimed["id"] == job["id"]
        task_db.mark_failed(job["id"], "error 2")
        j = task_db.get_job(job["id"])
        assert j["status"] == "DEAD_LETTER"
        assert j["retry_count"] == 2
        assert j["error"] == "error 2"

    def test_mark_dead_letter_immediately(self, task_db_path):
        from src.db import task_db
        job = task_db.create_job({"files": {"test.txt": "aGVsbG8="}})
        task_db.mark_dead_letter(job["id"], "fatal error")
        j = task_db.get_job(job["id"])
        assert j["status"] == "DEAD_LETTER"
        assert j["error"] == "fatal error"

    def test_list_jobs_filter_by_status(self, task_db_path):
        from src.db import task_db
        task_db.create_job({"a": 1})
        task_db.create_job({"b": 2})
        j = task_db.claim_next_job("w1")  # flips one to PROCESSING
        pending = task_db.list_jobs(status="PENDING")
        assert len(pending) == 1
        processing = task_db.list_jobs(status="PROCESSING")
        assert len(processing) == 1
        assert processing[0]["id"] == j["id"]

    def test_get_dead_letter_jobs(self, task_db_path):
        from src.db import task_db
        job = task_db.create_job({"files": {"x": "x"}}, max_retries=1)
        task_db.claim_next_job("w1")
        task_db.mark_failed(job["id"], "boom")
        dl = task_db.get_dead_letter_jobs()
        assert len(dl) == 1
        assert dl[0]["id"] == job["id"]


# ════════════════════════════════════════════════════════════════
# 2. TaskQueue — producer/consumer
# ════════════════════════════════════════════════════════════════

class TestTaskQueue:
    def test_enqueue_creates_pending_job(self, task_db_path):
        from src.workers.task_queue import TaskQueue
        q = TaskQueue()
        job = q.enqueue({"files": {"test.txt": "aGVsbG8="}})
        assert job["status"] == "PENDING"
        assert job["id"]

    def test_dequeue_claims_job(self, task_db_path):
        from src.workers.task_queue import TaskQueue
        q = TaskQueue(worker_id="test-w")
        q.enqueue({"files": {"test.txt": "aGVsbG8="}})
        job = q.dequeue(timeout=2.0)
        assert job is not None
        assert job["status"] == "PROCESSING"
        assert job["worker_id"] == "test-w"

    def test_complete_sets_completed(self, task_db_path):
        from src.workers.task_queue import TaskQueue
        from src.db import task_db
        q = TaskQueue()
        job = q.enqueue({"files": {"test.txt": "aGVsbG8="}})
        q.complete(job["id"], {"result": "ok"})
        j = task_db.get_job(job["id"])
        assert j["status"] == "COMPLETED"
        assert j["result"]["result"] == "ok"

    def test_fail_retries_then_dead_letters(self, task_db_path):
        from src.workers.task_queue import TaskQueue
        from src.db import task_db
        q = TaskQueue()
        job = q.enqueue({"files": {"x": "x"}}, max_retries=2)
        q.fail(job["id"], "error 1")
        j = task_db.get_job(job["id"])
        assert j["status"] == "PENDING"  # re-queued for retry
        # Claim + fail again → dead-letter
        task_db.claim_next_job("w1")
        q.fail(job["id"], "error 2")
        j = task_db.get_job(job["id"])
        assert j["status"] == "DEAD_LETTER"

    def test_requeue_stale_processing(self, task_db_path):
        from src.workers.task_queue import TaskQueue
        from src.db import task_db
        # Create + claim a job so it's PROCESSING.
        task_db.create_job({"files": {"x": "x"}})
        task_db.claim_next_job("dead-worker")
        stale = task_db.list_jobs(status="PROCESSING")
        assert len(stale) == 1
        q = TaskQueue()
        count = q.requeue_stale_processing()
        assert count == 1
        # The stale job should now be PENDING (or DEAD_LETTER if retries exhausted).
        pending = task_db.list_jobs(status="PENDING")
        dl = task_db.list_jobs(status="DEAD_LETTER")
        assert len(pending) + len(dl) == 1


# ════════════════════════════════════════════════════════════════
# 3. ScanWorker — execution + state transitions
# ════════════════════════════════════════════════════════════════

class TestScanWorker:
    def test_worker_completes_job(self, task_db_path, mock_pipeline):
        from src.workers.scan_worker import ScanWorker
        from src.workers.task_queue import TaskQueue
        from src.db import task_db

        q = TaskQueue()
        q.enqueue({
            "files": {"test.txt": base64.b64encode(b"Hello world").decode()},
            "threshold": 0.5,
        })

        worker = ScanWorker(queue=q)
        worker.start()
        # Poll for completion.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            jobs = task_db.list_jobs(status="COMPLETED")
            if jobs:
                break
            time.sleep(0.2)
        worker.stop()

        jobs = task_db.list_jobs(status="COMPLETED")
        assert len(jobs) == 1
        assert jobs[0]["result"]["documents_processed"] == 1
        assert jobs[0]["result"]["total_chunks"] > 0

    def test_worker_retries_on_failure(self, task_db_path):
        from src.workers.scan_worker import ScanWorker, execute_scan_job
        from src.workers.task_queue import TaskQueue
        from src.db import task_db

        q = TaskQueue()
        q.enqueue({"files": {}}, max_retries=2)  # empty files → ValueError

        worker = ScanWorker(queue=q)
        worker.start()
        deadline = time.time() + 10.0
        while time.time() < deadline:
            dl = task_db.list_jobs(status="DEAD_LETTER")
            if dl:
                break
            time.sleep(0.2)
        worker.stop()

        dl = task_db.list_jobs(status="DEAD_LETTER")
        assert len(dl) == 1
        assert "ValueError" in dl[0]["error"]


# ════════════════════════════════════════════════════════════════
# 4. REST API endpoints
# ════════════════════════════════════════════════════════════════

class TestTasksAPI:
    def test_submit_batch_scan_returns_202(self, task_db_path, api_client):
        response = api_client.post(
            "/api/v1/tasks/batch-scan",
            json={
                "files": [
                    {"filename": "test1.txt", "content_base64": base64.b64encode(b"Hello").decode()},
                ],
                "threshold": 0.6,
            },
        )
        assert response.status_code == 202
        body = response.json()
        assert body["job_id"]
        assert body["status"] == "PENDING"

    def test_submit_batch_scan_rejects_empty_files(self, task_db_path, api_client):
        response = api_client.post(
            "/api/v1/tasks/batch-scan",
            json={"files": []},
        )
        assert response.status_code == 422  # Pydantic min_length=1

    def test_get_job_status_returns_404_for_missing(self, task_db_path, api_client):
        response = api_client.get("/api/v1/tasks/nonexistent-uuid")
        assert response.status_code == 404

    def test_get_job_status_returns_job(self, task_db_path, api_client):
        from src.db import task_db
        job = task_db.create_job({"files": {"test.txt": "aGVsbG8="}})
        response = api_client.get(f"/api/v1/tasks/{job['id']}")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == job["id"]
        assert body["status"] == "PENDING"

    def test_list_jobs_returns_all(self, task_db_path, api_client):
        from src.db import task_db
        task_db.create_job({"a": 1})
        task_db.create_job({"b": 2})
        response = api_client.get("/api/v1/tasks")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2

    def test_list_jobs_filter_by_status(self, task_db_path, api_client):
        from src.db import task_db
        task_db.create_job({"a": 1})
        task_db.claim_next_job("w1")  # → PROCESSING
        task_db.create_job({"b": 2})  # → PENDING
        response = api_client.get("/api/v1/tasks?status=PENDING")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1

    def test_list_dead_letter(self, task_db_path, api_client):
        from src.db import task_db
        job = task_db.create_job({"x": "x"}, max_retries=1)
        task_db.claim_next_job("w1")
        task_db.mark_failed(job["id"], "boom")
        response = api_client.get("/api/v1/tasks/dead-letter/list")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["jobs"][0]["status"] == "DEAD_LETTER"

    def test_retry_dead_letter_job(self, task_db_path, api_client):
        from src.db import task_db
        job = task_db.create_job({"files": {"x": "x"}}, max_retries=1)
        task_db.claim_next_job("w1")
        task_db.mark_failed(job["id"], "boom")
        response = api_client.post(f"/api/v1/tasks/{job['id']}/retry")
        assert response.status_code == 200
        body = response.json()
        assert body["job_id"] != job["id"]  # new job created
        assert body["status"] == "PENDING"

    def test_retry_non_dead_letter_returns_409(self, task_db_path, api_client):
        from src.db import task_db
        job = task_db.create_job({"x": "x"})
        response = api_client.post(f"/api/v1/tasks/{job['id']}/retry")
        assert response.status_code == 409

    def test_invalid_status_filter_returns_400(self, task_db_path, api_client):
        response = api_client.get("/api/v1/tasks?status=INVALID")
        assert response.status_code == 400


# ════════════════════════════════════════════════════════════════
# 5. execute_scan_job — pure function tests
# ════════════════════════════════════════════════════════════════

class TestExecuteScanJob:
    def test_raises_on_empty_files(self):
        from src.workers.scan_worker import execute_scan_job
        with pytest.raises(ValueError, match="non-empty"):
            execute_scan_job({"files": {}})

    def test_raises_on_no_extracted_text(self, mock_pipeline):
        from src.workers.scan_worker import execute_scan_job
        with patch("src.core.document_parser.extract_text", return_value=""):
            with pytest.raises(ValueError, match="No text could be extracted"):
                execute_scan_job({"files": {"test.txt": base64.b64encode(b"").decode()}})

    def test_returns_result_with_mocked_pipeline(self, mock_pipeline):
        from src.workers.scan_worker import execute_scan_job
        payload = {
            "files": {
                "doc1.txt": base64.b64encode(b"Hello world this is a test").decode(),
                "doc2.txt": base64.b64encode(b"Another document here").decode(),
            },
            "threshold": 0.4,
        }
        result = execute_scan_job(payload)
        assert result["documents_processed"] == 2
        assert result["total_chunks"] > 0
        assert len(result["similarity_matrix"]) == 2
        assert result["similarity_matrix"][0][0] == 1.0  # self-similarity
