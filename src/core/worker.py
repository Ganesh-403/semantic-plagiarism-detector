"""
src/core/worker.py
------------------
Redis RQ background job queue for async document upload processing.

Queue architecture
------------------
- **Upload jobs** are enqueued by the Streamlit UI / FastAPI endpoint.
- An **RQ worker** (started separately) picks up jobs and runs the upload
  pipeline in the background.
- **Job status** is tracked in Redis with a 24-hour TTL so the frontend
  can poll for completion.

Usage
-----
Start the worker (in a separate terminal or Docker container)::

    rq worker upload --url redis://localhost:6379/0

Enqueue a job from the application::

    from src.core.worker import enqueue_upload_job
    job = enqueue_upload_job(file_bytes_dict, threshold=0.6)

Poll for status::

    from src.core.worker import get_job_status
    status = get_job_status(job.id)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from redis import Redis
from rq import Queue
from rq.job import Job

from src.core.config import PLAGIARISM_THRESHOLD
from src.core.processing import run_full_pipeline
from src.db.incidents import sync_flagged_incidents

logger = logging.getLogger(__name__)

# ── Redis connection ───────────────────────────────────────────────────────────

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_JOB_TTL = 86_400  # 24 hours — results expire after this


def _get_redis() -> Redis:
    return Redis.from_url(_REDIS_URL, decode_responses=False)


def _get_queue() -> Queue:
    return Queue("upload", connection=_get_redis(), default_timeout=3600)


# ── Job status helpers ─────────────────────────────────────────────────────────


def get_job_status(job_id: str) -> Optional[dict[str, Any]]:
    """Return the current status and (if finished) the result summary for a job.

    Returns ``None`` if the job ID is unknown or expired.
    """
    try:
        job = Job.fetch(job_id, connection=_get_redis())
    except Exception:
        return None

    status: dict[str, Any] = {
        "job_id": job.id,
        "status": job.get_status(),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
    }

    if job.is_failed:
        status["error"] = str(job.exc_info)
    elif job.is_finished:
        result: dict[str, Any] = job.return_value or {}
        status["result"] = {
            "document_count": result.get("document_count", 0),
            "flags_count": result.get("flags_count", 0),
        }

    return status


# ── Background task ────────────────────────────────────────────────────────────


def _run_upload_job(
    file_bytes_dict: dict[str, bytes],
    threshold: float = PLAGIARISM_THRESHOLD,
    ignore_phrases: Optional[str] = None,
    ocr_language: str = "eng",
    ocr_dpi: int = 300,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict[str, Any]:
    """Background task executed by the RQ worker.

    Runs the full pipeline then syncs incidents to the database.  Returns a
    JSON-serialisable summary dict (pickle-free so the frontend can read it).
    """
    logger.info(
        "Background job started: %d files, threshold=%.2f",
        len(file_bytes_dict),
        threshold,
    )

    start = time.perf_counter()

    try:
        pipeline_result = run_full_pipeline(
            file_bytes_dict,
            ocr_language=ocr_language,
            ocr_dpi=ocr_dpi,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            threshold=threshold,
            ignore_phrases=ignore_phrases,
        )

        incidents = sync_flagged_incidents(pipeline_result.flags)

        elapsed = time.perf_counter() - start
        logger.info(
            "Background job finished: %d documents, %d flags, %.2fs elapsed",
            len(pipeline_result.raw_texts),
            len(pipeline_result.flags),
            elapsed,
        )

        return {
            "document_count": len(pipeline_result.raw_texts),
            "flags_count": len(pipeline_result.flags),
            "incidents_count": len(incidents),
            "elapsed_seconds": round(elapsed, 2),
        }

    except Exception:
        logger.exception("Background job failed")
        raise


# ── Public API ─────────────────────────────────────────────────────────────────


def enqueue_upload_job(
    file_bytes_dict: dict[str, bytes],
    *,
    threshold: float = PLAGIARISM_THRESHOLD,
    ignore_phrases: Optional[str] = None,
    ocr_language: str = "eng",
    ocr_dpi: int = 300,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Job:
    """Enqueue an upload processing job to the Redis RQ queue.

    Args:
        file_bytes_dict: Mapping of filename -> raw bytes for each uploaded file.
        threshold:       Similarity threshold for plagiarism flagging.
        ignore_phrases:  Optional comma-separated phrases to ignore.
        ocr_language:    Tesseract OCR language code.
        ocr_dpi:         DPI for OCR processing.
        chunk_size:      Chunk size in words.
        chunk_overlap:   Chunk overlap in words.

    Returns:
        The ``rq.job.Job`` instance.  Use ``job.id`` to poll status via
        :func:`get_job_status`.
    """
    queue = _get_queue()
    job = queue.enqueue(
        _run_upload_job,
        file_bytes_dict,
        threshold=threshold,
        ignore_phrases=ignore_phrases,
        ocr_language=ocr_language,
        ocr_dpi=ocr_dpi,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        job_timeout=3600,
        result_ttl=_JOB_TTL,
        failure_ttl=_JOB_TTL,
    )
    logger.info("Enqueued upload job %s (%d files)", job.id, len(file_bytes_dict))
    return job
