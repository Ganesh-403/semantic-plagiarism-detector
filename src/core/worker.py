"""
src/core/worker.py
------------------
Celery background job queue for async document upload processing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import time

from src.core.config import PLAGIARISM_THRESHOLD
from src.celery_app.tasks import run_pipeline_job

logger = logging.getLogger(__name__)

class DummyJob:
    def __init__(self, job_id):
        self.id = job_id

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    from src.celery_app.celery_config import celery_app
def get_job_status(job_id: str) -> Optional[dict[str, Any]]:
    """Return the current status and (if finished) the result summary for a job.

    Returns ``None`` if the job ID is unknown or expired.
    """
    try:
        task = celery_app.AsyncResult(job_id)
    except Exception:
        return None
    
    status = {
        "job_id": task.id,
        "status": task.state,
    }
    
    if task.state == 'FAILURE':
        status["error"] = str(task.info)
    elif task.state == 'SUCCESS':
        status["result"] = task.info
    elif task.state == 'PROCESSING':
        status["progress_info"] = task.info

    return status


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
) -> DummyJob:
    
    config = {
        "threshold": threshold,
        "ignore_phrases": ignore_phrases,
        "ocr_language": ocr_language,
        "ocr_dpi": ocr_dpi,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    
    task = run_pipeline_job.delay(file_bytes_dict, config)
    logger.info("Enqueued celery upload job %s (%d files)", task.id, len(file_bytes_dict))
    
    return DummyJob(task.id)
