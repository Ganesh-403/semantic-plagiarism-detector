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

def enqueue_upload_job(
    file_bytes_dict: Dict[str, bytes],
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
