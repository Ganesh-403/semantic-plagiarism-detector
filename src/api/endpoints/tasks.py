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

"""
src/api/endpoints/tasks.py
--------------------------
REST endpoints for submitting batch scanning jobs and polling their
status (Issue #3146).

Endpoints:
    POST /api/v1/tasks/batch-scan
        Submit a batch of files (base64-encoded) for asynchronous scanning.
        Returns 202 + job_id immediately.

    GET /api/v1/tasks/{job_id}
        Poll the status of a submitted job.

    GET /api/v1/tasks
        List all jobs (optionally filtered by status).

    GET /api/v1/tasks/dead-letter/list
        List jobs that exhausted all retries and were moved to dead-letter.

    POST /api/v1/tasks/{job_id}/retry
        Re-queue a dead-lettered job for another attempt.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.db import task_db
from src.workers.task_queue import get_default_queue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["Batch Task Queue"])


# ── Pydantic schemas ───────────────────────────────────────────


class FilePayload(BaseModel):
    """A single file encoded as base64."""

    filename: str = Field(..., description="Original filename (e.g. 'assignment1.pdf')")
    content_base64: str = Field(..., description="Base64-encoded file bytes")


class BatchScanRequest(BaseModel):
    """Request body for POST /api/v1/tasks/batch-scan."""

    files: list[FilePayload] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="1–50 files to scan in this batch job.",
    )
    threshold: float = Field(default=0.59, ge=0.0, le=1.0)
    top_k: int = Field(default=3, ge=1, le=10)
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)


class BatchScanResponse(BaseModel):
    """202 Accepted response with the job ID."""

    job_id: str
    status: str = "PENDING"
    message: str = (
        "Batch scan job submitted. Poll GET /api/v1/tasks/{job_id} for status."
    )


class JobStatusResponse(BaseModel):
    """Full job status payload returned by GET /api/v1/tasks/{job_id}."""

    id: str
    status: str
    payload: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]
    total: int


# ── Endpoints ──────────────────────────────────────────────────


@router.post(
    "/batch-scan",
    response_model=BatchScanResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_batch_scan(request: BatchScanRequest) -> BatchScanResponse:
    """Submit a batch of documents for asynchronous plagiarism scanning.

    The request body contains 1–50 files encoded as base64. The API
    returns immediately with a ``job_id``; the actual scanning happens
    in a background worker. Poll ``GET /api/v1/tasks/{job_id}`` to
    check progress.
    """
    files_dict: dict[str, str] = {}
    for f in request.files:
        files_dict[f.filename] = f.content_base64

    payload = {
        "files": files_dict,
        "threshold": request.threshold,
        "top_k": request.top_k,
        "chunk_size": request.chunk_size,
        "chunk_overlap": request.chunk_overlap,
    }

    q = get_default_queue()
    job = q.enqueue(payload)

    return BatchScanResponse(
        job_id=job["id"],
        status=job["status"],
        message="Batch scan job submitted. Poll GET /api/v1/tasks/{job_id} for status.",
    )


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Poll the status of a submitted batch scan job."""
    job = task_db.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return JobStatusResponse(**job)


@router.get(
    "",
    response_model=JobListResponse,
)
async def list_jobs(
    status_filter: Optional[str] = Query(
        default=None,
        alias="status",
        description="Filter by status: PENDING, PROCESSING, COMPLETED, FAILED, DEAD_LETTER",
    ),
    limit: int = Query(default=100, ge=1, le=500),
) -> JobListResponse:
    """List all jobs, optionally filtered by status."""
    if status_filter and status_filter not in task_db.VALID_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{status_filter}'. Valid: {', '.join(task_db.VALID_STATUSES)}",
        )
    jobs = task_db.list_jobs(status=status_filter, limit=limit)
    return JobListResponse(
        jobs=[JobStatusResponse(**j) for j in jobs],
        total=len(jobs),
    )


@router.get(
    "/dead-letter/list",
    response_model=JobListResponse,
)
async def list_dead_letter_jobs(
    limit: int = Query(default=50, ge=1, le=200),
) -> JobListResponse:
    """List all jobs that exhausted their retries and were moved to dead-letter."""
    jobs = task_db.get_dead_letter_jobs(limit=limit)
    return JobListResponse(
        jobs=[JobStatusResponse(**j) for j in jobs],
        total=len(jobs),
    )


@router.post(
    "/{job_id}/retry",
    response_model=BatchScanResponse,
    status_code=status.HTTP_200_OK,
)
async def retry_dead_letter_job(job_id: str) -> BatchScanResponse:
    """Re-queue a dead-lettered job for another attempt.

    Resets retry_count to 0 and status to PENDING.
    """
    job = task_db.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    if job["status"] != "DEAD_LETTER":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job '{job_id}' is not in DEAD_LETTER status (current: {job['status']}).",
        )

    q = get_default_queue()
    new_job = q.enqueue(job["payload"])

    return BatchScanResponse(
        job_id=new_job["id"],
        status=new_job["status"],
        message=f"Re-queued dead-lettered job '{job_id}' as new job '{new_job['id']}'.",
    )
