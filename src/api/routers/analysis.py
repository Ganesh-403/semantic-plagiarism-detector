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

"""src/api/routers/analysis.py - Plagiarism analysis and scan job management router."""

import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Security,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity

from src.api.dependencies import (
    get_corpus_documents_with_embeddings,
    get_current_user,
    validate_content_type,
    verify_bearer_token,
)
from src.api.schemas import (
    AsyncScanJobCancelResponse,
    AsyncScanJobResponse,
    AsyncScanStatusResponse,
    ErrorResponse,
    ScanTextRequest,
    SimilarityCheckResponse,
)
from src.core.document_parser import extract_text
from src.core.embedding_model import (
    embed_chunks,
    get_document_embedding,
    release_large_batch_memory,
)
from src.core.metrics import spd_scan_duration_seconds
from src.core.similarity import PLAGIARISM_THRESHOLD, find_most_similar_chunks
from src.core.text_chunking import chunk_document
from src.db.corpus_db import get_document_by_hash
from src.security.mime_validator import is_executable_upload
from src.utils.file_streaming import stream_upload_file_to_disk
from src.utils.hash_util import calculate_file_sha256

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Plagiarism Detection"])

total_scans = 0
scan_jobs: dict[str, dict[str, Any]] = {}
SCAN_JOB_TTL_SECONDS = int(os.getenv("SCAN_JOB_TTL_SECONDS", 7200))  # 2 hours default
MAX_IN_MEMORY_SCAN_JOBS = int(os.getenv("MAX_IN_MEMORY_SCAN_JOBS", 10000))


def cleanup_expired_scan_jobs(
    max_age_seconds: int = SCAN_JOB_TTL_SECONDS,
    max_capacity: int = MAX_IN_MEMORY_SCAN_JOBS,
) -> int:
    """Remove completed or failed scan jobs from the in-memory dictionary.

    Evicts:
    1. Jobs whose completion/failure time (or created time) is older than `max_age_seconds` (default: 2 hours).
    2. Oldest completed/failed jobs if the total in-memory jobs exceed `max_capacity`.

    Args:
        max_age_seconds: Maximum lifetime in seconds for finished scan jobs.
        max_capacity: Maximum number of scan jobs permitted in memory.

    Returns:
        int: Number of evicted jobs.
    """
    now = datetime.now(timezone.utc)
    expired_job_ids = []

    for job_id, job in list(scan_jobs.items()):
        status_val = job.get("status")
        # Only evict finished jobs (completed, failed, or cancelled); active jobs in queued/processing state must remain
        if status_val not in ("completed", "failed", "cancelled"):
            continue

        ts_str = job.get("completed_at") or job.get("created_at")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if (now - ts).total_seconds() >= max_age_seconds:
                    expired_job_ids.append(job_id)
            except (ValueError, TypeError):
                pass

    for job_id in expired_job_ids:
        scan_jobs.pop(job_id, None)

    # Capacity-based eviction for remaining completed/failed/cancelled jobs if limit exceeded
    if len(scan_jobs) > max_capacity:
        finished_jobs = [
            (jid, jdata.get("completed_at") or jdata.get("created_at") or "")
            for jid, jdata in scan_jobs.items()
            if jdata.get("status") in ("completed", "failed", "cancelled")
        ]
        finished_jobs.sort(key=lambda x: x[1])
        excess = len(scan_jobs) - max_capacity
        for jid, _ in finished_jobs[:excess]:
            scan_jobs.pop(jid, None)
            expired_job_ids.append(jid)

    return len(expired_job_ids)


def _process_scan_job(
    job_id: str,
    file_input: Any,
    filename: str,
    threshold: float,
    top_k: int,
) -> None:
    if job_id not in scan_jobs or scan_jobs[job_id].get("status") == "cancelled":
        if isinstance(file_input, (str, os.PathLike)) and os.path.exists(file_input):
            try:
                os.unlink(file_input)
            except Exception:
                pass
        return

    def _is_cancelled() -> bool:
        return scan_jobs.get(job_id, {}).get("status") == "cancelled"

    scan_jobs[job_id]["status"] = "processing"
    chunks: list = []

    try:
        with spd_scan_duration_seconds.labels(stage="parsing").time():
            extracted_text = extract_text(file_input, filename)
        scan_jobs[job_id]["progress_percent"] = 20
        scan_jobs[job_id]["stage"] = "text extraction"

        if not extracted_text.strip():
            if not _is_cancelled():
                scan_jobs[job_id]["status"] = "failed"
                scan_jobs[job_id]["completed_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                scan_jobs[job_id][
                    "error"
                ] = "Failed to extract readable text from the uploaded file."
                cleanup_expired_scan_jobs()
            return

        if _is_cancelled():
            return

        words = extracted_text.split()
        word_count = len(words)

        with spd_scan_duration_seconds.labels(stage="chunking").time():
            chunks = chunk_document(extracted_text)
            if not chunks:
                chunks = [extracted_text[:1000]]
        scan_jobs[job_id]["progress_percent"] = 40
        scan_jobs[job_id]["stage"] = "chunking"

        if isinstance(file_input, (str, os.PathLike)) and os.path.exists(file_input):
            file_hash = calculate_file_sha256(str(file_input))
            existing_doc_filename = get_document_by_hash(file_hash)
            if existing_doc_filename:
                scan_jobs[job_id]["status"] = "completed"
                scan_jobs[job_id]["progress_percent"] = 100
                scan_jobs[job_id]["stage"] = "done"
                scan_jobs[job_id]["completed_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                scan_jobs[job_id]["result"] = {
                    "filename": filename,
                    "word_count": word_count,
                    "chunk_count": len(chunks),
                    "plagiarism_flagged": True,
                    "threshold_used": threshold,
                    "plagiarism_density": 100,
                    "overall_document_similarity": 1.0,
                    "max_chunk_similarity": 1.0,
                    "matched_documents_count": 1,
                    "matched_documents": [
                        {
                            "filename": existing_doc_filename,
                            "document_similarity_score": 1.0,
                            "max_chunk_similarity_score": 1.0,
                            "severity": "🔴 High",
                            "flagged_chunks": [],
                        }
                    ],
                }
                return

        with spd_scan_duration_seconds.labels(stage="embedding").time():
            uploaded_embeddings = embed_chunks(chunks)
            doc_embedding = get_document_embedding(uploaded_embeddings)
            corpus_docs = get_corpus_documents_with_embeddings()
        scan_jobs[job_id]["progress_percent"] = 70
        scan_jobs[job_id]["stage"] = "embedding"

        matched_documents = []
        max_overall_score = 0.0
        max_chunk_overall_score = 0.0
        uploaded_chunks_flagged = np.zeros(len(chunks), dtype=bool)

        with spd_scan_duration_seconds.labels(stage="matrix comparison").time():
            for corpus_filename, corpus_data in corpus_docs.items():
                if corpus_filename == filename:
                    continue

                c_embeddings = corpus_data["embeddings"]
                c_chunks = corpus_data["chunks"]

                if c_embeddings.size == 0:
                    continue

                c_doc_embedding = get_document_embedding(c_embeddings)
                sim_doc = float(
                    np.clip(
                        cosine_similarity(
                            doc_embedding.reshape(1, -1), c_doc_embedding.reshape(1, -1)
                        )[0, 0],
                        0.0,
                        1.0,
                    )
                )
                sim_matrix = cosine_similarity(uploaded_embeddings, c_embeddings)
                sim_chunk = float(np.max(sim_matrix))

                chunk_maxes = np.max(sim_matrix, axis=1)
                uploaded_chunks_flagged |= chunk_maxes >= threshold

                combined_score = max(sim_doc, sim_chunk)
                max_overall_score = max(max_overall_score, sim_doc)
                max_chunk_overall_score = max(max_chunk_overall_score, sim_chunk)

                if combined_score >= threshold:
                    severity = "🔴 High" if combined_score >= 0.90 else "🟡 Medium"

                    similar_chunks = find_most_similar_chunks(
                        chunks_a=[chunk.text for chunk in chunks],
                        chunks_b=[chunk.text for chunk in c_chunks],
                        emb_a=uploaded_embeddings,
                        emb_b=c_embeddings,
                        top_k=top_k,
                        threshold=threshold,
                    )

                    flagged_chunks = [
                        {
                            "uploaded_chunk": pair[0],
                            "matched_chunk": pair[1],
                            "similarity_score": round(float(pair[2]), 4),
                        }
                        for pair in similar_chunks
                    ]

                    matched_documents.append(
                        {
                            "filename": corpus_filename,
                            "document_similarity_score": round(sim_doc, 4),
                            "max_chunk_similarity_score": round(sim_chunk, 4),
                            "severity": severity,
                            "flagged_chunks": flagged_chunks,
                        }
                    )

            matched_documents.sort(
                key=lambda x: x["max_chunk_similarity_score"], reverse=True
            )
            is_flagged = (
                len(matched_documents) > 0 or max_chunk_overall_score >= threshold
            )

            total_flagged = int(np.sum(uploaded_chunks_flagged))
            plagiarism_density = (
                int(round((total_flagged / len(chunks)) * 100))
                if len(chunks) > 0
                else 0
            )

        scan_jobs[job_id]["progress_percent"] = 90
        scan_jobs[job_id]["stage"] = "comparison"

        scan_jobs[job_id]["status"] = "completed"
        scan_jobs[job_id]["progress_percent"] = 100
        scan_jobs[job_id]["stage"] = "done"
        scan_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        scan_jobs[job_id]["result"] = {
            "filename": filename,
            "word_count": word_count,
            "chunk_count": len(chunks),
            "plagiarism_flagged": is_flagged,
            "threshold_used": threshold,
            "plagiarism_density": plagiarism_density,
            "overall_document_similarity": round(max_overall_score, 4),
            "max_chunk_similarity": round(max_chunk_overall_score, 4),
            "matched_documents_count": len(matched_documents),
            "matched_documents": matched_documents,
        }
    except Exception as exc:
        if _is_cancelled():
            return
        scan_jobs[job_id]["status"] = "failed"
        scan_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        scan_jobs[job_id]["error"] = str(exc)
    finally:
        cleanup_expired_scan_jobs()
        # Issue #3479: free NumPy/PyTorch heap memory after large batch scans.
        release_large_batch_memory(len(chunks))
        if isinstance(file_input, (str, os.PathLike)) and os.path.exists(file_input):
            try:
                os.unlink(file_input)
            except Exception:
                pass


@router.get(
    "/api/v1/incidents",
    summary="Get recorded plagiarism incidents",
    status_code=status.HTTP_200_OK,
)
def get_incidents(
    limit: int = Query(
        default=50, ge=1, le=500, description="Max number of incidents to return"
    ),
    offset: int = Query(default=0, ge=0, description="Number of incidents to skip"),
    _token: str = Depends(verify_bearer_token),
):
    """Retrieve recorded plagiarism incidents from the database."""
    from src.db.incidents import get_all_incidents

    try:
        incidents = get_all_incidents(limit=limit, offset=offset)
        return {
            "incidents": [
                {
                    "incident_id": inc.incident_id,
                    "document_a": inc.document_a,
                    "document_b": inc.document_b,
                    "similarity_score": inc.similarity_score,
                    "severity_rank": inc.severity_rank,
                    "review_status": inc.review_status,
                    "date_flagged": inc.date_flagged,
                    "threshold_at_time_of_flag": inc.threshold_at_time_of_flag,
                }
                for inc in incidents
            ],
            "limit": limit,
            "offset": offset,
            "count": len(incidents),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch incidents: {str(exc)}",
        )


@router.post(
    "/api/v1/scan",
    response_model=SimilarityCheckResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Unprocessable Entity"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def scan_document(
    file: UploadFile = File(
        ..., description="Document file to scan (.pdf, .docx, .txt)"
    ),
    threshold: float = Query(
        default=PLAGIARISM_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for flagging plagiarism (default: 0.59)",
    ),
    top_k: int = Query(
        default=3,
        ge=1,
        le=100,
        description="Number of top matching paragraph pairs to include per matched document",
    ),
    reprocess: bool = Query(
        default=False,
        description="Bypass duplicate detection and process the file anyway",
    ),
    _user: dict = Security(get_current_user, scopes=["write"]),
    _content_type: None = Depends(validate_content_type),
):
    """Scan an uploaded document against the indexed corpus database for plagiarism."""
    global total_scans
    total_scans += 1
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must be provided.",
        )

    filename = file.filename

    file_head = await file.read(64)
    await file.seek(0)
    if is_executable_upload(file_head, filename):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported Media Type: executable and script files are not allowed.",
        )

    temp_path = await stream_upload_file_to_disk(file)

    try:
        if not reprocess:
            file_hash = calculate_file_sha256(temp_path)
            existing_doc = get_document_by_hash(file_hash)
            if existing_doc:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
                return JSONResponse(
                    status_code=status.HTTP_409_CONFLICT,
                    content={
                        "duplicate": True,
                        "message": "This file has already been uploaded.",
                    },
                )

        with spd_scan_duration_seconds.labels(stage="parsing").time():
            extracted_text = extract_text(temp_path, filename)
        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to extract readable text from the uploaded file.",
            )

        return _perform_text_scan(
            extracted_text=extracted_text,
            filename=filename,
            threshold=threshold,
            top_k=top_k,
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


def _perform_text_scan(
    extracted_text: str,
    filename: str,
    threshold: float,
    top_k: int = 3,
) -> dict:
    """Core embedding and vector comparison pipeline shared by file scan and raw text scan endpoints."""
    if not extracted_text or not extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to extract readable text from the submission.",
        )

    words = extracted_text.split()
    word_count = len(words)

    with spd_scan_duration_seconds.labels(stage="chunking").time():
        chunks = chunk_document(extracted_text)
        if not chunks:
            chunks = [extracted_text[:1000]]

    with spd_scan_duration_seconds.labels(stage="embedding").time():
        uploaded_embeddings = embed_chunks(chunks)
        doc_embedding = get_document_embedding(uploaded_embeddings)
        corpus_docs = get_corpus_documents_with_embeddings()

    matched_documents = []
    max_overall_score = 0.0
    max_chunk_overall_score = 0.0
    uploaded_chunks_flagged = np.zeros(len(chunks), dtype=bool)

    with spd_scan_duration_seconds.labels(stage="matrix comparison").time():
        for corpus_filename, corpus_data in corpus_docs.items():
            if corpus_filename == filename:
                continue

            c_embeddings = corpus_data["embeddings"]
            c_chunks = corpus_data["chunks"]

            if c_embeddings.size == 0:
                continue

            c_doc_embedding = get_document_embedding(c_embeddings)
            sim_doc = float(
                np.clip(
                    cosine_similarity(
                        doc_embedding.reshape(1, -1), c_doc_embedding.reshape(1, -1)
                    )[0, 0],
                    0.0,
                    1.0,
                )
            )

            sim_matrix = cosine_similarity(uploaded_embeddings, c_embeddings)
            sim_chunk = float(np.max(sim_matrix))

            chunk_maxes = np.max(sim_matrix, axis=1)
            uploaded_chunks_flagged |= chunk_maxes >= threshold

            combined_score = max(sim_doc, sim_chunk)
            max_overall_score = max(max_overall_score, sim_doc)
            max_chunk_overall_score = max(max_chunk_overall_score, sim_chunk)

            if combined_score >= threshold:
                severity = "🔴 High" if combined_score >= 0.90 else "🟡 Medium"

                similar_chunks = find_most_similar_chunks(
                    chunks_a=[chunk.text for chunk in chunks],
                    chunks_b=[chunk.text for chunk in c_chunks],
                    emb_a=uploaded_embeddings,
                    emb_b=c_embeddings,
                    top_k=top_k,
                    threshold=threshold,
                )

                flagged_chunks = [
                    {
                        "uploaded_chunk": pair[0],
                        "matched_chunk": pair[1],
                        "similarity_score": round(float(pair[2]), 4),
                    }
                    for pair in similar_chunks
                ]

                matched_documents.append(
                    {
                        "filename": corpus_filename,
                        "document_similarity_score": round(sim_doc, 4),
                        "max_chunk_similarity_score": round(sim_chunk, 4),
                        "severity": severity,
                        "flagged_chunks": flagged_chunks,
                    }
                )

        matched_documents.sort(
            key=lambda x: x["max_chunk_similarity_score"], reverse=True
        )
        is_flagged = len(matched_documents) > 0 or max_chunk_overall_score >= threshold

        total_flagged = int(np.sum(uploaded_chunks_flagged))
        plagiarism_density = (
            int(round((total_flagged / len(chunks)) * 100)) if len(chunks) > 0 else 0
        )

    return {
        "filename": filename,
        "word_count": word_count,
        "chunk_count": len(chunks),
        "plagiarism_flagged": is_flagged,
        "threshold_used": threshold,
        "plagiarism_density": plagiarism_density,
        "overall_document_similarity": round(max_overall_score, 4),
        "max_chunk_similarity": round(max_chunk_overall_score, 4),
        "matched_documents_count": len(matched_documents),
        "matched_documents": matched_documents,
    }


@router.post(
    "/api/v1/scan/text",
    response_model=SimilarityCheckResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        409: {"description": "Conflict - Duplicate content"},
        422: {"model": ErrorResponse, "description": "Unprocessable Entity"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def scan_text(
    request_data: ScanTextRequest,
    _user: dict = Security(get_current_user, scopes=["write"]),
):
    """Scan raw text content directly for plagiarism without requiring a multipart file upload."""
    global total_scans
    total_scans += 1

    if not request_data.text or not request_data.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text content cannot be empty.",
        )

    filename = request_data.filename or "submission.txt"

    if not request_data.reprocess:
        text_hash = hashlib.sha256(request_data.text.encode("utf-8")).hexdigest()
        existing_doc = get_document_by_hash(text_hash)
        if existing_doc:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "duplicate": True,
                    "message": "This file has already been uploaded.",
                },
            )

    return _perform_text_scan(
        extracted_text=request_data.text,
        filename=filename,
        threshold=request_data.threshold,
        top_k=request_data.top_k,
    )


@router.post(
    "/api/v1/scan/async",
    response_model=AsyncScanJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        415: {"model": ErrorResponse, "description": "Unsupported Media Type"},
        422: {"model": ErrorResponse, "description": "Unprocessable Entity"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def scan_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(
        ..., description="Document file to scan (.pdf, .docx, .txt)"
    ),
    threshold: float = Query(
        default=PLAGIARISM_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for flagging plagiarism (default: 0.59)",
    ),
    top_k: int = Query(
        default=3,
        ge=1,
        le=100,
        description="Number of top matching paragraph pairs to include per matched document",
    ),
    reprocess: bool = Query(
        default=False,
        description="Bypass duplicate detection and process the file anyway",
    ),
    _user: dict = Security(get_current_user, scopes=["write"]),
    _content_type: None = Depends(validate_content_type),
):
    """Enqueue a document scanning job for asynchronous background processing."""
    global total_scans
    total_scans += 1
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must be provided.",
        )

    filename = file.filename
    temp_path = await stream_upload_file_to_disk(file)

    if not reprocess:
        file_hash = calculate_file_sha256(temp_path)
        existing_doc = get_document_by_hash(file_hash)
        if existing_doc:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "duplicate": True,
                    "message": "This file has already been uploaded.",
                },
            )

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    status_url = f"/api/v1/scan/status/{job_id}"

    cleanup_expired_scan_jobs()

    scan_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress_percent": 0,
        "stage": "",
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
    }

    background_tasks.add_task(
        _process_scan_job,
        job_id,
        temp_path,
        filename,
        threshold,
        top_k,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": status_url,
        "message": "Scan job successfully queued for asynchronous processing.",
    }


@router.get(
    "/api/v1/scan/status/{job_id}",
    response_model=AsyncScanStatusResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Not Found"},
    },
)
def get_async_scan_status(
    job_id: str,
    _user: dict = Security(get_current_user, scopes=["read"]),
):
    """Retrieve the status and results of an asynchronous scan job."""
    cleanup_expired_scan_jobs()
    if job_id not in scan_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan job '{job_id}' not found.",
        )

    job = scan_jobs[job_id]
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress_percent": job.get("progress_percent", 0),
        "stage": job.get("stage", ""),
        "filename": job["filename"],
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
        "result": job.get("result"),
        "error": job.get("error"),
    }


@router.delete(
    "/api/v1/scan/jobs/{job_id}",
    response_model=AsyncScanJobCancelResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Not Found"},
    },
)
def cancel_async_scan_job(
    job_id: str,
    _user: dict = Security(get_current_user, scopes=["write"]),
):
    """Cancel an active or queued asynchronous document scan job."""
    cleanup_expired_scan_jobs()
    if job_id not in scan_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan job '{job_id}' not found.",
        )

    scan_jobs[job_id]["status"] = "cancelled"
    scan_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
    scan_jobs[job_id]["error"] = "Job was cancelled by user request."

    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": f"Scan job '{job_id}' successfully cancelled.",
    }
