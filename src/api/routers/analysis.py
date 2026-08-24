"""src/api/routers/analysis.py - Plagiarism analysis and scan job management router."""

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
    AsyncScanJobResponse,
    AsyncScanStatusResponse,
    ErrorResponse,
    SimilarityCheckResponse,
)
from src.core.document_parser import extract_text
from src.core.embedding_model import embed_chunks, get_document_embedding
from src.core.similarity import (
    PLAGIARISM_THRESHOLD,
    chunk_max_similarity,
    find_most_similar_chunks,
)
from src.core.text_chunking import chunk_document
from src.db.corpus_db import get_document_by_hash
from src.security.mime_validator import is_executable_upload
from src.utils.file_streaming import stream_upload_file_to_disk
from src.utils.hash_util import calculate_file_sha256

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Plagiarism Detection"])

total_scans = 0
scan_jobs: Dict[str, Dict[str, Any]] = {}

def check_exact_match_fast_path(
    file_input: Any,
    filename: str,
    word_count: int,
    chunk_count: int,
    threshold: float
) -> Dict[str, Any] | None:
    file_hash = None
    try:
        file_hash = calculate_file_sha256(file_input)
    except Exception as e:
        logger.warning(f"Failed to hash input for fast path: {e}")

    exact_match_filename = None
    if file_hash:
        try:
            exact_match_filename = get_document_by_hash(file_hash)
        except Exception as e:
            logger.warning(f"Failed to query hash for fast path: {e}")

    if exact_match_filename and exact_match_filename != filename:
        return {
            "filename": filename,
            "word_count": word_count,
            "chunk_count": chunk_count,
            "plagiarism_flagged": True,
            "threshold_used": threshold,
            "plagiarism_density": 100,
            "overall_document_similarity": 1.0,
            "max_chunk_similarity": 1.0,
            "matched_documents_count": 1,
            "matched_documents": [
                {
                    "filename": exact_match_filename,
                    "document_similarity_score": 1.0,
                    "max_chunk_similarity_score": 1.0,
                    "severity": "🔴 High",
                    "flagged_chunks": [],
                }
            ],
        }
    return None

def _process_scan_job(
    job_id: str,
    file_input: Any,
    filename: str,
    threshold: float,
    top_k: int,
) -> None:
    if job_id not in scan_jobs:
        if isinstance(file_input, (str, os.PathLike)) and os.path.exists(file_input):
            try:
                os.unlink(file_input)
            except Exception:
                pass
        return

    scan_jobs[job_id]["status"] = "processing"

    try:
        extracted_text = extract_text(file_input, filename)
        if not extracted_text.strip():
            scan_jobs[job_id]["status"] = "failed"
            scan_jobs[job_id][
                "error"
            ] = "Failed to extract readable text from the uploaded file."
            return

        words = extracted_text.split()
        word_count = len(words)

        chunks = chunk_document(extracted_text)
        if not chunks:
            chunks = [extracted_text[:1000]]

        fast_path_result = check_exact_match_fast_path(
            file_input=file_input,
            filename=filename,
            word_count=word_count,
            chunk_count=len(chunks),
            threshold=threshold,
        )
        if fast_path_result:
            scan_jobs[job_id]["status"] = "completed"
            scan_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            scan_jobs[job_id]["result"] = fast_path_result
            return

        uploaded_embeddings = embed_chunks(chunks)
        doc_embedding = get_document_embedding(uploaded_embeddings)
        corpus_docs = get_corpus_documents_with_embeddings()

        matched_documents = []
        max_overall_score = 0.0
        max_chunk_overall_score = 0.0
        uploaded_chunks_flagged = np.zeros(len(chunks), dtype=bool)

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
            uploaded_chunks_flagged |= (chunk_maxes >= threshold)

            combined_score = max(sim_doc, sim_chunk)
            max_overall_score = max(max_overall_score, sim_doc)
            max_chunk_overall_score = max(max_chunk_overall_score, sim_chunk)

            if combined_score >= threshold:
                severity = "🔴 High" if combined_score >= 0.90 else "🟡 Medium"

                similar_chunks = find_most_similar_chunks(
                    chunks_a=chunks,
                    chunks_b=c_chunks,
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
        plagiarism_density = int(round((total_flagged / len(chunks)) * 100)) if len(chunks) > 0 else 0

        scan_jobs[job_id]["status"] = "completed"
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
        scan_jobs[job_id]["status"] = "failed"
        scan_jobs[job_id]["error"] = str(exc)
    finally:
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
        le=10,
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

        extracted_text = extract_text(temp_path, filename)
        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to extract readable text from the uploaded file.",
            )

        words = extracted_text.split()
        word_count = len(words)

        chunks = chunk_document(extracted_text)
        if not chunks:
            chunks = [extracted_text[:1000]]

        fast_path_result = check_exact_match_fast_path(
            file_input=temp_path,
            filename=filename,
            word_count=word_count,
            chunk_count=len(chunks),
            threshold=threshold,
        )
        if fast_path_result:
            return fast_path_result

        uploaded_embeddings = embed_chunks(chunks)
        doc_embedding = get_document_embedding(uploaded_embeddings)
        corpus_docs = get_corpus_documents_with_embeddings()

        matched_documents = []
        max_overall_score = 0.0
        max_chunk_overall_score = 0.0
        uploaded_chunks_flagged = np.zeros(len(chunks), dtype=bool)

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
            uploaded_chunks_flagged |= (chunk_maxes >= threshold)

            combined_score = max(sim_doc, sim_chunk)
            max_overall_score = max(max_overall_score, sim_doc)
            max_chunk_overall_score = max(max_chunk_overall_score, sim_chunk)

            if combined_score >= threshold:
                severity = "🔴 High" if combined_score >= 0.90 else "🟡 Medium"

                similar_chunks = find_most_similar_chunks(
                    chunks_a=chunks,
                    chunks_b=c_chunks,
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
        plagiarism_density = int(round((total_flagged / len(chunks)) * 100)) if len(chunks) > 0 else 0

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
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


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
        le=10,
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

    scan_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
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
    if job_id not in scan_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan job '{job_id}' not found.",
        )

    job = scan_jobs[job_id]
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "filename": job["filename"],
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
        "result": job.get("result"),
        "error": job.get("error"),
    }
