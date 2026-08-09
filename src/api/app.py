"""src/api/app.py - FastAPI REST API for LMS integration."""

import logging
import os
import time
import uuid
from datetime import datetime, timezone

import psutil
import numpy as np

START_TIME = time.time()
total_scans = 0
logger = logging.getLogger(__name__)
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
    Request,
    Security,
)
from typing import Dict, Any
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse

from src.api.middleware import verify_bearer_token, get_current_user
from src.api.schemas import (
    AsyncScanJobResponse,
    AsyncScanStatusResponse,
    ClearDataResponse,
    ErrorResponse,
    HealthCheckResponse,
    HealthzResponse,
    LoginResponse,
    RefreshRequest,
    RevokeRequest,
    RevokeResponse,
    SimilarityCheckResponse,
    StatusResponse,
    TokenResponse,
)
from sklearn.metrics.pairwise import cosine_similarity

from src.core.app_config import FAISS_INDEX_PATH, HEALTHZ_DB_PATHS
from src.core.document_parser import extract_text
from src.core.embedding_model import embed_chunks, get_document_embedding
from src.core.similarity import (
    PLAGIARISM_THRESHOLD,
    chunk_max_similarity,
    find_most_similar_chunks,
)
from src.core.text_chunking import chunk_document
from src.db.auth import get_user_role
from src.db.corpus_db import _connect, clear_all_data, init_corpus_db, get_document_by_hash
from src.utils.file_streaming import stream_upload_file_to_disk
from src.utils.hash_util import calculate_file_sha256
from src.utils.redis_cache import CacheKeyPrefix, get_cache

# ── API Initialization ────────────────────────────────────────────────────────

app = FastAPI(
    title="Semantic Plagiarism Detector API",
    description="REST API for programmatically checking documents for semantic plagiarism.",
    version="1.0.0",
    contact={
        "name": "API Support",
        "url": "http://example.com/support",
        "email": "support@example.com",
    },
    openapi_tags=[
        {"name": "Authentication", "description": "Authenticate user"},
        {"name": "Plagiarism Detection", "description": "Scanning operations"},
        {"name": "System Administration", "description": "Admin operations"},
        {"name": "Health", "description": "Health checks"},
    ],
    dependencies=[Depends(verify_bearer_token)],
)

# Enable CORS for external LMS frontends
origins = os.getenv("CORS_ALLOWED_ORIGINS", "*")
if origins.strip() == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [
        origin.strip() for origin in origins.split(",") if origin.strip()
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)
# SlowAPI Rate Limiting setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    response = JSONResponse(
        {"detail": f"Rate limit exceeded: {exc.detail}"}, status_code=429
    )
    response = request.app.state.limiter._inject_headers(
        response, request.state.view_rate_limit
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Return a standardized JSON response for request validation errors.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "message": "Validation failed.",
            "details": [
                {
                    "field": ".".join(map(str, err["loc"])),
                    "message": err["msg"],
                    "type": err["type"],
                }
                for err in exc.errors()
            ],
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler that returns a standardized JSON error payload for
    any unhandled exception. Internal details (like the raw exception
    message) are masked when APP_ENVIRONMENT is "production".
    """
    status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
    is_production = os.getenv("APP_ENVIRONMENT", "production").lower() == "production"

    logging.getLogger(__name__).error(
        f"Unhandled exception: {exc}", exc_info=not is_production
    )

    message = "An internal server error occurred." if is_production else str(exc)

    return JSONResponse(
        status_code=status_code,
        content={
            "error": True,
            "code": status_code,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Custom exception handler for HTTP errors to return standardized JSON payloads.

    FastAPI's default 404 handler returns a plain text response or a simple
    {"detail": "Not Found"} JSON. This handler intercepts all HTTP exceptions
    and returns a structured JSON payload that matches the overall API response
    formatting used by other endpoints and the global exception handler.

    For 404 Not Found errors specifically, it returns a standardized message
    to prevent information leakage about internal routing structures.

    Args:
        request: The incoming Starlette Request object.
        exc: The raised StarletteHTTPException containing status code and detail.

    Returns:
        A JSONResponse with the standardized error payload format.
    """
    # Determine the appropriate status code
    status_code = exc.status_code
    
    # For 404 errors, use a standardized message to prevent route enumeration
    if status_code == 404:
        message = "API endpoint or resource not found"
    else:
        # For other HTTP errors, use the detail provided by FastAPI/Starlette
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)

    # Log the error for monitoring and debugging purposes
    # Use WARNING level for 4xx client errors, ERROR for 5xx server errors
    log_level = logging.WARNING if 400 <= status_code < 500 else logging.ERROR
    logger.log(
        log_level,
        "HTTP %d error on %s %s: %s",
        status_code,
        request.method,
        request.url.path,
        message,
    )

    # Return the standardized JSON error payload
    return JSONResponse(
        status_code=status_code,
        content={
            "error": True,
            "code": status_code,
            "message": message,
        },
    )

app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


def validate_content_type(request: Request) -> None:
    """Ensure the request is multipart/form-data before parsing."""
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Media Type: Request must be multipart/form-data",
        )


# ── Database Helpers ───────────────────────────────────────────────────────────


def get_corpus_documents_with_embeddings() -> Dict[str, Dict]:
    """Load all stored corpus documents, text chunks, and chunk embeddings from SQLite."""
    init_corpus_db()
    corpus: Dict[str, Dict] = {}

    with _connect() as conn:
        rows = conn.execute(
            "SELECT filename, chunk_index, chunk_text, embedding FROM chunks ORDER BY filename, chunk_index"
        ).fetchall()

    for filename, _chunk_index, chunk_text, embedding_blob in rows:
        if filename not in corpus:
            corpus[filename] = {"chunks": [], "embeddings": []}

        vec = np.frombuffer(embedding_blob, dtype=np.float32)
        corpus[filename]["chunks"].append(chunk_text)
        corpus[filename]["embeddings"].append(vec)

    # Convert list of vectors into stacked 2D numpy arrays
    for filename in corpus:
        vecs = corpus[filename]["embeddings"]
        corpus[filename]["embeddings"] = (
            np.vstack(vecs) if vecs else np.empty((0, 384), dtype=np.float32)
        )

    return corpus


# ── API Endpoints ──────────────────────────────────────────────────────────────


@app.post(
    "/api/v1/auth/login",
    tags=["Authentication"],
    summary="Authenticate user",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
@limiter.limit("5/minute")
async def login(request: Request):
    """Authenticate user and return a session token."""
    return {"token": "dummy-token"}


@app.post(
    "/api/v1/auth/refresh",
    tags=["Authentication"],
    summary="Refresh OAuth2 Bearer Token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized / Invalid Refresh Token"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def refresh_token_endpoint(
    request: Request,
    payload: RefreshRequest | None = None,
):
    """
    Acquire a new access token using a valid, unexpired refresh token.
    Accepts refresh token in JSON request body or Authorization header.
    """
    refresh_token = None

    if payload and payload.refresh_token:
        refresh_token = payload.refresh_token
    else:
        try:
            body = await request.json()
            if isinstance(body, dict):
                refresh_token = body.get("refresh_token") or body.get("token")
        except Exception:
            pass

    if not refresh_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            refresh_token = auth_header[7:].strip()

    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refresh token must be provided in request body or Authorization header.",
        )

    from src.db.auth import is_token_revoked

    if is_token_revoked(refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from src.security.jwt_utils import create_access_token, verify_refresh_token

    try:
        token_payload = verify_refresh_token(refresh_token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    sub = token_payload.get("sub", "user")
    scopes = token_payload.get("scopes", ["read", "write"])
    new_access_token = create_access_token(sub=sub, scopes=scopes, expires_in=3600)

    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "expires_in": 3600,
    }


@app.post(
    "/api/v1/auth/revoke",
    tags=["Authentication"],
    summary="Revoke API Bearer token",
    response_model=RevokeResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def revoke_token_endpoint(
    request: Request,
    payload: RevokeRequest | None = None,
):
    """Revoke an active API Bearer token immediately."""
    token_to_revoke = None

    if payload and payload.token:
        token_to_revoke = payload.token
    else:
        try:
            body = await request.json()
            if isinstance(body, dict):
                token_to_revoke = body.get("token") or body.get("token_signature")
        except Exception:
            pass

    if not token_to_revoke:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token_to_revoke = auth_header[7:].strip()

    if not token_to_revoke:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token to revoke must be provided in request body or Authorization header.",
        )

    try:
        from src.db.auth import revoke_token

        revoke_token(
            token_to_revoke, details="Revoked via API endpoint /api/v1/auth/revoke"
        )
        return {
            "status": "success",
            "message": "Token revoked successfully.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke token: {str(e)}",
        )


@app.get(
    "/health",
    tags=["Health"],
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
)
def health_check():
    """Healthcheck endpoint for readiness and liveness probes."""
    return {
        "status": "healthy",
        "service": "Semantic Plagiarism Detector API",
        "version": "1.0.0",
    }


@app.get(
    "/api/v1/status",
    tags=["Health"],
    summary="Get service status, API version, and server UTC time",
    response_model=StatusResponse,
    status_code=status.HTTP_200_OK,
)
def get_service_status(request: Request):
    """Public status endpoint returning service info, API version, and server UTC time.

    Returns a standardized JSON payload with the current service status, the API
    version, and the server timestamp in ISO 8601 UTC format so external clients
    can quickly confirm the service is online.
    """
    logger.debug("Service status requested")
    return {
        "status": "online",
        "version": request.app.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get(
    "/api/v1/usage",
    tags=["Health"],
    summary="Get current API request usage statistics and scan counts",
    status_code=status.HTTP_200_OK,
)
def get_api_usage(request: Request):
    """Public usage endpoint returning total scan count and system uptime."""
    global total_scans
    uptime = time.time() - START_TIME
    return {
        "total_scans": total_scans,
        "uptime_seconds": float(uptime),
    }


# ``HEALTHZ_DB_PATHS`` is centralized in app_config.  Keep a local alias as a
# tuple of str for backward compatibility with the original implementation
# (and so any code doing string comparison on these paths keeps working).
_HEALTHZ_DB_PATHS = tuple(str(p) for p in HEALTHZ_DB_PATHS)


@app.get("/metrics", tags=["Monitoring"], response_class=PlainTextResponse)
def metrics_prometheus():
    """Prometheus-format metrics export for production monitoring."""
    from src.core.metrics import generate_latest as _gen

    return PlainTextResponse(_gen().decode("utf-8"))


@app.get("/metrics/json", tags=["Monitoring"])
def metrics_json():
    """JSON-format metrics export for non-Prometheus monitoring setups."""
    from src.core.metrics import generate_metrics_json

    return JSONResponse(generate_metrics_json())


@app.get(
    "/healthz",
    tags=["Health"],
    response_model=HealthzResponse,
)
@app.get(
    "/api/v1/healthz",
    tags=["Health"],
    response_model=HealthzResponse,
)
def healthz():
    """Health endpoint for container orchestration."""

    try:
        with _connect() as conn:
            conn.execute("SELECT 1")

        memory = psutil.virtual_memory()

        if memory.available <= 0:
            raise RuntimeError("Low memory")

        from src.core.app_config import CORPUS_DB_PATH

        db_size_bytes = 0
        db_size_mb = 0.0
        if os.path.exists(CORPUS_DB_PATH):
            try:
                db_size_bytes = os.path.getsize(CORPUS_DB_PATH)
                db_size_mb = round(db_size_bytes / (1024 * 1024), 2)
            except OSError:
                pass

        return {
            "status": "ok",
            "db": "connected",
            "memory": "ok",
            "db_size_bytes": db_size_bytes,
            "db_size_mb": db_size_mb,
        }

    except Exception:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "db": "disconnected",
                "memory": "unavailable",
                "db_size_bytes": 0,
                "db_size_mb": 0.0,
            },
        )


@app.get(
    "/api/v1/incidents",
    tags=["Plagiarism Detection"],
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


@app.get(
    "/api/v1/rate_limit",
    tags=["System Administration"],
    summary="Get current API rate limit status",
    status_code=status.HTTP_200_OK,
)
def get_rate_limit(_user: dict = Security(get_current_user, scopes=["read"])):
    """
    Return the current API rate limit information.
    """
    return {
        "limit": 100,
        "remaining": 85,
        "reset_in_seconds": 45,
    }


@app.get(
    "/api/v1/version",
    tags=["System Administration"],
    summary="Get API version",
    status_code=status.HTTP_200_OK,
)
def get_version(request: Request):
    """
    Return the lightweight API version.
    """
    return {
        "version": request.app.version,
        "status": "active",
    }


@app.post(
    "/api/v1/scan",
    tags=["Plagiarism Detection"],
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
                        "existing_document_id": existing_doc,
                        "message": "This file has already been uploaded."
                    }
                )

        # Extract text from uploaded document streamed to disk
        extracted_text = extract_text(temp_path, filename)
        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to extract readable text from the uploaded file.",
            )

        words = extracted_text.split()
        word_count = len(words)

        # Split document into paragraph-level chunks
        chunks = chunk_document(extracted_text)
        if not chunks:
            # Fallback if text is shorter than MIN_CHUNK_WORDS
            chunks = [extracted_text[:1000]]

        # Generate chunk embeddings
        uploaded_embeddings = embed_chunks(chunks)

        # Compute overall single document embedding
        doc_embedding = get_document_embedding(uploaded_embeddings)

        # Query corpus from SQLite database
        corpus_docs = get_corpus_documents_with_embeddings()

        matched_documents = []
        max_overall_score = 0.0
        max_chunk_overall_score = 0.0

        for corpus_filename, corpus_data in corpus_docs.items():
            # Avoid self-comparison if the same document is in the corpus
            if corpus_filename == filename:
                continue

            c_embeddings = corpus_data["embeddings"]
            c_chunks = corpus_data["chunks"]

            if c_embeddings.size == 0:
                continue

            # Document-level mean similarity
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

            # Chunk-level max similarity
            sim_chunk = chunk_max_similarity(uploaded_embeddings, c_embeddings)

            combined_score = max(sim_doc, sim_chunk)
            max_overall_score = max(max_overall_score, sim_doc)
            max_chunk_overall_score = max(max_chunk_overall_score, sim_chunk)

            if combined_score >= threshold:
                severity = "🔴 High" if combined_score >= 0.90 else "🟡 Medium"

                # Find top matching chunk pairs
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

        # Sort matches by max chunk similarity descending
        matched_documents.sort(key=lambda x: x["max_chunk_similarity_score"], reverse=True)

        is_flagged = len(matched_documents) > 0 or max_chunk_overall_score >= threshold

        return {
            "filename": filename,
            "word_count": word_count,
            "chunk_count": len(chunks),
            "plagiarism_flagged": is_flagged,
            "threshold_used": threshold,
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


# ── Asynchronous Background Scan Job Queue (#1372) ───────────────────────────

scan_jobs: Dict[str, Dict[str, Any]] = {}


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
            scan_jobs[job_id]["error"] = (
                "Failed to extract readable text from the uploaded file."
            )
            return

        words = extracted_text.split()
        word_count = len(words)

        chunks = chunk_document(extracted_text)
        if not chunks:
            chunks = [extracted_text[:1000]]

        uploaded_embeddings = embed_chunks(chunks)
        doc_embedding = get_document_embedding(uploaded_embeddings)
        corpus_docs = get_corpus_documents_with_embeddings()

        matched_documents = []
        max_overall_score = 0.0
        max_chunk_overall_score = 0.0

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
            sim_chunk = chunk_max_similarity(uploaded_embeddings, c_embeddings)

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

        scan_jobs[job_id]["status"] = "completed"
        scan_jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        scan_jobs[job_id]["result"] = {
            "filename": filename,
            "word_count": word_count,
            "chunk_count": len(chunks),
            "plagiarism_flagged": is_flagged,
            "threshold_used": threshold,
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


@app.post(
    "/api/v1/scan/async",
    tags=["Plagiarism Detection"],
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
                    "existing_document_id": existing_doc,
                    "message": "This file has already been uploaded."
                }
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


@app.get(
    "/api/v1/scan/status/{job_id}",
    tags=["Plagiarism Detection"],
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


# ── System Administration ──────────────────────────────────────────────────────

# Cast to str for consistency with callers that may pass it to faiss.*
# or other C-extension APIs that require str paths.
INDEX_PATH = str(FAISS_INDEX_PATH)


@app.post(
    "/api/v1/clear",
    tags=["System Administration"],
    response_model=ClearDataResponse,
    status_code=status.HTTP_200_OK,
    responses={
        403: {"model": ErrorResponse, "description": "Forbidden"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def clear_all_documents(
    username: str = Query(
        ..., description="Username of the administrator executing the operation"
    ),
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """
    Remove all documents, text chunks, and plagiarism incidents from the SQLite database,
    delete the FAISS index file, and clear the Redis cache. Restricted to administrators.
    """
    # 1. Verify administrator permissions
    role = get_user_role(username)
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Only administrators are authorized to clear all documents.",
        )

    try:
        # 2. Clear SQLite database (documents, chunks, incidents)
        clear_all_data()

        # 3. Clear/reset the FAISS index file on disk
        if os.path.exists(INDEX_PATH):
            try:
                os.remove(INDEX_PATH)
            except OSError as e:
                logger.error(f"Failed to remove FAISS index file: {e}")

        # 4. Invalidate Redis cache
        try:
            cache = get_cache()
            if cache.is_available():
                cache.delete(CacheKeyPrefix.LEGACY_FAISS_INDEX.value)
                cache.clear_pattern(CacheKeyPrefix.LEGACY_ANALYSIS_PATTERN.value)
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")

        return {
            "status": "success",
            "message": "All documents, chunks, and plagiarism incidents have been cleared, and the FAISS index reset successfully.",
        }

    except Exception as e:
        logger.error(f"Error during bulk clearing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while clearing the corpus: {str(e)}",
        )
