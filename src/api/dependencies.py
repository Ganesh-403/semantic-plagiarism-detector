"""src/api/dependencies.py - Shared API dependencies and context helpers."""

import logging
from typing import Dict

import numpy as np
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.middleware import (  # noqa: F401
    RequireScopes,
    extract_token_scopes,
    get_current_user,
    get_current_user_any,
    require_all_scopes,
    require_any_scope,
    require_any_scopes,
    require_scopes,
    validate_scopes,
    verify_bearer_token,
)
from src.db.corpus_db import _connect, init_corpus_db

logger = logging.getLogger(__name__)

# SlowAPI Rate Limiter instance
limiter = Limiter(key_func=get_remote_address)


def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom exception handler for rate limit exceeded errors adhering to RFC 7807."""
    detail = f"Rate limit exceeded: {exc.detail}"
    path = getattr(getattr(request, "url", None), "path", None)
    response = JSONResponse(
        status_code=429,
        content={
            "type": "about:blank",
            "title": "Too Many Requests",
            "status": 429,
            "detail": detail,
            "instance": path,
            "error": True,
            "code": 429,
            "message": detail,
        },
        media_type="application/problem+json",
    )
    response = request.app.state.limiter._inject_headers(
        response, request.state.view_rate_limit
    )
    return response


def validate_content_type(request: Request) -> None:
    """Ensure the request is multipart/form-data before parsing."""
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Media Type: Request must be multipart/form-data",
        )


def get_corpus_documents_with_embeddings() -> dict[str, dict]:
    """Load all stored corpus documents, text chunks, and chunk embeddings from SQLite."""
    init_corpus_db()
    corpus: dict[str, dict] = {}

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
