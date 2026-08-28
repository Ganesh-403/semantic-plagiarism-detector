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

"""src/api/routers/corpus.py - Corpus document management and reset router."""

import logging
import os

from fastapi import APIRouter, HTTPException, Query, Request, Security, status

from src.api.dependencies import get_current_user
from src.api.schemas import ClearDataResponse, ErrorResponse
from src.core.app_config import FAISS_INDEX_PATH
from src.db.auth import get_user_role, log_security_event
from src.db.corpus_db import clear_all_data
from src.utils.redis_cache import CacheNamespace, get_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System Administration"])
INDEX_PATH = str(FAISS_INDEX_PATH)


@router.post(
    "/api/v1/clear",
    response_model=ClearDataResponse,
    status_code=status.HTTP_200_OK,
    responses={
        403: {"model": ErrorResponse, "description": "Forbidden"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def clear_all_documents(
    request: Request,
    username: str = Query(
        ..., description="Username of the administrator executing the operation"
    ),
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """
    Remove all documents, text chunks, and plagiarism incidents from the SQLite database,
    delete the FAISS index file, and clear the Redis cache. Restricted to administrators.
    """
    role = get_user_role(username)
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Only administrators are authorized to clear all documents.",
        )

    try:
        clear_all_data()

        if os.path.exists(INDEX_PATH):
            try:
                os.remove(INDEX_PATH)
            except OSError as e:
                logger.error(f"Failed to remove FAISS index file: {e}")

        try:
            cache = get_cache()
            if cache.is_available():
                cache.clear_pattern(CacheNamespace.FAISS.build_key("*"))
                cache.clear_pattern(CacheNamespace.ANALYSIS.build_key("*"))
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")

        client_ip = "127.0.0.1"
        if request.client:
            client_ip = request.client.host
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).isoformat()

        log_security_event(
            event_type="CORPUS_CLEARED",
            username=username,
            details=f"Corpus cleared by administrator. Client IP: {client_ip}, Timestamp: {timestamp}.",
        )

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
