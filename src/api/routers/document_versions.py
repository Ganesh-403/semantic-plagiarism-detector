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

"""src/api/routers/document_versions.py - Document versioning API router."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Security, status

from src.api.dependencies import get_current_user
from src.api.schemas import (
    DocumentVersionDiffResponse,
    DocumentVersionLineageResponse,
    DocumentVersionListResponse,
    DocumentVersionMostRevisedResponse,
    DocumentVersionSnapshotResponse,
    DocumentVersionSummaryResponse,
    DocumentVersionTrendResponse,
    ErrorResponse,
)
from src.db.version_repo import init_version_repo_db, version_repo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Document Versioning"])

# Ensure DB is initialized on import
try:
    init_version_repo_db()
except Exception:
    logger.warning("Version repo DB init deferred")


# ---------------------------------------------------------------------------
# Versions CRUD
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/versions/snapshots",
    status_code=status.HTTP_201_CREATED,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def register_version(
    request: Request,
    user_id: str = Query(..., description="User who uploaded the document"),
    assignment_id: str = Query(..., description="Assignment identifier"),
    filename: str = Query("untitled", description="Document filename"),
    content_text: str = Query("", description="Full document text"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Register a new document version snapshot."""
    try:
        result = version_repo.register_version(
            user_id=user_id,
            assignment_id=assignment_id,
            filename=filename,
            content_text=content_text,
        )
        return result
    except Exception as e:
        logger.error("Failed to register version: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/v1/versions/snapshots",
    responses={
        200: {"description": "List of version snapshots"},
        500: {"model": ErrorResponse},
    },
)
async def list_snapshots(
    request: Request,
    user_id: str | None = Query(None, description="Filter by user"),
    assignment_id: str | None = Query(None, description="Filter by assignment"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """List document version snapshots with filtering and pagination."""
    try:
        return version_repo.list_versions(
            user_id=user_id,
            assignment_id=assignment_id,
            page=page,
            per_page=per_page,
        )
    except Exception as e:
        logger.error("Failed to list snapshots: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/v1/versions/snapshots/{doc_hash}",
    responses={
        200: {"description": "Version snapshot details"},
        404: {"description": "Not found"},
    },
)
async def get_snapshot(
    doc_hash: str,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Retrieve a single version snapshot by document hash."""
    snapshot = version_repo.get_snapshot(doc_hash)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return snapshot


@router.delete(
    "/api/v1/versions/snapshots/{doc_hash}",
    responses={
        200: {"description": "Deleted successfully"},
        404: {"description": "Not found"},
    },
)
async def delete_snapshot(
    doc_hash: str,
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Delete a specific version snapshot."""
    deleted = version_repo.delete_version(doc_hash)
    if not deleted:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return {"deleted": True, "document_hash": doc_hash}


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/versions/lineage",
    responses={
        200: {"description": "List of version lineages"},
        500: {"model": ErrorResponse},
    },
)
async def list_lineages(
    request: Request,
    user_id: str | None = Query(None, description="Filter by user"),
    assignment_id: str | None = Query(None, description="Filter by assignment"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """List all tracked version lineages."""
    try:
        return version_repo.list_lineages(
            user_id=user_id,
            assignment_id=assignment_id,
            page=page,
            per_page=per_page,
        )
    except Exception as e:
        logger.error("Failed to list lineages: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/v1/versions/lineage/{user_id}/{assignment_id}",
    responses={
        200: {"description": "Full version lineage"},
    },
)
async def get_lineage(
    user_id: str,
    assignment_id: str,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Get the complete version lineage for a user + assignment."""
    lineage = version_repo.get_lineage(user_id, assignment_id)
    return {
        "user_id": user_id,
        "assignment_id": assignment_id,
        "versions": lineage,
        "total": len(lineage),
    }


@router.delete(
    "/api/v1/versions/lineage/{user_id}/{assignment_id}",
    responses={
        200: {"description": "Deleted successfully"},
    },
)
async def delete_lineage(
    user_id: str,
    assignment_id: str,
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Delete an entire version lineage."""
    deleted = version_repo.delete_lineage(user_id, assignment_id)
    return {"deleted": deleted, "user_id": user_id, "assignment_id": assignment_id}


# ---------------------------------------------------------------------------
# Diffs
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/versions/diffs",
    status_code=status.HTTP_201_CREATED,
    responses={
        500: {"model": ErrorResponse},
    },
)
async def register_diff(
    request: Request,
    parent_hash: str = Query(..., description="Parent version hash"),
    child_hash: str = Query(..., description="Child version hash"),
    similarity: float = Query(0.0, ge=0.0, le=1.0),
    added_words: int = Query(0, ge=0),
    removed_words: int = Query(0, ge=0),
    changed_words: int = Query(0, ge=0),
    jaccard_index: float = Query(0.0, ge=0.0, le=1.0),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Register a pairwise diff between two versions."""
    try:
        diff_id = version_repo.register_diff(
            parent_hash=parent_hash,
            child_hash=child_hash,
            similarity=similarity,
            added_words=added_words,
            removed_words=removed_words,
            changed_words=changed_words,
            jaccard_index=jaccard_index,
        )
        return {
            "diff_id": diff_id,
            "parent_hash": parent_hash,
            "child_hash": child_hash,
        }
    except Exception as e:
        logger.error("Failed to register diff: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/v1/versions/diffs/{parent_hash}/{child_hash}",
    responses={
        200: {"description": "Diff details"},
        404: {"description": "Not found"},
    },
)
async def get_diff(
    parent_hash: str,
    child_hash: str,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Get the diff record between two versions."""
    diff = version_repo.get_diff(parent_hash, child_hash)
    if not diff:
        raise HTTPException(status_code=404, detail="Diff not found")
    return diff


@router.get(
    "/api/v1/versions/diffs/{doc_hash}/all",
    responses={
        200: {"description": "All diffs for a version"},
    },
)
async def get_diffs_for_version(
    doc_hash: str,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Get all diffs involving a specific version."""
    diffs = version_repo.get_diffs_for_version(doc_hash)
    return {"document_hash": doc_hash, "diffs": diffs, "total": len(diffs)}


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/versions/analytics/summary",
    responses={
        200: {"description": "Analytics summary"},
    },
)
async def analytics_summary(
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Return aggregate statistics across all versions."""
    return version_repo.analytics_summary()


@router.get(
    "/api/v1/versions/analytics/trend/{user_id}/{assignment_id}",
    responses={
        200: {"description": "Similarity trend data"},
    },
)
async def similarity_trend(
    user_id: str,
    assignment_id: str,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Return the similarity trend across versions for a lineage."""
    trend = version_repo.similarity_trend(user_id, assignment_id)
    return {
        "user_id": user_id,
        "assignment_id": assignment_id,
        "trend": trend,
        "total_points": len(trend),
    }


@router.get(
    "/api/v1/versions/analytics/most-revised",
    responses={
        200: {"description": "Most revised documents"},
    },
)
async def most_revised(
    limit: int = Query(10, ge=1, le=50),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Return documents with the most versions."""
    return {"documents": version_repo.most_revised_documents(limit=limit)}


@router.get(
    "/api/v1/versions/analytics/highest-drift",
    responses={
        200: {"description": "Highest drift documents"},
    },
)
async def highest_drift(
    limit: int = Query(10, ge=1, le=50),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Return documents with the most drift between versions."""
    return {"documents": version_repo.highest_drift_documents(limit=limit)}
