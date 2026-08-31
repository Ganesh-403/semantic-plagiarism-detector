"""src/api/routers/batch_history.py - Batch analysis history API router."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Security, status

from src.api.dependencies import get_current_user
from src.api.schemas import (
    BatchAlertResponse,
    BatchHistorySummaryResponse,
    BatchRunCreateResponse,
    BatchRunDetailResponse,
    BatchRunListResponse,
    BatchTimelineEventResponse,
    BatchTrendDataResponse,
    ErrorResponse,
)
from src.db.batch_history_db import batch_repo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Batch Analysis History"])


# ---------------------------------------------------------------------------
# Batch Runs
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/batch/runs",
    response_model=BatchRunCreateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def create_batch_run(
    request: Request,
    threshold: float = Query(0.75, ge=0.0, le=1.0, description="Similarity threshold"),
    trigger: str = Query("manual", description="Trigger source"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Start a new batch analysis run."""
    try:
        run_id = batch_repo.create_batch_run(
            trigger_source=trigger,
            threshold_used=threshold,
            created_by=_user.get("sub", "unknown"),
        )
        batch_repo.add_timeline_event(
            event_type="batch_started",
            message=f"Batch run #{run_id} started (threshold={threshold}, trigger={trigger})",
            run_id=run_id,
            severity="info",
            metadata={"threshold": threshold, "trigger": trigger},
        )
        return BatchRunCreateResponse(
            run_id=run_id,
            status="running",
            message=f"Batch run #{run_id} created successfully",
        )
    except Exception as exc:
        logger.error("Failed to create batch run: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/batch/runs",
    response_model=BatchRunListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def list_batch_runs(
    request: Request,
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    trigger: Optional[str] = Query(None, alias="trigger_source", description="Filter by trigger"),
    start_date: Optional[str] = Query(None, description="ISO start date"),
    end_date: Optional[str] = Query(None, description="ISO end date"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """List batch analysis runs with filtering and pagination."""
    try:
        offset = (page - 1) * per_page
        runs = batch_repo.list_batch_runs(
            status=status_filter,
            trigger_source=trigger,
            start_date=start_date,
            end_date=end_date,
            limit=per_page,
            offset=offset,
        )
        total = batch_repo.count_batch_runs(
            status=status_filter,
            trigger_source=trigger,
        )
        total_pages = (total + per_page - 1) // per_page if total > 0 else 0
        return BatchRunListResponse(
            runs=runs,
            page=page,
            per_page=per_page,
            total_items=total,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )
    except Exception as exc:
        logger.error("Failed to list batch runs: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/batch/runs/{run_id}",
    response_model=BatchRunDetailResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_batch_run_detail(
    run_id: int,
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Retrieve detailed information about a specific batch run including its documents."""
    try:
        run = batch_repo.get_batch_run(run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch run #{run_id} not found",
            )
        documents = batch_repo.get_batch_documents(run_id)
        severity_dist = batch_repo.get_severity_distribution(run_id)
        return BatchRunDetailResponse(
            run=run,
            documents=documents,
            severity_distribution=severity_dist,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get batch run %s: %s", run_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.post(
    "/api/v1/batch/runs/{run_id}/complete",
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def complete_batch_run(
    run_id: int,
    request: Request,
    documents_scanned: int = Query(0),
    documents_flagged: int = Query(0),
    avg_similarity: float = Query(0.0),
    max_similarity: float = Query(0.0),
    duration_ms: int = Query(0),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Mark a batch run as completed with summary statistics."""
    try:
        run = batch_repo.get_batch_run(run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch run #{run_id} not found",
            )
        batch_repo.complete_batch_run(
            run_id,
            documents_scanned=documents_scanned,
            documents_flagged=documents_flagged,
            avg_similarity=avg_similarity,
            max_similarity=max_similarity,
            duration_ms=duration_ms,
        )
        batch_repo.add_timeline_event(
            event_type="batch_completed",
            message=f"Batch run #{run_id} completed: {documents_scanned} scanned, {documents_flagged} flagged",
            run_id=run_id,
            severity="success",
            metadata={
                "documents_scanned": documents_scanned,
                "documents_flagged": documents_flagged,
                "avg_similarity": avg_similarity,
                "max_similarity": max_similarity,
                "duration_ms": duration_ms,
            },
        )
        # Check for high-plagiarism alerts
        if documents_flagged > 0 and max_similarity > 0.9:
            batch_repo.create_alert(
                alert_type="high_plagiarism",
                title=f"High plagiarism detected in run #{run_id}",
                message=f"{documents_flagged} documents flagged with peak similarity {max_similarity:.1%}",
                run_id=run_id,
            )
        return {"status": "completed", "run_id": run_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to complete batch run %s: %s", run_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.post(
    "/api/v1/batch/runs/{run_id}/fail",
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def fail_batch_run(
    run_id: int,
    request: Request,
    error_message: str = Query("Unknown error"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Mark a batch run as failed."""
    try:
        run = batch_repo.get_batch_run(run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch run #{run_id} not found",
            )
        batch_repo.fail_batch_run(run_id, error_message)
        batch_repo.add_timeline_event(
            event_type="batch_failed",
            message=f"Batch run #{run_id} failed: {error_message}",
            run_id=run_id,
            severity="error",
            metadata={"error_message": error_message},
        )
        batch_repo.create_alert(
            alert_type="batch_failure",
            title=f"Batch run #{run_id} failed",
            message=error_message,
            run_id=run_id,
        )
        return {"status": "failed", "run_id": run_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to mark run %s as failed: %s", run_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.delete(
    "/api/v1/batch/runs/{run_id}",
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def delete_batch_run(
    run_id: int,
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Delete a batch run and all its associated records."""
    try:
        deleted = batch_repo.delete_batch_run(run_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch run #{run_id} not found",
            )
        return {"status": "deleted", "run_id": run_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete batch run %s: %s", run_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Batch Documents
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/batch/runs/{run_id}/documents",
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def add_batch_document(
    run_id: int,
    request: Request,
    document_name: str = Query(..., description="Document filename"),
    similarity_score: float = Query(0.0, description="Similarity score"),
    severity: str = Query("none", description="Severity level"),
    flagged: bool = Query(False, description="Whether flagged as plagiarized"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Add a document result to an existing batch run."""
    try:
        run = batch_repo.get_batch_run(run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch run #{run_id} not found",
            )
        doc_id = batch_repo.add_batch_document(
            run_id=run_id,
            document_name=document_name,
            similarity_score=similarity_score,
            severity=severity,
            flagged=flagged,
        )
        # Auto-generate alert for high-severity documents
        if severity == "high" and flagged:
            batch_repo.create_alert(
                alert_type="high_plagiarism",
                title=f"High plagiarism: {document_name}",
                message=f"Document '{document_name}' scored {similarity_score:.1%} similarity",
                run_id=run_id,
            )
        return {"status": "created", "document_id": doc_id, "run_id": run_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to add document to run %s: %s", run_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/batch/timeline",
    response_model=list[BatchTimelineEventResponse],
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_timeline(
    request: Request,
    run_id: Optional[int] = Query(None, description="Filter by run ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    start_date: Optional[str] = Query(None, description="ISO start date"),
    limit: int = Query(50, ge=1, le=200),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Retrieve the audit trail timeline events."""
    try:
        events = batch_repo.get_timeline_events(
            run_id=run_id,
            event_type=event_type,
            severity=severity,
            start_date=start_date,
            limit=limit,
        )
        return [BatchTimelineEventResponse(**e) for e in events]
    except Exception as exc:
        logger.error("Failed to retrieve timeline: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/batch/alerts",
    response_model=list[BatchAlertResponse],
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_alerts(
    request: Request,
    is_read: Optional[bool] = Query(None, description="Filter by read status"),
    limit: int = Query(50, ge=1, le=200),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Retrieve alerts with optional read-status filter."""
    try:
        alerts = batch_repo.get_alerts(is_read=is_read, limit=limit)
        return [BatchAlertResponse(**a) for a in alerts]
    except Exception as exc:
        logger.error("Failed to retrieve alerts: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.put(
    "/api/v1/batch/alerts/read-all",
    status_code=status.HTTP_200_OK,
)
async def mark_all_alerts_read(
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Mark all unread alerts as read."""
    try:
        count = batch_repo.mark_all_alerts_read()
        return {"status": "ok", "marked_read": count}
    except Exception as exc:
        logger.error("Failed to mark alerts as read: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/batch/alerts/unread-count",
    status_code=status.HTTP_200_OK,
)
async def get_unread_alert_count(
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Get the count of unread alerts."""
    try:
        count = batch_repo.get_unread_alert_count()
        return {"unread_count": count}
    except Exception as exc:
        logger.error("Failed to get unread alert count: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/batch/analytics/summary",
    response_model=BatchHistorySummaryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_summary(
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Get high-level summary statistics for all batch runs."""
    try:
        stats = batch_repo.get_summary_stats()
        return BatchHistorySummaryResponse(**stats)
    except Exception as exc:
        logger.error("Failed to get summary: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/batch/analytics/trends",
    response_model=list[BatchTrendDataResponse],
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_trends(
    request: Request,
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Get daily trend data for the last N days."""
    try:
        trends = batch_repo.get_trend_data(days=days)
        return [BatchTrendDataResponse(**t) for t in trends]
    except Exception as exc:
        logger.error("Failed to get trends: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/batch/analytics/severity",
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_severity_distribution(
    request: Request,
    run_id: Optional[int] = Query(None, description="Filter by specific run"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Get the severity distribution across document results."""
    try:
        dist = batch_repo.get_severity_distribution(run_id=run_id)
        return {"distribution": dist}
    except Exception as exc:
        logger.error("Failed to get severity distribution: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/batch/purge",
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def purge_old_runs(
    request: Request,
    days: int = Query(90, ge=7, le=365, description="Purge runs older than N days"),
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Purge batch runs older than the specified number of days."""
    try:
        deleted = batch_repo.purge_old_runs(days=days)
        batch_repo.add_timeline_event(
            event_type="system_maintenance",
            message=f"Purged {deleted} batch runs older than {days} days",
            severity="info",
            metadata={"days": days, "deleted_count": deleted},
        )
        return {"status": "purged", "deleted_count": deleted, "days_threshold": days}
    except Exception as exc:
        logger.error("Failed to purge old runs: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
