"""src/api/routers/health_score.py - Document health scoring API router."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Security, status

from src.api.dependencies import get_current_user
from src.api.schemas import (
    DocumentHealthScoreResponse,
    DocumentHealthListResponse,
    HealthScoreSummaryResponse,
    HealthGateConfigResponse,
    HealthGateCheckResponse,
    HealthDimensionAvgResponse,
    ErrorResponse,
)
from src.core.document_health_scorer import (
    compute_quality_gate,
    aggregate_reports,
)
from src.db.health_score_db import health_repo

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Document Health Scoring"])


# ---------------------------------------------------------------------------
# Score Management
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/health/score",
    response_model=DocumentHealthScoreResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def score_single_document(
    request: Request,
    filename: str = Query(..., description="Document filename to score"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """
    Score a single document and persist the health report.

    Computes the overall health score across metadata, chunk balance,
    embedding coverage, content quality, and fingerprint uniqueness
    dimensions, then runs the quality gate check.
    """
    try:
        from src.db.corpus_db import (
            get_all_documents,
            get_document_chunks_count,
            get_document_word_counts,
            get_chunks_for_documents,
        )

        docs = get_all_documents()
        doc = next((d for d in docs if d.filename == filename), None)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{filename}' not found in corpus",
            )

        # Build doc dict
        doc_dict = {
            "filename": doc.filename,
            "file_hash": doc.file_hash,
            "student_name": getattr(doc, "student_name", None),
            "class_section": getattr(doc, "class_section", None),
            "assignment_title": getattr(doc, "assignment_title", None),
            "detected_language": getattr(doc, "detected_language", None),
            "tags": getattr(doc, "tags", None),
        }

        # Get chunks
        chunks_result = get_chunks_for_documents([filename])
        chunk_texts: list[str] = []
        embeddings = None
        if filename in chunks_result:
            chunk_texts, embeddings = chunks_result[filename]

        total_chunks = len(chunk_texts)
        chunks_with_emb = len(embeddings) if embeddings is not None else 0
        word_counts = get_document_word_counts()
        total_words = word_counts.get(filename, 0)

        existing_hashes = {d.file_hash for d in docs if d.file_hash}

        from src.core.document_health_scorer import score_document
        report = score_document(
            doc=doc_dict,
            chunk_texts=chunk_texts,
            chunk_word_counts=[len(t.split()) for t in chunk_texts],
            total_chunks=total_chunks,
            chunks_with_embeddings=chunks_with_emb,
            total_words=total_words,
            existing_hashes=existing_hashes,
        )

        # Quality gate check
        config = health_repo.get_gate_config()
        gate = compute_quality_gate(
            report,
            min_score=float(config.get("min_score", "60.0")),
            min_grade=config.get("min_grade", "D"),
        )

        # Persist
        score_id = health_repo.save_score(
            filename=filename,
            overall_score=report.overall_score,
            grade=report.grade,
            dimensions=[d.__dict__ for d in report.dimensions],
            metadata=report.metadata,
            gate_passed=gate["passed"],
            gate_reason=gate["reason"],
        )

        return DocumentHealthScoreResponse(
            id=score_id,
            filename=filename,
            overall_score=report.overall_score,
            grade=report.grade,
            dimensions=[d.__dict__ for d in report.dimensions],
            checked_at=report.checked_at,
            gate_passed=gate["passed"],
            gate_reason=gate["reason"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to score document %s: %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/health/scores",
    response_model=DocumentHealthListResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def list_health_scores(
    request: Request,
    min_score: Optional[float] = Query(None, description="Minimum score filter"),
    max_score: Optional[float] = Query(None, description="Maximum score filter"),
    grade: Optional[str] = Query(None, description="Filter by letter grade"),
    gate_passed: Optional[bool] = Query(None, description="Filter by gate result"),
    sort_by: str = Query("overall_score", description="Sort field"),
    sort_order: str = Query("DESC", description="Sort direction"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """List document health scores with filtering and pagination."""
    try:
        offset = (page - 1) * per_page
        scores = health_repo.list_scores(
            min_score=min_score,
            max_score=max_score,
            grade=grade,
            gate_passed=gate_passed,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=per_page,
            offset=offset,
        )
        total = health_repo.count_scores(
            min_score=min_score,
            gate_passed=gate_passed,
        )
        total_pages = (total + per_page - 1) // per_page if total > 0 else 0
        return DocumentHealthListResponse(
            scores=scores,
            page=page,
            per_page=per_page,
            total_items=total,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )
    except Exception as exc:
        logger.error("Failed to list health scores: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/health/scores/{filename}",
    response_model=DocumentHealthScoreResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Score not found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_document_score(
    filename: str,
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Get the latest health score for a specific document."""
    try:
        score = health_repo.get_latest_score(filename)
        if not score:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No health score found for '{filename}'",
            )
        return DocumentHealthScoreResponse(
            id=score.get("id", 0),
            filename=score.get("filename", filename),
            overall_score=score.get("overall_score", 0),
            grade=score.get("grade", "F"),
            dimensions=score.get("dimension_data", []),
            checked_at=score.get("checked_at", ""),
            gate_passed=bool(score.get("gate_passed", 0)),
            gate_reason=score.get("gate_reason", ""),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get score for %s: %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/health/scores/{filename}/history",
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_score_history(
    filename: str,
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Retrieve historical health scores for a document."""
    try:
        history = health_repo.get_score_history(filename, limit=limit)
        return {"filename": filename, "history": history, "count": len(history)}
    except Exception as exc:
        logger.error("Failed to get history for %s: %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Quality Gate
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/health/gate/config",
    response_model=HealthGateConfigResponse,
    status_code=status.HTTP_200_OK,
)
async def get_gate_config(
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Retrieve the current quality gate configuration."""
    config = health_repo.get_gate_config()
    return HealthGateConfigResponse(
        min_score=float(config.get("min_score", "60.0")),
        min_grade=config.get("min_grade", "D"),
        enabled=config.get("enabled", "true") == "true",
    )


@router.put(
    "/api/v1/health/gate/config",
    response_model=HealthGateConfigResponse,
    status_code=status.HTTP_200_OK,
)
async def update_gate_config(
    request: Request,
    min_score: Optional[float] = Query(None, ge=0, le=100),
    min_grade: Optional[str] = Query(None),
    enabled: Optional[bool] = Query(None),
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Update the quality gate configuration."""
    if min_score is not None:
        health_repo.set_gate_config("min_score", str(min_score))
    if min_grade is not None:
        health_repo.set_gate_config("min_grade", min_grade)
    if enabled is not None:
        health_repo.set_gate_config("enabled", str(enabled).lower())

    config = health_repo.get_gate_config()
    return HealthGateConfigResponse(
        min_score=float(config.get("min_score", "60.0")),
        min_grade=config.get("min_grade", "D"),
        enabled=config.get("enabled", "true") == "true",
    )


@router.post(
    "/api/v1/health/gate/check",
    response_model=HealthGateCheckResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "No score found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def check_quality_gate(
    request: Request,
    filename: str = Query(..., description="Document filename"),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst"]),
):
    """Check if a document passes the quality gate based on its latest score."""
    try:
        score = health_repo.get_latest_score(filename)
        if not score:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No health score found for '{filename}'. Run scoring first.",
            )

        config = health_repo.get_gate_config()
        from src.core.document_health_scorer import HealthReport

        report = HealthReport(
            filename=filename,
            overall_score=score.get("overall_score", 0),
            grade=score.get("grade", "F"),
            dimensions=[],
            checked_at=score.get("checked_at", ""),
        )

        gate = compute_quality_gate(
            report,
            min_score=float(config.get("min_score", "60.0")),
            min_grade=config.get("min_grade", "D"),
        )

        return HealthGateCheckResponse(
            filename=filename,
            passed=gate["passed"],
            reason=gate["reason"],
            overall_score=gate["overall_score"],
            grade=gate["grade"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed quality gate check for %s: %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/health/analytics/summary",
    response_model=HealthScoreSummaryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_health_summary(
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Get aggregate health score statistics."""
    try:
        summary = health_repo.get_score_summary()
        return HealthScoreSummaryResponse(**summary)
    except Exception as exc:
        logger.error("Failed to get health summary: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/health/analytics/dimensions",
    response_model=HealthDimensionAvgResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_dimension_averages(
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Get average score per health dimension."""
    try:
        avgs = health_repo.get_dimension_averages()
        return HealthDimensionAvgResponse(dimensions=avgs)
    except Exception as exc:
        logger.error("Failed to get dimension averages: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/health/analytics/worst",
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_worst_documents(
    request: Request,
    limit: int = Query(10, ge=1, le=50),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Get the documents with the lowest health scores."""
    try:
        docs = health_repo.get_worst_documents(limit=limit)
        return {"documents": docs, "count": len(docs)}
    except Exception as exc:
        logger.error("Failed to get worst documents: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get(
    "/api/v1/health/analytics/best",
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_best_documents(
    request: Request,
    limit: int = Query(10, ge=1, le=50),
    _user: dict = Security(get_current_user, scopes=["admin", "analyst", "viewer"]),
):
    """Get the documents with the highest health scores."""
    try:
        docs = health_repo.get_best_documents(limit=limit)
        return {"documents": docs, "count": len(docs)}
    except Exception as exc:
        logger.error("Failed to get best documents: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------


@router.delete(
    "/api/v1/health/scores/{filename}",
    status_code=status.HTTP_200_OK,
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def delete_document_scores(
    filename: str,
    request: Request,
    _user: dict = Security(get_current_user, scopes=["admin"]),
):
    """Delete all health scores for a document."""
    try:
        count = health_repo.delete_scores_for_document(filename)
        return {"status": "deleted", "filename": filename, "records_deleted": count}
    except Exception as exc:
        logger.error("Failed to delete scores for %s: %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
