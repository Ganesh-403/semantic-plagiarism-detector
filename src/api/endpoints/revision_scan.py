"""
src/api/endpoints/revision_scan.py
----------------------------------
FastAPI router for Document Revision Time-Series and Burst Analysis.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

from src.core.burst_detection_engine import analyze_revision_bursts
from src.db.revision_bursts_db import (
    initialize_revision_bursts_db,
    log_revision_burst_analysis,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/revision-scan", tags=["Revision Scan"])
initialize_revision_bursts_db()


class RevisionScanRequest(BaseModel):
    timestamps: List[float] = Field(..., description="List of keystroke timestamps.")
    document_id: str = "api_doc"


class RevisionScanResponse(BaseModel):
    document_id: str
    risk_score: float
    is_ghostwritten: bool
    burst_ratio: float
    variance_mean_ratio: float


@router.post("/analyze", response_model=RevisionScanResponse)
async def analyze_revision_bursts_endpoint(request: RevisionScanRequest):
    try:
        result = analyze_revision_bursts(request.timestamps)
        log_revision_burst_analysis(
            request.document_id,
            result["risk_score"],
            result["is_ghostwritten"],
            result["burst_metrics"]["burst_ratio"],
        )
        return RevisionScanResponse(
            document_id=request.document_id,
            risk_score=result["risk_score"],
            is_ghostwritten=result["is_ghostwritten"],
            burst_ratio=result["burst_metrics"]["burst_ratio"],
            variance_mean_ratio=result["poisson_deviation"]["variance_mean_ratio"],
        )
    except Exception as e:
        logger.error("Revision scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
