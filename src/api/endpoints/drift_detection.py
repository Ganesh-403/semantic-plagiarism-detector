"""
src/api/endpoints/drift_detection.py
------------------------------------
FastAPI router for Intra-Document Style Drift Detection.

Provides REST endpoints to submit documents for sliding-window stylometric
analysis and change-point detection.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

from src.core.style_drift_detector import extract_sliding_window_features
from src.core.changepoint_analysis import detect_cusum_changepoints
from src.db.drift_alerts_db import initialize_drift_db, log_drift_alert

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drift", tags=["Drift Detection"])

initialize_drift_db()


class DriftRequest(BaseModel):
    """Schema for drift detection requests."""

    text: str = Field(..., min_length=100, description="Document text to analyze.")
    document_id: str = Field("api_doc", description="Document ID for logging.")
    window_size: int = Field(500, ge=100, le=2000)
    step_size: int = Field(250, ge=50, le=1000)


class DriftResponse(BaseModel):
    """Schema for drift detection responses."""

    document_id: str
    num_windows: int
    changepoints: List[Dict[str, Any]]
    max_confidence: float


@router.post("/analyze", response_model=DriftResponse)
async def analyze_drift(request: DriftRequest):
    """Analyze a document for intra-document style drift and contract cheating."""
    try:
        features = extract_sliding_window_features(
            request.text, window_size=request.window_size, step_size=request.step_size
        )

        changepoints = detect_cusum_changepoints(features)
        max_conf = max([cp.get("confidence", 0.0) for cp in changepoints], default=0.0)

        log_drift_alert(request.document_id, changepoints)

        return DriftResponse(
            document_id=request.document_id,
            num_windows=len(features),
            changepoints=changepoints,
            max_confidence=max_conf,
        )
    except Exception as e:
        logger.error("Drift detection failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
