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

"""
src/api/endpoints/drift_detection.py
------------------------------------
FastAPI router for Intra-Document Style Drift Detection.

Provides REST endpoints to submit documents for sliding-window stylometric
analysis and change-point detection.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.changepoint_analysis import detect_cusum_changepoints
from src.core.style_drift_detector import extract_sliding_window_features
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
