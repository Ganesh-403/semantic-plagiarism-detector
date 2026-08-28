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
src/api/endpoints/calibration.py
--------------------------------
FastAPI router for Reviewer Calibration and IRR endpoints.

Provides REST endpoints to fetch reviewer calibration scores, log manual
overrides, and compute committee Inter-Rater Reliability.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.reliability_engine import (
    calculate_calibration_weight,
    compute_cohens_kappa,
    compute_fleiss_kappa,
    compute_reviewer_bias,
)
from src.db.reviewer_calibration_db import (
    get_reviewer_weight,
    initialize_calibration_db,
    log_review_override,
    update_reviewer_metrics,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calibration", tags=["Calibration"])

# Initialize DB on import
initialize_calibration_db()


class OverrideRequest(BaseModel):
    """Schema for logging a manual review override."""

    reviewer_id: str
    document_id: str
    automated_score: float = Field(..., ge=0.0, le=1.0)
    manual_score: float = Field(..., ge=0.0, le=1.0)


class KappaRequest(BaseModel):
    """Schema for computing Fleiss' or Cohen's Kappa."""

    ratings_matrix: list[list[int]] = Field(
        ..., description="Matrix of category counts per item."
    )


@router.post("/override", status_code=status.HTTP_201_CREATED)
async def submit_override(request: OverrideRequest):
    """Log a manual override and update the reviewer's calibration weight."""
    # In a real system, we would fetch all historical overrides for this reviewer
    # to compute the new bias metrics. Here we simulate it.
    success = log_review_override(
        request.reviewer_id,
        request.document_id,
        request.automated_score,
        request.manual_score,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to log override.")

    # Simulate metric update (would normally aggregate all historical data)
    bias = compute_reviewer_bias([request.manual_score], [request.automated_score])
    weight = calculate_calibration_weight(bias)

    update_reviewer_metrics(request.reviewer_id, bias, weight)

    return {"status": "success", "new_weight": weight}


@router.get("/weight/{reviewer_id}")
async def get_weight(reviewer_id: str):
    """Fetch the current calibration weight for a specific reviewer."""
    weight = get_reviewer_weight(reviewer_id)
    return {"reviewer_id": reviewer_id, "calibration_weight": weight}


@router.post("/kappa/fleiss")
async def compute_fleiss(request: KappaRequest):
    """Compute Fleiss' Kappa for a committee of raters."""
    try:
        kappa = compute_fleiss_kappa(request.ratings_matrix)
        return {"metric": "fleiss_kappa", "score": kappa}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# semantic-plagiarism-detector/src/api/endpoints/calibration.py

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from src.core.reliability_engine import ReliabilityEngine
from src.db.reviewer_calibration_db import ReviewerCalibrationDB

router = APIRouter(prefix="/api/v1/calibration", tags=["Reviewer Calibration & IRR"])
db = ReviewerCalibrationDB()
engine = ReliabilityEngine()


@router.get("/reviewer/{reviewer_id}")
def get_reviewer_calibration(reviewer_id: str) -> dict[str, Any]:
    """Fetches calibration scores and historical bias weighting for a specific reviewer."""
    history = db.fetch_reviewer_history(reviewer_id)
    if not history:
        raise HTTPException(status_code=404, detail="Reviewer history not found.")

    weights = engine.compute_reviewer_bias_weights(history)
    return {
        "reviewer_id": reviewer_id,
        "total_reviews": len(history),
        "calibration_weight": weights.get(reviewer_id, 1.0),
    }


@router.get("/committee/irr")
def get_committee_irr(ratings_matrix: list[list[int]]) -> dict[str, float]:
    """Computes and returns committee Inter-Rater Reliability (Fleiss' Kappa)."""
    kappa = engine.compute_fleiss_kappa(ratings_matrix)
    return {
        "fleiss_kappa": kappa,
        "reliability_status": (
            "High Agreement" if kappa > 0.6 else "Moderate/Low Agreement"
        ),
    }
