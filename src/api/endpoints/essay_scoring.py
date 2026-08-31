"""
src/api/endpoints/essay_scoring.py
----------------------------------
FastAPI router for Automated Essay Scoring and Trait Analysis.

Provides REST endpoints to submit essays for automated trait analysis
and holistic grading.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

from src.core.essay_scorer import (
    score_essay,
    DEFAULT_RUBRIC,
    ScoringRubric,
    RubricCriterion,
)
from src.db.essay_scores_db import initialize_essay_scores_db, log_essay_score

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/essay-scoring", tags=["Essay Scoring"])

initialize_essay_scores_db()


class ScoringRequest(BaseModel):
    """Schema for essay scoring requests."""

    text: str = Field(..., min_length=50, description="Essay text to score.")
    document_id: str = Field("api_essay", description="Document ID for logging.")


class ScoringResponse(BaseModel):
    """Schema for essay scoring responses."""

    document_id: str
    rubric_name: str
    final_grade: float
    traits: Dict[str, Any]
    criterion_scores: List[Dict[str, Any]]


@router.post("/score", response_model=ScoringResponse)
async def score_essay_endpoint(request: ScoringRequest):
    """Submit an essay for automated trait analysis and holistic grading."""
    try:
        result = score_essay(request.text, rubric=DEFAULT_RUBRIC)

        # Log the score
        log_essay_score(
            document_id=request.document_id,
            rubric_name=result["rubric_name"],
            final_grade=result["final_grade"],
            traits=result["traits"],
            criterion_scores=result["criterion_scores"],
        )

        return ScoringResponse(
            document_id=request.document_id,
            rubric_name=result["rubric_name"],
            final_grade=result["final_grade"],
            traits=result["traits"],
            criterion_scores=result["criterion_scores"],
        )

    except Exception as e:
        logger.error("Essay scoring failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
