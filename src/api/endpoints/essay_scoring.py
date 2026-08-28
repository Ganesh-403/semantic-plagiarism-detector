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
src/api/endpoints/essay_scoring.py
----------------------------------
FastAPI router for Automated Essay Scoring and Trait Analysis.

Provides REST endpoints to submit essays for automated trait analysis
and holistic grading.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.essay_scorer import (
    DEFAULT_RUBRIC,
    RubricCriterion,
    ScoringRubric,
    score_essay,
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
