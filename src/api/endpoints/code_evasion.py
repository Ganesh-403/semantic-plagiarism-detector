"""
src/api/endpoints/code_evasion.py
---------------------------------
FastAPI router for Code Execution Output Fingerprinting and Evasion Detection.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

from src.core.output_fingerprinter import generate_output_fingerprints
from src.core.test_evasion_detector import analyze_test_evasion
from src.db.evasion_logs_db import initialize_evasion_logs_db, log_evasion_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/code-evasion", tags=["Code Evasion"])

initialize_evasion_logs_db()


class EvasionRequest(BaseModel):
    """Schema for code evasion requests."""

    code: str = Field(..., min_length=1, description="Submitted source code.")
    original_output: str = Field(..., description="Output from original test case.")
    mutated_outputs: List[str] = Field(..., description="Outputs from mutated inputs.")
    document_id: str = Field("api_code", description="Document ID for logging.")


class EvasionResponse(BaseModel):
    """Schema for code evasion responses."""

    document_id: str
    evasion_risk_score: float
    is_suspicious: bool
    evasion_patterns: List[str]
    output_variance: float


@router.post("/analyze", response_model=EvasionResponse)
async def analyze_code_evasion(request: EvasionRequest):
    """Submit code for test-case evasion analysis."""
    try:
        output_metrics = generate_output_fingerprints(
            request.original_output, request.mutated_outputs
        )

        result = analyze_test_evasion(request.code, output_metrics)

        log_evasion_analysis(
            document_id=request.document_id,
            evasion_risk_score=result["evasion_risk_score"],
            is_suspicious=result["is_suspicious"],
            evasion_patterns=result["evasion_patterns"],
        )

        return EvasionResponse(
            document_id=request.document_id,
            evasion_risk_score=result["evasion_risk_score"],
            is_suspicious=result["is_suspicious"],
            evasion_patterns=result["evasion_patterns"],
            output_variance=result["output_variance"],
        )
    except Exception as e:
        logger.error("Code evasion analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
