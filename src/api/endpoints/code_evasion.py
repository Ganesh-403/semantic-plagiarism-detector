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
src/api/endpoints/code_evasion.py
---------------------------------
FastAPI router for Code Execution Output Fingerprinting and Evasion Detection.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

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
