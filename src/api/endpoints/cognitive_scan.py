"""
src/api/endpoints/cognitive_scan.py
-----------------------------------
FastAPI router for Cognitive Load and Readability Fingerprinting.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from src.core.readability_analyzer import extract_readability_timeseries
from src.core.cognitive_load_fingerprinter import analyze_cognitive_load
from src.db.cognitive_load_db import (
    initialize_cognitive_load_db,
    log_cognitive_load_analysis,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cognitive-scan", tags=["Cognitive Scan"])
initialize_cognitive_load_db()


class CognitiveScanRequest(BaseModel):
    text: str = Field(
        ..., min_length=50, description="Text to analyze for cognitive load."
    )
    document_id: str = "api_doc"
    window_size: int = Field(100, ge=20, le=500)


class CognitiveScanResponse(BaseModel):
    document_id: str
    ai_probability: float
    is_ai_generated: bool
    fk_variance: float
    cli_variance: float


@router.post("/analyze", response_model=CognitiveScanResponse)
async def analyze_cognitive_load_endpoint(request: CognitiveScanRequest):
    try:
        timeseries = extract_readability_timeseries(request.text, request.window_size)
        result = analyze_cognitive_load(timeseries)

        log_cognitive_load_analysis(
            request.document_id,
            result["variance_metrics"]["ai_probability"],
            result["is_ai_generated"],
            result["variance_metrics"]["fk_variance"],
        )

        return CognitiveScanResponse(
            document_id=request.document_id,
            ai_probability=result["variance_metrics"]["ai_probability"],
            is_ai_generated=result["is_ai_generated"],
            fk_variance=result["variance_metrics"]["fk_variance"],
            cli_variance=result["variance_metrics"]["cli_variance"],
        )
    except Exception as e:
        logger.error("Cognitive scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
