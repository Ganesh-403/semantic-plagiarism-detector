"""
src/api/endpoints/steganography_scan.py
---------------------------------------
FastAPI router for Steganography and Prompt Injection Detection.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

from src.core.steganography_extractor import extract_zero_width_payloads
from src.core.prompt_injection_detector import analyze_steganography
from src.db.steganography_logs_db import (
    initialize_steganography_logs_db,
    log_steganography_analysis,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/steganography-scan", tags=["Steganography Scan"])
initialize_steganography_logs_db()


class SteganographyScanRequest(BaseModel):
    text: str = Field(..., min_length=1)
    document_id: str = "api_doc"


class SteganographyScanResponse(BaseModel):
    document_id: str
    is_injection: bool
    risk_score: float
    zero_width_payloads: int
    matched_patterns: List[str]


@router.post("/analyze", response_model=SteganographyScanResponse)
async def analyze_steganography_endpoint(request: SteganographyScanRequest):
    try:
        result = analyze_steganography(request.text)
        log_steganography_analysis(
            request.document_id,
            result["is_injection"],
            result["risk_score"],
            result["matched_patterns"],
        )
        return SteganographyScanResponse(
            document_id=request.document_id,
            is_injection=result["is_injection"],
            risk_score=result["risk_score"],
            zero_width_payloads=result["zero_width_payloads"],
            matched_patterns=result["matched_patterns"],
        )
    except Exception as e:
        logger.error("Steganography scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
