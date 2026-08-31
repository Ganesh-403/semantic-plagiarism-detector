"""
src/api/endpoints/audio_scan.py
-------------------------------
FastAPI router for Audio Transcript Prosody Analysis.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

from src.core.cadence_fingerprinter import analyze_prosody
from src.db.audio_prosody_db import initialize_audio_prosody_db, log_prosody_analysis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audio-scan", tags=["Audio Scan"])
initialize_audio_prosody_db()


class AudioScanRequest(BaseModel):
    transcript: List[Dict[str, Any]] = Field(..., description="Timestamped transcript.")
    document_id: str = "api_audio"


class AudioScanResponse(BaseModel):
    document_id: str
    is_synthetic: bool
    pause_variance: float
    speech_rate_variance: float


@router.post("/analyze", response_model=AudioScanResponse)
async def analyze_audio_prosody(request: AudioScanRequest):
    try:
        result = analyze_prosody(request.transcript)
        log_prosody_analysis(
            request.document_id,
            result["is_synthetic"],
            result["cadence"]["pause_variance"],
        )
        return AudioScanResponse(
            document_id=request.document_id,
            is_synthetic=result["is_synthetic"],
            pause_variance=result["cadence"]["pause_variance"],
            speech_rate_variance=result["speech_rate_variance"],
        )
    except Exception as e:
        logger.error("Audio scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
