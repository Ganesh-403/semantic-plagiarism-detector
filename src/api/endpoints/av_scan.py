"""
src/api/endpoints/av_scan.py
----------------------------
FastAPI router for Multimedia Audio-Visual Sync and Dubbing Plagiarism Detection.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging
from src.core.av_sync_extractor import extract_av_features
from src.core.dubbing_fingerprinter import analyze_dubbing
from src.db.multimedia_forensics_db import (
    initialize_multimedia_forensics_db,
    log_av_forensics,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/av-scan", tags=["AV Scan"])
initialize_multimedia_forensics_db()


class AVScanRequest(BaseModel):
    # In a real implementation, these would be UploadFile objects.
    # For this API, we accept base64 or raw byte strings for testing.
    video_a_bytes: str = Field(..., description="Raw bytes of video A (simulated).")
    audio_a_bytes: str = Field(..., description="Raw bytes of audio A.")
    video_b_bytes: str = Field(..., description="Raw bytes of video B.")
    audio_b_bytes: str = Field(..., description="Raw bytes of audio B.")
    media_a_id: str = "api_media_a"
    media_b_id: str = "api_media_b"


class AVScanResponse(BaseModel):
    media_a_id: str
    media_b_id: str
    overall_score: float
    is_dubbed: bool
    pitch_similarity: float
    dubbing_probability: float


@router.post("/analyze", response_model=AVScanResponse)
async def analyze_multimedia(request: AVScanRequest):
    try:
        # Convert string representations to bytes for processing
        vid_a = request.video_a_bytes.encode("utf-8")
        aud_a = request.audio_a_bytes.encode("utf-8")
        vid_b = request.video_b_bytes.encode("utf-8")
        aud_b = request.audio_b_bytes.encode("utf-8")

        feat_a = extract_av_features(vid_a, aud_a)
        feat_b = extract_av_features(vid_b, aud_b)

        result = analyze_dubbing(feat_a, feat_b)
        log_av_forensics(
            request.media_a_id,
            request.media_b_id,
            result["dubbing_probability"],
            result["is_dubbed"],
        )

        return AVScanResponse(
            media_a_id=request.media_a_id,
            media_b_id=request.media_b_id,
            overall_score=result["overall_score"],
            is_dubbed=result["is_dubbed"],
            pitch_similarity=result["pitch_similarity"],
            dubbing_probability=result["dubbing_probability"],
        )
    except Exception as e:
        logger.error("AV scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
