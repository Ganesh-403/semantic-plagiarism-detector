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
src/api/endpoints/ai_watermark.py
---------------------------------
FastAPI router for AI-Generated Text Watermark Verification.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.ai_watermark_extractor import extract_token_distribution
from src.core.watermark_statistical_test import verify_watermark_presence
from src.db.watermark_verification_db import (
    initialize_watermark_verification_db,
    log_watermark_verification,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai-watermark", tags=["AI Watermark"])

initialize_watermark_verification_db()


class WatermarkRequest(BaseModel):
    """Schema for watermark verification requests."""

    text: str = Field(..., min_length=10, description="Text to analyze.")
    document_id: str = Field("api_doc", description="Document ID for logging.")


class WatermarkResponse(BaseModel):
    """Schema for watermark verification responses."""

    document_id: str
    z_score: float
    p_value: float
    is_watermarked: bool
    observed_ratio: float


@router.post("/verify", response_model=WatermarkResponse)
async def verify_ai_watermark(request: WatermarkRequest):
    """Submit text for AI watermark verification."""
    try:
        token_metrics = extract_token_distribution(request.text)
        result = verify_watermark_presence(token_metrics)

        log_watermark_verification(
            document_id=request.document_id,
            z_score=result["z_score"],
            p_value=result["p_value"],
            is_watermarked=result["is_watermarked"],
        )

        return WatermarkResponse(
            document_id=request.document_id,
            z_score=result["z_score"],
            p_value=result["p_value"],
            is_watermarked=result["is_watermarked"],
            observed_ratio=result["observed_ratio"],
        )
    except Exception as e:
        logger.error("Watermark verification failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
