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
src/api/endpoints/translation_defense.py
----------------------------------------
FastAPI router for Cross-Lingual Back-Translation Defense.

Provides REST endpoints to analyze text for back-translation obfuscation.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.back_translation_simulator import simulate_back_translation
from src.core.translation_invariance_scorer import score_translation_invariance
from src.db.translation_attack_logs_db import (
    initialize_translation_logs_db,
    log_translation_attack,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/translation-defense", tags=["Translation Defense"])

initialize_translation_logs_db()


class DefenseRequest(BaseModel):
    """Schema for translation defense requests."""

    original_text: str = Field(..., min_length=10, description="Original text.")
    suspect_text: str = Field(
        ..., min_length=10, description="Suspected back-translated text."
    )
    document_id: str = Field("api_doc", description="Document ID for logging.")


class DefenseResponse(BaseModel):
    """Schema for translation defense responses."""

    document_id: str
    lexical_drift: float
    structural_variance: float
    invariance_score: float
    is_obfuscated: bool


@router.post("/analyze", response_model=DefenseResponse)
async def analyze_translation_defense(request: DefenseRequest):
    """Analyze text for back-translation obfuscation."""
    try:
        result = score_translation_invariance(
            request.original_text, request.suspect_text
        )

        log_translation_attack(
            document_id=request.document_id,
            lexical_drift=result["lexical_drift"],
            structural_variance=result["structural_variance"],
            invariance_score=result["invariance_score"],
            is_obfuscated=result["is_obfuscated"],
        )

        return DefenseResponse(
            document_id=request.document_id,
            lexical_drift=result["lexical_drift"],
            structural_variance=result["structural_variance"],
            invariance_score=result["invariance_score"],
            is_obfuscated=result["is_obfuscated"],
        )
    except Exception as e:
        logger.error("Translation defense analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
