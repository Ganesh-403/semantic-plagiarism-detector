"""
src/api/endpoints/pptx_scan.py
------------------------------
FastAPI router for Presentation Slide (PPTX) Plagiarism Detection.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

from src.core.pptx_slide_extractor import extract_presentation_deck
from src.core.slide_sequence_aligner import compute_deck_similarity
from src.db.presentation_plagiarism_db import (
    initialize_presentation_plagiarism_db,
    log_presentation_alignment,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pptx-scan", tags=["PPTX Scan"])
initialize_presentation_plagiarism_db()


class PPTXScanResponse(BaseModel):
    deck_a_id: str
    deck_b_id: str
    overall_score: float
    is_cloned_deck: bool
    text_similarity: float
    layout_similarity: float


@router.post("/analyze", response_model=PPTXScanResponse)
async def analyze_presentation_slides(
    file_a: UploadFile = File(..., description="First PPTX file."),
    file_b: UploadFile = File(..., description="Second PPTX file."),
    deck_a_id: str = "api_deck_a",
    deck_b_id: str = "api_deck_b",
):
    try:
        bytes_a = await file_a.read()
        bytes_b = await file_b.read()

        deck_a = extract_presentation_deck(bytes_a)
        deck_b = extract_presentation_deck(bytes_b)

        result = compute_deck_similarity(deck_a, deck_b)
        log_presentation_alignment(
            deck_a_id, deck_b_id, result["overall_score"], result["is_cloned_deck"]
        )

        return PPTXScanResponse(
            deck_a_id=deck_a_id,
            deck_b_id=deck_b_id,
            overall_score=result["overall_score"],
            is_cloned_deck=result["is_cloned_deck"],
            text_similarity=result["text_similarity"],
            layout_similarity=result["layout_similarity"],
        )
    except Exception as e:
        logger.error("PPTX scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
