"""
src/api/endpoints/cross_modal_scan.py
-------------------------------------
FastAPI router for Cross-Modal Semantic Alignment.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

from src.core.pseudocode_parser import extract_logical_blocks
from src.core.cross_modal_aligner import (
    extract_code_logical_blocks,
    compute_cross_modal_similarity,
)
from src.db.cross_modal_logs_db import (
    initialize_cross_modal_logs_db,
    log_cross_modal_alignment,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cross-modal-scan", tags=["Cross-Modal Scan"])
initialize_cross_modal_logs_db()


class CrossModalScanRequest(BaseModel):
    text_description: str = Field(
        ..., min_length=10, description="Natural language algorithmic description."
    )
    source_code: str = Field(
        ..., min_length=10, description="Source code implementation."
    )
    text_doc_id: str = "api_text"
    code_doc_id: str = "api_code"


class CrossModalScanResponse(BaseModel):
    text_doc_id: str
    code_doc_id: str
    overall_score: float
    is_translation: bool
    structural_similarity: float
    semantic_similarity: float


@router.post("/analyze", response_model=CrossModalScanResponse)
async def analyze_cross_modal_plagiarism(request: CrossModalScanRequest):
    try:
        text_blocks = extract_logical_blocks(request.text_description)
        code_blocks = extract_code_logical_blocks(request.source_code)

        result = compute_cross_modal_similarity(text_blocks, code_blocks)
        log_cross_modal_alignment(
            request.text_doc_id,
            request.code_doc_id,
            result["overall_score"],
            result["is_translation"],
        )

        return CrossModalScanResponse(
            text_doc_id=request.text_doc_id,
            code_doc_id=request.code_doc_id,
            overall_score=result["overall_score"],
            is_translation=result["is_translation"],
            structural_similarity=result["structural_similarity"],
            semantic_similarity=result["semantic_similarity"],
        )
    except Exception as e:
        logger.error("Cross-modal scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
