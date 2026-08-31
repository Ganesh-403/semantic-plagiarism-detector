"""
src/api/endpoints/code_comment_scan.py
--------------------------------------
FastAPI router for Code Comment and Docstring Semantic Alignment.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from src.core.docstring_extractor import extract_python_blocks, extract_generic_blocks
from src.core.code_comment_aligner import compute_coherence_score
from src.db.code_comment_db import (
    initialize_code_comment_db,
    log_code_comment_alignment,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/code-comment-scan", tags=["Code Comment Scan"])
initialize_code_comment_db()


class CodeCommentScanRequest(BaseModel):
    source_code: str = Field(..., min_length=10, description="Source code to analyze.")
    language: str = Field(
        "python", description="Programming language ('python', 'java', 'cpp', 'js')."
    )
    document_id: str = "api_code"


class CodeCommentScanResponse(BaseModel):
    document_id: str
    overall_coherence: float
    is_mismatch: bool
    block_count: int
    empty_comment_ratio: float


@router.post("/analyze", response_model=CodeCommentScanResponse)
async def analyze_code_comments(request: CodeCommentScanRequest):
    try:
        if request.language.lower() == "python":
            blocks = extract_python_blocks(request.source_code)
        else:
            # Default to C-style comments for Java, C++, JS
            blocks = extract_generic_blocks(request.source_code)

        result = compute_coherence_score(blocks)
        log_code_comment_alignment(
            request.document_id, result["overall_coherence"], result["is_mismatch"]
        )

        return CodeCommentScanResponse(
            document_id=request.document_id,
            overall_coherence=result["overall_coherence"],
            is_mismatch=result["is_mismatch"],
            block_count=result["block_count"],
            empty_comment_ratio=result["empty_comment_ratio"],
        )
    except Exception as e:
        logger.error("Code comment scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
