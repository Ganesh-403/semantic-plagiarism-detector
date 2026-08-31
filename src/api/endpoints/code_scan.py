"""
src/api/endpoints/code_scan.py
------------------------------
FastAPI router for Code Plagiarism Detection via AST Analysis.

Provides REST endpoints to submit code files and retrieve structural
match reports, detecting plagiarism regardless of variable renaming
or whitespace obfuscation.
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field
from typing import Dict, Any
import logging

from src.core.code_similarity_engine import compare_code_snippets

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/code", tags=["Code Analysis"])


class CodeScanResponse(BaseModel):
    """Schema for code similarity scan responses."""

    file_a: str
    file_b: str
    jaccard_similarity: float = Field(..., ge=0.0, le=1.0)
    levenshtein_similarity: float = Field(..., ge=0.0, le=1.0)
    overall_score: float = Field(..., ge=0.0, le=1.0)


@router.post("/scan", response_model=CodeScanResponse)
async def scan_code_similarity(
    file_a: UploadFile = File(..., description="First Python source code file."),
    file_b: UploadFile = File(..., description="Second Python source code file."),
):
    """Submit two code files for structural plagiarism analysis.

    Parses both files into normalized Abstract Syntax Trees (ASTs) and
    computes structural similarity scores. This detects plagiarism even
    if variables are renamed or comments are added.
    """
    try:
        # Read file contents
        content_a = await file_a.read()
        content_b = await file_b.read()

        code_a = content_a.decode("utf-8")
        code_b = content_b.decode("utf-8")

        # Run comparison
        scores = compare_code_snippets(code_a, code_b)

        return CodeScanResponse(
            file_a=file_a.filename or "file_a.py",
            file_b=file_b.filename or "file_b.py",
            **scores,
        )

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Files must be valid UTF-8 encoded text.",
        )
    except Exception as e:
        logger.error("Code scan failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during code analysis: {str(e)}",
        )
