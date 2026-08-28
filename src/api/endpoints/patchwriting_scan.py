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
src/api/endpoints/patchwriting_scan.py
--------------------------------------
FastAPI router for Mosaic Plagiarism (Patchwriting) Detection.

Provides REST endpoints to submit text pairs for syntactic structural analysis.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.patchwriting_detector import detect_patchwriting
from src.db.patchwriting_logs_db import (
    initialize_patchwriting_db,
    log_patchwriting_detection,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patchwriting", tags=["Patchwriting"])

initialize_patchwriting_db()


class ScanRequest(BaseModel):
    """Schema for patchwriting scan requests."""

    text_a: str = Field(
        ..., min_length=10, description="First text (e.g., student submission)."
    )
    text_b: str = Field(
        ..., min_length=10, description="Second text (e.g., source material)."
    )
    n_gram_size: int = Field(
        3, ge=2, le=5, description="N-gram size for syntactic comparison."
    )
    threshold: float = Field(
        0.60, ge=0.0, le=1.0, description="Similarity threshold for flagging."
    )


class ScanResponse(BaseModel):
    """Schema for patchwriting scan responses."""

    syntactic_jaccard: float
    ngram_overlap: float
    is_patchwriting: bool
    pos_sequence_a: list[str]
    pos_sequence_b: list[str]


@router.post("/scan", response_model=ScanResponse)
async def scan_patchwriting(request: ScanRequest):
    """Analyze two texts for mosaic plagiarism via POS normalization."""
    try:
        result = detect_patchwriting(
            request.text_a,
            request.text_b,
            n=request.n_gram_size,
            threshold=request.threshold,
        )

        # Log the detection (using placeholder IDs for this stateless endpoint)
        log_patchwriting_detection(
            "api_text_a",
            "api_text_b",
            result["syntactic_jaccard"],
            result["ngram_overlap"],
            result["is_patchwriting"],
        )

        return ScanResponse(**result)

    except Exception as e:
        logger.error("Patchwriting scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# semantic-plagiarism-detector/src/api/endpoints/patchwriting_scan.py

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.patchwriting_detector import PatchwritingDetector
from src.db.patchwriting_logs_db import PatchwritingLogsDB

router = APIRouter(prefix="/api/v1/patchwriting", tags=["Mosaic Plagiarism Detection"])
db = PatchwritingLogsDB()


class ScanRequest(BaseModel):
    submission_id: str
    source_id: str
    source_text: str
    student_text: str


@router.post("/scan")
def scan_mosaic_plagiarism(payload: ScanRequest) -> dict[str, Any]:
    """Exposes syntactic POS analysis and mosaic plagiarism detection via REST."""
    try:
        results = PatchwritingDetector.compute_syntactic_similarity(
            payload.source_text, payload.student_text
        )

        # Log detection event
        db.log_structural_clone(
            submission_id=payload.submission_id,
            source_id=payload.source_id,
            similarity_score=results["similarity_score"],
            metrics=results,
        )

        return {
            "status": "success",
            "submission_id": payload.submission_id,
            "analysis": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
