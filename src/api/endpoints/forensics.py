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
src/api/endpoints/forensics.py
------------------------------
FastAPI router for Document Provenance and Metadata Forensics.

Provides REST endpoints to submit documents for deep metadata analysis
and provenance risk scoring.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.core.edit_history_analyzer import compute_provenance_risk_score
from src.db.provenance_logs_db import initialize_provenance_db, log_provenance_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forensics", tags=["Forensics"])

initialize_provenance_db()


class ForensicsResponse(BaseModel):
    """Schema for forensics analysis responses."""

    document_id: str
    file_type: str
    risk_score: float
    is_suspicious: bool
    metadata: Dict[str, Any]


@router.post("/analyze", response_model=ForensicsResponse)
async def analyze_provenance(
    file: UploadFile = File(..., description="Document to analyze (PDF or DOCX)."),
    document_id: str = "api_doc",
    estimated_pages: int = Field(
        1, ge=1, description="Estimated page count for velocity analysis."
    ),
):
    """Submit a document for deep metadata and provenance analysis."""
    try:
        file_bytes = await file.read()
        file_type = file.filename.split(".")[-1].lower() if file.filename else "unknown"

        result = compute_provenance_risk_score(file_bytes, file_type, estimated_pages)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Log the analysis
        log_provenance_analysis(
            document_id=document_id,
            file_type=file_type,
            risk_score=result["risk_score"],
            is_suspicious=result["is_suspicious"],
            metadata=result["metadata"],
        )

        return ForensicsResponse(
            document_id=document_id,
            file_type=file_type,
            risk_score=result["risk_score"],
            is_suspicious=result["is_suspicious"],
            metadata=result["metadata"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Forensics analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
