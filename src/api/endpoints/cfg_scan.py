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
src/api/endpoints/cfg_scan.py
------------------------------
FastAPI router for Control Flow Graph (CFG) Plagiarism Detection.

Provides REST endpoints to submit code pairs for deep algorithmic
cloning analysis via CFG isomorphism.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.cfg_generator import generate_cfg
from src.core.graph_isomorphism_engine import compare_cfgs
from src.db.cfg_logs_db import initialize_cfg_db, log_cfg_comparison

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cfg-scan", tags=["CFG Scan"])

initialize_cfg_db()


class CFGScanRequest(BaseModel):
    """Schema for CFG scan requests."""

    code_a: str = Field(..., min_length=1, description="First source code (Python).")
    code_b: str = Field(..., min_length=1, description="Second source code (Python).")
    doc_a_id: str = "api_code_a"
    doc_b_id: str = "api_code_b"


class CFGScanResponse(BaseModel):
    """Schema for CFG scan responses."""

    doc_a_id: str
    doc_b_id: str
    edit_distance: int
    structural_similarity: float
    is_exact_clone: bool
    hash_a: str
    hash_b: str


@router.post("/analyze", response_model=CFGScanResponse)
async def analyze_cfg(request: CFGScanRequest):
    """Submit two code files for CFG-based algorithmic plagiarism analysis."""
    try:
        blocks_a = generate_cfg(request.code_a)
        blocks_b = generate_cfg(request.code_b)

        if not blocks_a or not blocks_b:
            raise HTTPException(
                status_code=400,
                detail="Failed to parse one or both code files into CFGs.",
            )

        result = compare_cfgs(blocks_a, blocks_b)

        # Log the comparison
        log_cfg_comparison(
            doc_a_id=request.doc_a_id,
            doc_b_id=request.doc_b_id,
            hash_a=result["hash_a"],
            hash_b=result["hash_b"],
            edit_distance=result["edit_distance"],
            similarity=result["structural_similarity"],
            is_exact_clone=result["is_exact_clone"],
        )

        return CFGScanResponse(
            doc_a_id=request.doc_a_id,
            doc_b_id=request.doc_b_id,
            edit_distance=result["edit_distance"],
            structural_similarity=result["structural_similarity"],
            is_exact_clone=result["is_exact_clone"],
            hash_a=result["hash_a"],
            hash_b=result["hash_b"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("CFG scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
