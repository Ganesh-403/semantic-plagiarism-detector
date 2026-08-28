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
src/api/endpoints/multimodal_scan.py
------------------------------------
FastAPI router for Multimodal Plagiarism Detection.

Handles multimodal submissions and returns visual/math match reports.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.core.equation_ast_parser import (
    compute_tree_edit_distance,
    normalize_equation_ast,
    tokenize_latex,
)
from src.core.image_phash_engine import compute_hamming_distance
from src.db.multimodal_corpus_db import initialize_multimodal_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multimodal", tags=["Multimodal"])

initialize_multimodal_db()


class EquationRequest(BaseModel):
    """Schema for equation comparison."""

    latex_a: str
    latex_b: str


class EquationResponse(BaseModel):
    """Schema for equation comparison response."""

    edit_distance: int
    similarity_score: float


@router.post("/compare-equations", response_model=EquationResponse)
async def compare_equations(request: EquationRequest):
    """Compare two LaTeX equations for structural plagiarism."""
    tokens_a = tokenize_latex(request.latex_a)
    tokens_b = tokenize_latex(request.latex_b)

    norm_a = normalize_equation_ast(tokens_a)
    norm_b = normalize_equation_ast(tokens_b)

    distance = compute_tree_edit_distance(norm_a, norm_b)
    max_len = max(len(norm_a), len(norm_b), 1)
    similarity = 1.0 - (distance / max_len)

    return EquationResponse(edit_distance=distance, similarity_score=similarity)
