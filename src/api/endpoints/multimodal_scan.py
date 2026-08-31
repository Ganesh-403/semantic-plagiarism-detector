"""
src/api/endpoints/multimodal_scan.py
------------------------------------
FastAPI router for Multimodal Plagiarism Detection.

Handles multimodal submissions and returns visual/math match reports.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

from src.core.image_phash_engine import compute_hamming_distance
from src.core.equation_ast_parser import (
    tokenize_latex,
    normalize_equation_ast,
    compute_tree_edit_distance,
)
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
