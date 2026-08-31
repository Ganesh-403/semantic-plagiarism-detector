"""
src/api/endpoints/math_scan.py
------------------------------
FastAPI router for Mathematical Equation Structural Plagiarism Detection.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from src.core.equation_ast_extractor import extract_equation_ast
from src.core.math_structure_aligner import compute_math_similarity
from src.db.math_plagiarism_db import initialize_math_plagiarism_db, log_math_alignment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/math-scan", tags=["Math Scan"])
initialize_math_plagiarism_db()


class MathScanRequest(BaseModel):
    latex_a: str = Field(..., min_length=1, description="First LaTeX equation.")
    latex_b: str = Field(..., min_length=1, description="Second LaTeX equation.")
    eq_a_id: str = "api_eq_a"
    eq_b_id: str = "api_eq_b"


class MathScanResponse(BaseModel):
    eq_a_id: str
    eq_b_id: str
    structural_similarity: float
    is_structural_plagiarism: bool
    edit_distance: int


@router.post("/analyze", response_model=MathScanResponse)
async def analyze_math_equations(request: MathScanRequest):
    try:
        tree_a = extract_equation_ast(request.latex_a)
        tree_b = extract_equation_ast(request.latex_b)

        result = compute_math_similarity(tree_a, tree_b)
        log_math_alignment(
            request.eq_a_id,
            request.eq_b_id,
            result["structural_similarity"],
            result["is_structural_plagiarism"],
        )

        return MathScanResponse(
            eq_a_id=request.eq_a_id,
            eq_b_id=request.eq_b_id,
            structural_similarity=result["structural_similarity"],
            is_structural_plagiarism=result["is_structural_plagiarism"],
            edit_distance=result["edit_distance"],
        )
    except Exception as e:
        logger.error("Math scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
