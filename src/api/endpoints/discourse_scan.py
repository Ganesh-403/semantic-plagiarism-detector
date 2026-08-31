"""
src/api/endpoints/discourse_scan.py
-----------------------------------
FastAPI router for Hierarchical Discourse Tree and Rhetorical Structure Analysis.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from src.core.discourse_tree_parser import parse_discourse_tree
from src.core.rhetorical_structure_aligner import compute_rhetorical_alignment
from src.db.discourse_trees_db import (
    initialize_discourse_trees_db,
    log_discourse_alignment,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/discourse-scan", tags=["Discourse Scan"])
initialize_discourse_trees_db()


class DiscourseScanRequest(BaseModel):
    text_a: str = Field(..., min_length=50, description="First document text.")
    text_b: str = Field(..., min_length=50, description="Second document text.")
    doc_a_id: str = "api_doc_a"
    doc_b_id: str = "api_doc_b"


class DiscourseScanResponse(BaseModel):
    doc_a_id: str
    doc_b_id: str
    structural_similarity: float
    is_structural_plagiarism: bool
    edit_distance: int


@router.post("/analyze", response_model=DiscourseScanResponse)
async def analyze_discourse_trees(request: DiscourseScanRequest):
    try:
        tree_a = parse_discourse_tree(request.text_a)
        tree_b = parse_discourse_tree(request.text_b)

        result = compute_rhetorical_alignment(tree_a, tree_b)
        log_discourse_alignment(
            request.doc_a_id,
            request.doc_b_id,
            result["structural_similarity"],
            result["is_structural_plagiarism"],
        )

        return DiscourseScanResponse(
            doc_a_id=request.doc_a_id,
            doc_b_id=request.doc_b_id,
            structural_similarity=result["structural_similarity"],
            is_structural_plagiarism=result["is_structural_plagiarism"],
            edit_distance=result["edit_distance"],
        )
    except Exception as e:
        logger.error("Discourse scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
