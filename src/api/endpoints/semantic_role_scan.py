"""
src/api/endpoints/semantic_role_scan.py
---------------------------------------
FastAPI router for Semantic Role Labeling and Argument Structure Analysis.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from src.core.semantic_role_extractor import extract_document_semantic_roles
from src.core.argument_structure_aligner import compute_role_sequence_similarity
from src.db.semantic_roles_db import (
    initialize_semantic_roles_db,
    log_semantic_role_alignment,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/semantic-role-scan", tags=["Semantic Role Scan"])
initialize_semantic_roles_db()


class SemanticRoleScanRequest(BaseModel):
    text_a: str = Field(..., min_length=10)
    text_b: str = Field(..., min_length=10)
    doc_a_id: str = "api_doc_a"
    doc_b_id: str = "api_doc_b"


class SemanticRoleScanResponse(BaseModel):
    doc_a_id: str
    doc_b_id: str
    structural_similarity: float
    lexical_similarity: float
    is_deep_paraphrase: bool


@router.post("/analyze", response_model=SemanticRoleScanResponse)
async def analyze_semantic_roles(request: SemanticRoleScanRequest):
    try:
        triples_a = extract_document_semantic_roles(request.text_a)
        triples_b = extract_document_semantic_roles(request.text_b)

        result = compute_role_sequence_similarity(triples_a, triples_b)
        log_semantic_role_alignment(
            request.doc_a_id,
            request.doc_b_id,
            result["structural_similarity"],
            result["is_deep_paraphrase"],
        )

        return SemanticRoleScanResponse(
            doc_a_id=request.doc_a_id,
            doc_b_id=request.doc_b_id,
            structural_similarity=result["structural_similarity"],
            lexical_similarity=result["lexical_similarity"],
            is_deep_paraphrase=result["is_deep_paraphrase"],
        )
    except Exception as e:
        logger.error("Semantic role scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
