"""
src/api/endpoints/concept_scan.py
---------------------------------
FastAPI router for Knowledge Graph Conceptual Plagiarism Detection.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import logging

from src.core.knowledge_graph_extractor import (
    extract_spo_triples,
    build_knowledge_graph,
)
from src.core.graph_alignment_scorer import compute_conceptual_overlap
from src.db.concept_graphs_db import initialize_concept_graphs_db, log_concept_alignment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/concept-scan", tags=["Concept Scan"])
initialize_concept_graphs_db()


class ConceptScanRequest(BaseModel):
    text_a: str = Field(..., min_length=10)
    text_b: str = Field(..., min_length=10)
    doc_a_id: str = "api_doc_a"
    doc_b_id: str = "api_doc_b"


class ConceptScanResponse(BaseModel):
    doc_a_id: str
    doc_b_id: str
    conceptual_score: float
    is_conceptual_plagiarism: bool
    shared_edges: int


@router.post("/analyze", response_model=ConceptScanResponse)
async def analyze_concept_plagiarism(request: ConceptScanRequest):
    try:
        triples_a = extract_spo_triples(request.text_a)
        triples_b = extract_spo_triples(request.text_b)
        graph_a = build_knowledge_graph(triples_a)
        graph_b = build_knowledge_graph(triples_b)

        result = compute_conceptual_overlap(graph_a, graph_b)
        log_concept_alignment(
            request.doc_a_id,
            request.doc_b_id,
            result["conceptual_score"],
            result["is_conceptual_plagiarism"],
        )

        return ConceptScanResponse(
            doc_a_id=request.doc_a_id,
            doc_b_id=request.doc_b_id,
            conceptual_score=result["conceptual_score"],
            is_conceptual_plagiarism=result["is_conceptual_plagiarism"],
            shared_edges=result["shared_edges"],
        )
    except Exception as e:
        logger.error("Concept scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
