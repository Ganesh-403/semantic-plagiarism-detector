"""
src/api/endpoints/citation_context.py
-------------------------------------
FastAPI router for Citation Context and Semantic Alignment Analysis.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

from src.core.citation_context_analyzer import (
    extract_citation_contexts,
    map_citations_to_references,
)
from src.core.semantic_citation_aligner import analyze_citation_alignment
from src.db.citation_context_db import (
    initialize_citation_context_db,
    log_citation_alignment,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/citation-context", tags=["Citation Context"])

initialize_citation_context_db()


class ContextRequest(BaseModel):
    """Schema for citation context requests."""

    text: str = Field(..., min_length=10, description="Document text.")
    references: Dict[str, str] = Field(
        ..., description="Mapping of citation IDs to abstracts."
    )
    document_id: str = Field("api_doc", description="Document ID for logging.")


class ContextResponse(BaseModel):
    """Schema for citation context responses."""

    document_id: str
    contexts: List[Dict[str, Any]]


@router.post("/analyze", response_model=ContextResponse)
async def analyze_citation_context(request: ContextRequest):
    """Submit a document for citation context analysis."""
    try:
        contexts = extract_citation_contexts(request.text)
        mapped = map_citations_to_references(contexts, request.references)
        results = analyze_citation_alignment(mapped)

        # Log the results
        for ctx in results:
            log_citation_alignment(
                document_id=request.document_id,
                citation_id=ctx["citation_id"],
                alignment_score=ctx["alignment_score"],
                is_bluffing=ctx["is_bluffing"],
            )

        return ContextResponse(document_id=request.document_id, contexts=results)
    except Exception as e:
        logger.error("Citation context analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
