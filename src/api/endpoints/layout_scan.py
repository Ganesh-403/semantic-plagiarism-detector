"""
src/api/endpoints/layout_scan.py
--------------------------------
FastAPI router for Document Structural Layout Plagiarism Detection.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any
import logging

from src.core.layout_tree_extractor import parse_html_layout, parse_markdown_layout
from src.core.formatting_similarity_engine import compute_layout_similarity
from src.db.layout_logs_db import initialize_layout_logs_db, log_layout_comparison

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/layout-scan", tags=["Layout Scan"])

initialize_layout_logs_db()


class LayoutScanRequest(BaseModel):
    """Schema for layout scan requests."""

    content_a: str = Field(
        ..., description="First document content (HTML or Markdown)."
    )
    content_b: str = Field(..., description="Second document content.")
    format: str = Field("html", description="Content format ('html' or 'markdown').")
    doc_a_id: str = "api_doc_a"
    doc_b_id: str = "api_doc_b"


class LayoutScanResponse(BaseModel):
    """Schema for layout scan responses."""

    doc_a_id: str
    doc_b_id: str
    edit_distance: int
    structural_similarity: float
    is_structural_clone: bool


@router.post("/analyze", response_model=LayoutScanResponse)
async def analyze_layout(request: LayoutScanRequest):
    """Submit documents for structural layout analysis."""
    try:
        if request.format.lower() == "markdown":
            tree_a = parse_markdown_layout(request.content_a)
            tree_b = parse_markdown_layout(request.content_b)
        else:
            tree_a = parse_html_layout(request.content_a)
            tree_b = parse_html_layout(request.content_b)

        result = compute_layout_similarity(tree_a, tree_b)

        log_layout_comparison(
            doc_a_id=request.doc_a_id,
            doc_b_id=request.doc_b_id,
            edit_distance=result["edit_distance"],
            similarity=result["structural_similarity"],
            is_clone=result["is_structural_clone"],
        )

        return LayoutScanResponse(
            doc_a_id=request.doc_a_id,
            doc_b_id=request.doc_b_id,
            edit_distance=result["edit_distance"],
            structural_similarity=result["structural_similarity"],
            is_structural_clone=result["is_structural_clone"],
        )
    except Exception as e:
        logger.error("Layout scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
