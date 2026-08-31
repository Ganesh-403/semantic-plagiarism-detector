"""
src/api/endpoints/notebook_scan.py
----------------------------------
FastAPI router for Jupyter Notebook Data Lineage Plagiarism Detection.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from src.core.notebook_graph_extractor import extract_notebook_graph
from src.core.data_lineage_aligner import compute_lineage_similarity
from src.db.notebook_lineage_db import (
    initialize_notebook_lineage_db,
    log_notebook_alignment,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/notebook-scan", tags=["Notebook Scan"])
initialize_notebook_lineage_db()


class NotebookScanRequest(BaseModel):
    nb_a_json: str = Field(
        ..., min_length=10, description="Raw JSON of first notebook."
    )
    nb_b_json: str = Field(
        ..., min_length=10, description="Raw JSON of second notebook."
    )
    nb_a_id: str = "api_nb_a"
    nb_b_id: str = "api_nb_b"


class NotebookScanResponse(BaseModel):
    nb_a_id: str
    nb_b_id: str
    overall_score: float
    is_cloned_workflow: bool
    execution_distance: int
    lineage_similarity: float


@router.post("/analyze", response_model=NotebookScanResponse)
async def analyze_notebook_lineage(request: NotebookScanRequest):
    try:
        graph_a = extract_notebook_graph(request.nb_a_json)
        graph_b = extract_notebook_graph(request.nb_b_json)

        result = compute_lineage_similarity(graph_a, graph_b)
        log_notebook_alignment(
            request.nb_a_id,
            request.nb_b_id,
            result["overall_score"],
            result["is_cloned_workflow"],
        )

        return NotebookScanResponse(
            nb_a_id=request.nb_a_id,
            nb_b_id=request.nb_b_id,
            overall_score=result["overall_score"],
            is_cloned_workflow=result["is_cloned_workflow"],
            execution_distance=result["execution_distance"],
            lineage_similarity=result["lineage_similarity"],
        )
    except Exception as e:
        logger.error("Notebook scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
