"""
src/api/endpoints/api_graph_scan.py
-----------------------------------
FastAPI router for API Call Graph and Dependency Chain Analysis.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from src.core.api_call_graph_extractor import extract_python_api_graph
from src.core.dependency_chain_aligner import compute_api_graph_similarity
from src.db.api_graph_db import initialize_api_graph_db, log_api_graph_alignment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api-graph-scan", tags=["API Graph Scan"])
initialize_api_graph_db()


class APIGraphScanRequest(BaseModel):
    code_a: str = Field(..., min_length=10, description="First source code.")
    code_b: str = Field(..., min_length=10, description="Second source code.")
    code_a_id: str = "api_code_a"
    code_b_id: str = "api_code_b"


class APIGraphScanResponse(BaseModel):
    code_a_id: str
    code_b_id: str
    overall_score: float
    is_clone: bool
    node_similarity: float
    sequence_similarity: float


@router.post("/analyze", response_model=APIGraphScanResponse)
async def analyze_api_graphs(request: APIGraphScanRequest):
    try:
        graph_a = extract_python_api_graph(request.code_a)
        graph_b = extract_python_api_graph(request.code_b)

        result = compute_api_graph_similarity(graph_a, graph_b)
        log_api_graph_alignment(
            request.code_a_id,
            request.code_b_id,
            result["overall_score"],
            result["is_clone"],
        )

        return APIGraphScanResponse(
            code_a_id=request.code_a_id,
            code_b_id=request.code_b_id,
            overall_score=result["overall_score"],
            is_clone=result["is_clone"],
            node_similarity=result["node_similarity"],
            sequence_similarity=result["sequence_similarity"],
        )
    except Exception as e:
        logger.error("API graph scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
