"""
src/api/endpoints/table_scan.py
-------------------------------
FastAPI router for Tabular Data and CSV Structural Plagiarism Detection.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging

from src.core.tabular_data_extractor import extract_table_fingerprint
from src.core.table_structure_aligner import compute_table_similarity
from src.db.tabular_plagiarism_db import (
    initialize_tabular_plagiarism_db,
    log_table_alignment,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/table-scan", tags=["Table Scan"])
initialize_tabular_plagiarism_db()


class TableScanRequest(BaseModel):
    csv_a: str = Field(..., min_length=1, description="First CSV content.")
    csv_b: str = Field(..., min_length=1, description="Second CSV content.")
    table_a_id: str = "api_table_a"
    table_b_id: str = "api_table_b"


class TableScanResponse(BaseModel):
    table_a_id: str
    table_b_id: str
    overall_score: float
    is_cloned_dataset: bool
    schema_similarity: float
    distribution_similarity: float


@router.post("/analyze", response_model=TableScanResponse)
async def analyze_tabular_data(request: TableScanRequest):
    try:
        fp_a = extract_table_fingerprint(request.csv_a)
        fp_b = extract_table_fingerprint(request.csv_b)

        result = compute_table_similarity(fp_a, fp_b)
        log_table_alignment(
            request.table_a_id,
            request.table_b_id,
            result["overall_score"],
            result["is_cloned_dataset"],
        )

        return TableScanResponse(
            table_a_id=request.table_a_id,
            table_b_id=request.table_b_id,
            overall_score=result["overall_score"],
            is_cloned_dataset=result["is_cloned_dataset"],
            schema_similarity=result["schema_similarity"],
            distribution_similarity=result["distribution_similarity"],
        )
    except Exception as e:
        logger.error("Table scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
