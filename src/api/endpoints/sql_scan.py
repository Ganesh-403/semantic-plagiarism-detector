"""
src/api/endpoints/sql_scan.py
-----------------------------
FastAPI router for SQL Query Execution Plan and Schema Dependency Plagiarism Detection.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging
from src.core.sql_ast_extractor import extract_sql_ast
from src.core.query_plan_aligner import compute_sql_similarity
from src.db.sql_plagiarism_db import initialize_sql_plagiarism_db, log_sql_alignment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sql-scan", tags=["SQL Scan"])
initialize_sql_plagiarism_db()


class SQLScanRequest(BaseModel):
    query_a: str = Field(..., min_length=5, description="First SQL query.")
    query_b: str = Field(..., min_length=5, description="Second SQL query.")
    query_a_id: str = "api_query_a"
    query_b_id: str = "api_query_b"


class SQLScanResponse(BaseModel):
    query_a_id: str
    query_b_id: str
    overall_score: float
    is_cloned_logic: bool
    ast_similarity: float
    schema_similarity: float


@router.post("/analyze", response_model=SQLScanResponse)
async def analyze_sql_queries(request: SQLScanRequest):
    try:
        ast_a = extract_sql_ast(request.query_a)
        ast_b = extract_sql_ast(request.query_b)
        result = compute_sql_similarity(ast_a, ast_b)
        log_sql_alignment(
            request.query_a_id,
            request.query_b_id,
            result["overall_score"],
            result["is_cloned_logic"],
        )
        return SQLScanResponse(
            query_a_id=request.query_a_id,
            query_b_id=request.query_b_id,
            overall_score=result["overall_score"],
            is_cloned_logic=result["is_cloned_logic"],
            ast_similarity=result["ast_similarity"],
            schema_similarity=result["schema_similarity"],
        )
    except Exception as e:
        logger.error("SQL scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
