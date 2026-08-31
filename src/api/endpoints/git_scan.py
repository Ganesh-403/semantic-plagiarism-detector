"""
src/api/endpoints/git_scan.py
-----------------------------
FastAPI router for Git Commit Graph and Covert Collaboration Detection.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging
from src.core.git_graph_extractor import parse_git_log
from src.core.covert_collaboration_analyzer import analyze_covert_collaboration
from src.db.git_forensics_db import initialize_git_forensics_db, log_git_forensics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/git-scan", tags=["Git Scan"])
initialize_git_forensics_db()


class GitScanRequest(BaseModel):
    log_a: str = Field(..., description="Git log dump A.")
    log_b: str = Field(..., description="Git log dump B.")
    log_a_id: str = "api_log_a"
    log_b_id: str = "api_log_b"


class GitScanResponse(BaseModel):
    log_a_id: str
    log_b_id: str
    overall_score: float
    is_covert_collaboration: bool
    timestamp_burstiness: float


@router.post("/analyze", response_model=GitScanResponse)
async def analyze_git_logs(request: GitScanRequest):
    try:
        graph_a = parse_git_log(request.log_a)
        graph_b = parse_git_log(request.log_b)
        result = analyze_covert_collaboration(graph_a, graph_b)
        log_git_forensics(
            request.log_a_id,
            request.log_b_id,
            result["overall_score"],
            result["is_covert_collaboration"],
        )
        return GitScanResponse(
            log_a_id=request.log_a_id,
            log_b_id=request.log_b_id,
            overall_score=result["overall_score"],
            is_covert_collaboration=result["is_covert_collaboration"],
            timestamp_burstiness=result["timestamp_burstiness"],
        )
    except Exception as e:
        logger.error("Git scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
