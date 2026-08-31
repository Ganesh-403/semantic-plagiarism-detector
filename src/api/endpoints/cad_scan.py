"""
src/api/endpoints/cad_scan.py
-----------------------------
FastAPI router for 3D Mesh and CAD Model Plagiarism Detection.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import logging
from src.core.mesh_geometry_extractor import extract_mesh_descriptor
from src.core.spatial_shape_aligner import compute_cad_similarity
from src.db.cad_plagiarism_db import initialize_cad_plagiarism_db, log_cad_alignment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cad-scan", tags=["CAD Scan"])
initialize_cad_plagiarism_db()


class CADScanRequest(BaseModel):
    stl_a: str = Field(..., description="ASCII STL content of model A.")
    stl_b: str = Field(..., description="ASCII STL content of model B.")
    model_a_id: str = "api_model_a"
    model_b_id: str = "api_model_b"


class CADScanResponse(BaseModel):
    model_a_id: str
    model_b_id: str
    overall_score: float
    is_cloned_geometry: bool
    hausdorff_distance: float


@router.post("/analyze", response_model=CADScanResponse)
async def analyze_cad_models(request: CADScanRequest):
    try:
        desc_a = extract_mesh_descriptor(request.stl_a)
        desc_b = extract_mesh_descriptor(request.stl_b)
        result = compute_cad_similarity(desc_a, desc_b)
        log_cad_alignment(
            request.model_a_id,
            request.model_b_id,
            result["overall_score"],
            result["is_cloned_geometry"],
        )
        return CADScanResponse(
            model_a_id=request.model_a_id,
            model_b_id=request.model_b_id,
            overall_score=result["overall_score"],
            is_cloned_geometry=result["is_cloned_geometry"],
            hausdorff_distance=result["hausdorff_distance"],
        )
    except Exception as e:
        logger.error("CAD scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
