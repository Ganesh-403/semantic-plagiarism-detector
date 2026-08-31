"""
src/api/endpoints/image_scan.py
-------------------------------
FastAPI router for Image Screenshot OCR and Layout Analysis.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import logging
import hashlib

from src.core.image_ocr_extractor import extract_image_ocr
from src.core.screenshot_layout_analyzer import compute_layout_coherence
from src.db.ocr_extractions_db import initialize_ocr_extractions_db, log_ocr_extraction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/image-scan", tags=["Image Scan"])
initialize_ocr_extractions_db()


class ImageScanResponse(BaseModel):
    document_id: str
    image_hash: str
    block_count: int
    layout_coherence: float
    extracted_text_preview: str


@router.post("/analyze", response_model=ImageScanResponse)
async def analyze_image_screenshot(
    file: UploadFile = File(..., description="Image file to process."),
    document_id: str = "api_image",
):
    try:
        image_bytes = await file.read()
        ocr_result = extract_image_ocr(image_bytes)
        layout_result = compute_layout_coherence(ocr_result)

        log_ocr_extraction(
            document_id=document_id,
            image_hash=ocr_result.image_hash,
            block_count=layout_result["block_count"],
            layout_coherence=layout_result["layout_coherence"],
            extracted_text=layout_result["extracted_text"],
        )

        # Return first 200 chars as preview
        preview = layout_result["extracted_text"][:200]

        return ImageScanResponse(
            document_id=document_id,
            image_hash=ocr_result.image_hash,
            block_count=layout_result["block_count"],
            layout_coherence=layout_result["layout_coherence"],
            extracted_text_preview=preview,
        )
    except Exception as e:
        logger.error("Image scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
