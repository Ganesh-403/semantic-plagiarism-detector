"""
src/api/endpoints/tool_detection.py
-----------------------------------
FastAPI router for Paraphrase Tool Fingerprinting and Attribution.

Provides REST endpoints to analyze text and attribute it to specific
automated paraphrasing tools.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any
import logging

from src.core.paraphrase_fingerprinter import (
    extract_paraphrase_fingerprint,
    attribute_paraphrase_tool
)
from src.db.tool_signatures_db import initialize_signatures_db, log_attribution

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tool-detection", tags=["Tool Detection"])

initialize_signatures_db()


class DetectionRequest(BaseModel):
    """Schema for tool detection requests."""
    text: str = Field(..., min_length=50, description="Text to analyze for tool attribution.")
    document_id: str = Field("api_doc", description="Optional document ID for logging.")


class DetectionResponse(BaseModel):
    """Schema for tool detection responses."""
    attributed_tool: str
    confidence: float
    scores: dict[str, float]
    fingerprint: dict[str, float]


@router.post("/analyze", response_model=DetectionResponse)
async def analyze_tool_usage(request: DetectionRequest):
    """Analyze text to attribute it to a specific paraphrasing tool."""
    try:
        # Extract fingerprint
        fingerprint = extract_paraphrase_fingerprint(request.text)
        
        # Attribute to tool
        attribution = attribute_paraphrase_tool(fingerprint)
        
        # Log the attribution
        log_attribution(
            request.document_id,
            attribution["attributed_tool"],
            attribution["confidence"],
            fingerprint
        )
        
        return DetectionResponse(
            attributed_tool=attribution["attributed_tool"],
            confidence=attribution["confidence"],
            scores=attribution["scores"],
            fingerprint=fingerprint
        )
        
    except Exception as e:
        logger.error("Tool detection failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# semantic-plagiarism-detector/src/api/endpoints/tool_detection.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from src.core.paraphrase_fingerprinter import ParaphraseFingerprinter
from src.db.tool_signatures_db import ToolSignaturesDB

router = APIRouter(prefix="/api/v1/paraphrase-detection", tags=["Paraphrase Tool Fingerprinting"])
db = ToolSignaturesDB()
fingerprinter = ParaphraseFingerprinter()

class TextPayload(BaseModel):
    text: str

@router.post("/detect")
def detect_paraphrase_tool(payload: TextPayload) -> dict[str, Any]:
    """Exposes automated paraphrase tool fingerprinting and attribution via REST."""
    try:
        features = fingerprinter.extract_fingerprint(payload.text)
        match_result = db.match_signature(features)
        
        return {
            "status": "success",
            "extracted_features": features,
            "attribution": match_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
