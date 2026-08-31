"""
src/api/endpoints/template_scan.py
----------------------------------
FastAPI router for Document Formatting Entropy and Template Fingerprinting.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging

from src.core.formatting_entropy_extractor import (
    extract_docx_styles,
    extract_latex_macros,
)
from src.core.template_fingerprinter import (
    generate_template_fingerprint,
    compare_template_fingerprints,
)
from src.db.template_fingerprints_db import (
    initialize_template_fingerprints_db,
    log_template_comparison,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/template-scan", tags=["Template Scan"])
initialize_template_fingerprints_db()


class TemplateScanRequest(BaseModel):
    content_a: str = Field(
        ..., description="Content of document A (LaTeX source or base64 DOCX)."
    )
    content_b: str = Field(..., description="Content of document B.")
    file_type: str = Field("latex", description="File type ('latex' or 'docx').")
    doc_a_id: str = "api_doc_a"
    doc_b_id: str = "api_doc_b"


class TemplateScanResponse(BaseModel):
    doc_a_id: str
    doc_b_id: str
    is_template_plagiarism: bool
    is_exact_match: bool
    entropy_delta: float


@router.post("/analyze", response_model=TemplateScanResponse)
async def analyze_templates(request: TemplateScanRequest):
    try:
        if request.file_type.lower() == "docx":
            # In a real implementation, content would be base64 decoded bytes
            # For this API, we assume LaTeX text for simplicity unless bytes are passed
            styles_a = extract_docx_styles(request.content_a.encode("utf-8"))
            styles_b = extract_docx_styles(request.content_b.encode("utf-8"))
        else:
            styles_a = extract_latex_macros(request.content_a)
            styles_b = extract_latex_macros(request.content_b)

        fp_a = generate_template_fingerprint(styles_a, request.file_type)
        fp_b = generate_template_fingerprint(styles_b, request.file_type)

        result = compare_template_fingerprints(fp_a, fp_b)
        log_template_comparison(
            request.doc_a_id,
            request.doc_b_id,
            result["is_template_plagiarism"],
            result["entropy_delta"],
        )

        return TemplateScanResponse(
            doc_a_id=request.doc_a_id,
            doc_b_id=request.doc_b_id,
            is_template_plagiarism=result["is_template_plagiarism"],
            is_exact_match=result["is_exact_match"],
            entropy_delta=result["entropy_delta"],
        )
    except Exception as e:
        logger.error("Template scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
