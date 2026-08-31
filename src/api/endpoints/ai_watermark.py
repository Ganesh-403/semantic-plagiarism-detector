"""
src/api/endpoints/ai_watermark.py
---------------------------------
FastAPI router for AI-Generated Text Watermark Extraction and Statistical Verification.

Provides REST endpoints to submit text for token probability distribution analysis,
n-gram frequency extraction, and Maryland-scheme statistical hypothesis verification.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.core.ai_watermark_extractor import AIWatermarkExtractor
from src.core.watermark_statistical_test import WatermarkStatisticalTester
from src.db.watermark_verification_db import (
    get_verification_by_id,
    get_verification_count,
    get_verifications_for_document,
    initialize_watermark_verification_db,
    list_recent_verifications,
    save_verification_result,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai-watermark", tags=["AI Watermark Verification"])

# Ensure DB schema is initialized
initialize_watermark_verification_db()


class WatermarkVerifyRequest(BaseModel):
    """Schema for AI text watermark verification requests."""

    text: str = Field(
        ..., min_length=1, description="Text content to verify for AI watermark."
    )
    document_id: str = Field(
        "api_doc", description="Optional document or submission ID."
    )
    secret_key: str = Field(
        "default_maryland_key",
        description="Secret key or seed for pseudo-random green list partitioning.",
    )
    gamma: float = Field(
        0.5, gt=0.0, lt=1.0, description="Expected green list proportion."
    )
    z_threshold: float = Field(
        4.0, ge=0.0, description="Z-score threshold for positive watermark detection."
    )
    significance_alpha: float = Field(
        0.01, gt=0.0, lt=1.0, description="Significance level alpha threshold."
    )
    context_window_size: int = Field(
        1, ge=0, le=10, description="Preceding context tokens count (k-gram context)."
    )
    confidence_level: float = Field(
        0.95, gt=0.0, lt=1.0, description="Confidence level for proportion interval."
    )
    include_ngrams: bool = Field(
        False, description="Whether to include n-gram frequency breakdown."
    )
    include_token_details: bool = Field(
        False, description="Whether to include per-token green/red classification."
    )


# Alias for backward compatibility
WatermarkRequest = WatermarkVerifyRequest


class ConfidenceIntervalResponse(BaseModel):
    """Schema for confidence interval output."""

    confidence_level: float
    lower_bound: float
    upper_bound: float
    point_estimate: float


class WatermarkVerifyResponse(BaseModel):
    """Schema for AI watermark verification responses."""

    verification_id: str
    document_id: str
    total_tokens: int
    green_tokens: int
    red_tokens: int
    observed_green_ratio: float
    expected_green_ratio: float
    z_score: float
    p_value: float
    exact_p_value: float
    asymptotic_p_value: float
    confidence_interval: ConfidenceIntervalResponse
    is_watermarked: bool
    confidence_score: float
    watermark_scheme: str
    token_entropy: float
    ngram_frequencies: Optional[dict[int, dict[str, int]]] = None
    token_details: Optional[list[dict[str, Any]]] = None
    created_at: str
    observed_ratio: Optional[float] = None


# Alias for backward compatibility
WatermarkResponse = WatermarkVerifyResponse


@router.post("/verify", response_model=WatermarkVerifyResponse, status_code=status.HTTP_200_OK)
async def verify_ai_watermark(request: WatermarkVerifyRequest):
    """Submit text for statistical AI watermark extraction and hypothesis verification."""
    try:
        extractor = AIWatermarkExtractor(
            secret_key=request.secret_key,
            gamma=request.gamma,
            context_window_size=request.context_window_size,
        )

        features = extractor.extract_features(
            text=request.text,
            include_token_details=request.include_token_details,
        )

        tester = WatermarkStatisticalTester(
            gamma=request.gamma,
            z_threshold=request.z_threshold,
            significance_alpha=request.significance_alpha,
            confidence_level=request.confidence_level,
        )

        test_result = tester.test(
            green_tokens=features.green_token_count,
            total_tokens=features.total_scored_tokens,
        )

        metadata = {
            "secret_key": request.secret_key,
            "context_window_size": request.context_window_size,
            "token_entropy": features.token_distribution.entropy if features.token_distribution else 0.0,
            "unique_tokens": features.token_distribution.unique_tokens if features.token_distribution else 0,
        }

        saved_record = save_verification_result(
            document_id=request.document_id,
            total_tokens=test_result.total_tokens,
            green_tokens=test_result.green_tokens,
            red_tokens=test_result.red_tokens,
            observed_green_ratio=test_result.observed_green_ratio,
            expected_green_ratio=test_result.expected_green_ratio,
            z_score=test_result.z_score,
            p_value=test_result.p_value,
            confidence_level=test_result.confidence_interval.confidence_level,
            ci_lower=test_result.confidence_interval.lower_bound,
            ci_upper=test_result.confidence_interval.upper_bound,
            confidence_score=test_result.confidence_score,
            is_watermarked=test_result.is_watermarked,
            watermark_scheme="Maryland-Kirchenbauer",
            metadata=metadata,
        )

        v_id = saved_record["verification_id"] if saved_record else "WMV-TEMP"
        created_at = saved_record["created_at"] if saved_record else datetime.now(timezone.utc).isoformat()

        ci_resp = ConfidenceIntervalResponse(
            confidence_level=test_result.confidence_interval.confidence_level,
            lower_bound=test_result.confidence_interval.lower_bound,
            upper_bound=test_result.confidence_interval.upper_bound,
            point_estimate=test_result.confidence_interval.point_estimate,
        )

        return WatermarkVerifyResponse(
            verification_id=v_id,
            document_id=request.document_id,
            total_tokens=test_result.total_tokens,
            green_tokens=test_result.green_tokens,
            red_tokens=test_result.red_tokens,
            observed_green_ratio=test_result.observed_green_ratio,
            expected_green_ratio=test_result.expected_green_ratio,
            z_score=test_result.z_score,
            p_value=test_result.p_value,
            exact_p_value=test_result.exact_p_value,
            asymptotic_p_value=test_result.asymptotic_p_value,
            confidence_interval=ci_resp,
            is_watermarked=test_result.is_watermarked,
            confidence_score=test_result.confidence_score,
            watermark_scheme="Maryland-Kirchenbauer",
            token_entropy=features.token_distribution.entropy if features.token_distribution else 0.0,
            ngram_frequencies=features.ngram_frequencies if request.include_ngrams else None,
            token_details=features.token_details if request.include_token_details else None,
            created_at=created_at,
            observed_ratio=test_result.observed_green_ratio,
        )

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error("Watermark verification failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during watermark verification: {str(e)}",
        )


@router.get("/verifications/{verification_id}", response_model=dict[str, Any])
async def get_verification(verification_id: str):
    """Retrieve an existing watermark verification by verification ID."""
    record = get_verification_by_id(verification_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification ID '{verification_id}' not found.",
        )
    return record


@router.get("/document/{document_id}", response_model=list[dict[str, Any]])
async def get_document_verifications(
    document_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Retrieve all watermark verification records for a specific document ID."""
    return get_verifications_for_document(document_id, limit=limit, offset=offset)


@router.get("/stats", response_model=dict[str, Any])
async def get_stats():
    """Retrieve statistical watermark verification engine metrics and totals."""
    total = get_verification_count()
    recent = list_recent_verifications(limit=5)
    return {
        "total_verifications": total,
        "recent_sample_count": len(recent),
        "status": "active",
        "scheme": "Maryland-Kirchenbauer",
    }
