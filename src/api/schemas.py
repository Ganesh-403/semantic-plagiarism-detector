"""Pydantic schemas for OpenAPI / Swagger UI response model annotations."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class LoginResponse(BaseModel):
    """Response schema for authentication login."""

    token: str = Field(..., description="Authentication session token")


class HealthCheckResponse(BaseModel):
    """Response schema for application readiness and liveness probes."""

    status: str = Field(..., description="Health status indicator")
    service: str = Field(..., description="Name of the service")
    version: str = Field(..., description="Service version string")


class HealthzResponse(BaseModel):
    """Response schema for health endpoint."""

    status: str = Field(..., description="Overall service status")
    db: str = Field(..., description="Database connectivity status")
    memory: str = Field(..., description="Memory status")
    db_size_bytes: int = Field(default=0, description="Corpus database file size in bytes")
    db_size_mb: float = Field(default=0.0, description="Corpus database file size in megabytes")


class StatusResponse(BaseModel):
    """Response schema for the public service status endpoint."""

    status: str = Field(..., description="Service status indicator")
    version: str = Field(..., description="API version string")
    timestamp: str = Field(..., description="Server UTC timestamp in ISO 8601 format")


class FlaggedChunkMatch(BaseModel):
    """Schema for individual paragraph or text chunk match pairs."""

    uploaded_chunk: str = Field(
        ..., description="Original text chunk from uploaded document"
    )
    matched_chunk: str = Field(
        ..., description="Matched text chunk from target document"
    )
    similarity_score: float = Field(
        ..., description="Cosine similarity score for chunk pair"
    )


class MatchedDocument(BaseModel):
    """Schema for document matching results in plagiarism detection."""

    filename: str = Field(
        ..., description="Filename of matched target document in corpus"
    )
    document_similarity_score: float = Field(
        ..., description="Overall document similarity score"
    )
    max_chunk_similarity_score: float = Field(
        ..., description="Maximum chunk similarity score"
    )
    severity: str = Field(..., description="Flagged severity level (High, Medium)")
    flagged_chunks: List[FlaggedChunkMatch] = Field(
        default_factory=list, description="List of top matching text chunk pairs"
    )


class SimilarityCheckResponse(BaseModel):
    """Response schema for document plagiarism scanning."""

    filename: str = Field(..., description="Filename of uploaded document")
    word_count: int = Field(..., description="Total word count of uploaded document")
    chunk_count: int = Field(..., description="Total text chunk count")
    plagiarism_flagged: bool = Field(
        ..., description="Whether plagiarism was flagged based on threshold"
    )
    threshold_used: float = Field(
        ..., description="Similarity threshold configured for scan"
    )
    overall_document_similarity: float = Field(
        ..., description="Highest overall document similarity score"
    )
    max_chunk_similarity: float = Field(
        ..., description="Highest chunk-level similarity score"
    )
    matched_documents_count: int = Field(
        ..., description="Total count of matched corpus documents"
    )
    matched_documents: List[MatchedDocument] = Field(
        default_factory=list, description="Detailed list of matched corpus documents"
    )


class DocumentUploadResponse(SimilarityCheckResponse):
    """Response schema for document upload and scan operations."""

    pass


class ClearDataResponse(BaseModel):
    """Response schema for bulk clearing administrative operation."""

    status: str = Field(..., description="Operation status (e.g. success)")
    message: str = Field(..., description="Summary message describing clearing action")


class IncidentResponse(ClearDataResponse):
    """Response schema for incident clearing operations."""

    pass


class ErrorResponse(BaseModel):
    """Response schema for API error responses."""

    detail: str = Field(..., description="Detailed error description message")


class AsyncScanJobResponse(BaseModel):
    """Response schema for queuing an asynchronous document scan job."""

    job_id: str = Field(..., description="Unique background scan job identifier")
    status: str = Field(..., description="Initial job status (queued)")
    status_url: str = Field(..., description="Relative endpoint URL to poll for job status")
    message: str = Field(..., description="Status description message")


class AsyncScanStatusResponse(BaseModel):
    """Response schema for checking the status of an asynchronous scan job."""

    job_id: str = Field(..., description="Unique background scan job identifier")
    status: str = Field(..., description="Current job status: queued, processing, completed, or failed")
    filename: str = Field(..., description="Filename of uploaded document being scanned")
    created_at: str = Field(..., description="ISO 8601 UTC timestamp when job was created")
    completed_at: str | None = Field(default=None, description="ISO 8601 UTC timestamp when job finished")
    result: SimilarityCheckResponse | None = Field(default=None, description="Detailed scan results when completed")
    error: str | None = Field(default=None, description="Error message if scan job failed")

