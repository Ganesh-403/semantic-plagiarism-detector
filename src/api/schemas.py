# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Pydantic schemas for OpenAPI / Swagger UI response model annotations."""

from __future__ import annotations

from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Generic Type Variable for Pagination
# ============================================================================

T = TypeVar("T")


# ============================================================================
# Authentication Schemas
# ============================================================================


class LoginResponse(BaseModel):
    """Response schema for authentication login."""

    token: str = Field(..., description="Authentication session token")


class RefreshRequest(BaseModel):
    """Request schema for token refresh."""

    refresh_token: str | None = Field(
        default=None, description="Valid OAuth2/JWT refresh token"
    )


class TokenResponse(BaseModel):
    """Response schema for OAuth2 Bearer token generation/refresh."""

    access_token: str = Field(
        ..., description="Newly issued OAuth2 Bearer access token"
    )
    token_type: str = Field(default="bearer", description="Token type (bearer)")
    expires_in: int = Field(
        default=3600, description="Token expiration lifetime in seconds"
    )


class RevokeRequest(BaseModel):
    """Request schema for token revocation."""

    token: str | None = Field(default=None, description="API Bearer token to revoke")


class RevokeResponse(BaseModel):
    """Response schema for token revocation."""

    status: str = Field(..., description="Revocation status indicator")
    message: str = Field(
        ..., description="Summary message describing revocation result"
    )


class TwoFactorSetupRequest(BaseModel):
    """Request schema for 2FA setup."""

    username: str | None = Field(default=None, description="Username to set up 2FA for")
    issuer: str | None = Field(
        default="SemanticPlagiarismDetector",
        description="TOTP Issuer name for authenticator apps",
    )


class TwoFactorSetupResponse(BaseModel):
    """Response schema for 2FA setup including TOTP secret, otpauth URL, and base64 PNG QR code data URI."""

    secret: str = Field(..., description="Base32 TOTP secret key")
    otpauth_url: str = Field(
        ..., description="otpauth:// TOTP URI for authenticator apps"
    )
    qr_code_data_uri: str = Field(
        ...,
        description="Base64-encoded PNG QR code data URI (data:image/png;base64,...)",
    )
    message: str = Field(
        default="2FA setup initialized successfully.", description="Status message"
    )


class TwoFactorDisableRequest(BaseModel):
    """Request schema for 2FA disable."""

    username: str | None = Field(
        default=None, description="Username to disable 2FA for"
    )
    password: str = Field(..., description="Current account password")
    otp_code: str = Field(..., description="Valid 2FA TOTP token")


class TwoFactorDisableResponse(BaseModel):
    """Response schema for 2FA disable."""

    message: str = Field(
        default="2FA has been successfully disabled.", description="Status message"
    )


# ============================================================================
# Health Check Schemas
# ============================================================================


class HealthCheckResponse(BaseModel):
    """Response schema for application readiness and liveness probes."""

    status: str = Field(..., description="Health status indicator")
    service: str = Field(..., description="Name of the service")
    version: str = Field(..., description="Service version string")


class HealthLiveResponse(BaseModel):
    """Response schema for Kubernetes liveness probe endpoint."""

    status: str = Field(
        default="alive", description="Liveness status indicator ('alive')"
    )
    service: str = Field(
        default="Semantic Plagiarism Detector API", description="Name of the service"
    )
    version: str = Field(..., description="Service version string")


class HealthReadyResponse(BaseModel):
    """Response schema for Kubernetes readiness probe endpoint."""

    status: str = Field(
        ..., description="Readiness status indicator ('ready' or 'not_ready')"
    )
    db: str = Field(
        ..., description="Database connectivity status ('connected' or 'disconnected')"
    )
    redis: str = Field(
        ..., description="Redis connectivity status ('connected' or 'disconnected')"
    )
    timestamp: str = Field(..., description="Server UTC timestamp in ISO 8601 format")


class HealthzResponse(BaseModel):
    """Response schema for health endpoint."""

    status: str = Field(..., description="Overall service status")
    db: str = Field(..., description="Database connectivity status")
    memory: str = Field(..., description="Memory status")
    disk: str = Field(default="ok", description="Disk status")
    db_size_bytes: int = Field(
        default=0, description="Corpus database file size in bytes"
    )
    db_size_mb: float = Field(
        default=0.0, description="Corpus database file size in megabytes"
    )
    warning: str | None = Field(
        default=None, description="Descriptive warning when service is degraded"
    )


class StatusResponse(BaseModel):
    """Response schema for the public service status endpoint."""

    status: str = Field(..., description="Service status indicator")
    version: str = Field(..., description="API version string")
    timestamp: str = Field(..., description="Server UTC timestamp in ISO 8601 format")


# ============================================================================
# Pagination Schemas
# ============================================================================


class PaginationParams(BaseModel):
    """
    Request schema for pagination query parameters.

    Used to parse pagination parameters from API request query strings.
    """

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    per_page: int = Field(
        default=20, ge=1, le=100, description="Number of items per page (max 100)"
    )

    model_config = ConfigDict(
        json_schema_extra={"example": {"page": 1, "per_page": 20}}
    )

    def get_offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.per_page

    def get_limit(self) -> int:
        """Get limit for database queries."""
        return self.per_page


class PaginationMeta(BaseModel):
    """
    Pagination metadata model without items.

    Useful for responses that need to return pagination info separately
    or for embedding in larger response structures.
    """

    page: int = Field(..., ge=1, description="Current page number (1-indexed)")
    per_page: int = Field(..., ge=1, description="Number of items per page")
    total_items: int = Field(
        ..., ge=0, description="Total number of items across all pages"
    )
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(False, description="Whether there is a next page")
    has_previous: bool = Field(False, description="Whether there is a previous page")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page": 1,
                "per_page": 20,
                "total_items": 100,
                "total_pages": 5,
                "has_next": True,
                "has_previous": False,
            }
        }
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Generic paginated response model for API endpoints.

    This is the main model for paginated responses, providing a consistent
    structure across all API endpoints that return paginated data.
    Mirrors the PaginationPage class structure for standardized OpenAPI generation.

    Type Parameters:
        T: The type of items in the response

    Attributes:
        items: List of items on the current page
        page: Current page number (1-indexed)
        per_page: Number of items per page
        total_items: Total number of items across all pages
        total_pages: Total number of pages
        has_next: Whether there is a next page
        has_previous: Whether there is a previous page
    """

    items: list[T] = Field(..., description="List of items on the current page")
    page: int = Field(..., ge=1, description="Current page number (1-indexed)")
    per_page: int = Field(..., ge=1, description="Number of items per page")
    total_items: int = Field(
        ..., ge=0, description="Total number of items across all pages"
    )
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(False, description="Whether there is a next page")
    has_previous: bool = Field(False, description="Whether there is a previous page")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [{"id": 1, "name": "Example Item"}],
                "page": 1,
                "per_page": 20,
                "total_items": 100,
                "total_pages": 5,
                "has_next": True,
                "has_previous": False,
            }
        }
    )

    @classmethod
    def from_pagination_data(
        cls, items: list[T], page: int, per_page: int, total_items: int
    ) -> PaginatedResponse[T]:
        """
        Create a PaginatedResponse from pagination data.

        This is a convenience method for constructing responses from raw data.

        Args:
            items: List of items on the current page
            page: Current page number (1-indexed)
            per_page: Number of items per page
            total_items: Total number of items across all pages

        Returns:
            PaginatedResponse instance
        """
        total_pages = (total_items + per_page - 1) // per_page if total_items > 0 else 0

        return cls(
            items=items,
            page=page,
            per_page=per_page,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )

    def to_pagination_meta(self) -> PaginationMeta:
        """Convert to PaginationMeta model."""
        return PaginationMeta(
            page=self.page,
            per_page=self.per_page,
            total_items=self.total_items,
            total_pages=self.total_pages,
            has_next=self.has_next,
            has_previous=self.has_previous,
        )


class CursorPaginationParams(BaseModel):
    """
    Request schema for cursor-based pagination query parameters.

    Useful for infinite scrolling and real-time feeds.
    """

    cursor: str | None = Field(
        default=None, description="Cursor for pagination (opaque string)"
    )
    page_size: int = Field(
        default=20, ge=1, le=100, description="Number of items per page (max 100)"
    )
    direction: str = Field(
        default="next", description="Pagination direction: 'next' or 'prev'"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cursor": "cursor_xyz_123",
                "page_size": 20,
                "direction": "next",
            }
        }
    )


class CursorPaginatedResponse(BaseModel, Generic[T]):
    """
    Cursor-based pagination response model.

    Useful for infinite scrolling and real-time feeds where offset-based
    pagination is not ideal.
    """

    items: list[T] = Field(..., description="List of items on the current page")
    next_cursor: str | None = Field(
        default=None, description="Cursor for the next page (null if no more items)"
    )
    prev_cursor: str | None = Field(
        default=None, description="Cursor for the previous page (null if at start)"
    )
    has_more: bool = Field(False, description="Whether there are more items")
    page_size: int = Field(..., ge=1, description="Number of items per page")
    total_items: int | None = Field(
        default=None, ge=0, description="Total items (if known, optional)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [{"id": "abc_123", "name": "Item 1"}],
                "next_cursor": "cursor_xyz_456",
                "prev_cursor": "cursor_abc_789",
                "has_more": True,
                "page_size": 20,
                "total_items": 100,
            }
        }
    )


# ============================================================================
# Plagiarism Detection Schemas
# ============================================================================


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
    flagged_chunks: list[FlaggedChunkMatch] = Field(
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
    plagiarism_density: int = Field(
        ..., description="Percentage of document chunks flagged as plagiarized"
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
    matched_documents: list[MatchedDocument] = Field(
        default_factory=list, description="Detailed list of matched corpus documents"
    )


class DocumentUploadResponse(SimilarityCheckResponse):
    """Response schema for document upload and scan operations."""

    pass


class ScanTextRequest(BaseModel):
    """Request payload schema for raw text submission plagiarism scanning."""

    text: str = Field(
        ...,
        min_length=1,
        description="Raw text submission content to scan for plagiarism",
    )
    filename: str = Field(
        default="submission.txt",
        description="Identifier filename for the text submission",
    )
    threshold: float = Field(
        default=0.59,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for flagging plagiarism (default: 0.59)",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of top matching paragraph pairs to include per matched document",
    )
    reprocess: bool = Field(
        default=False,
        description="Bypass duplicate detection and process the submission anyway",
    )


class ClearDataResponse(BaseModel):
    """Response schema for bulk clearing administrative operation."""

    status: str = Field(..., description="Operation status (e.g. success)")
    message: str = Field(..., description="Summary message describing clearing action")


class CorpusStatsResponse(BaseModel):
    """Response schema for high-level corpus statistics."""

    total_documents: int = Field(
        ..., description="Total count of active documents in the corpus"
    )
    total_chunks: int = Field(
        ..., description="Total count of text chunks in the corpus"
    )
    total_embeddings: int = Field(
        ..., description="Total count of vector embeddings stored"
    )
    last_updated: str = Field(
        ..., description="ISO 8601 UTC timestamp of last corpus update"
    )


class IncidentResponse(ClearDataResponse):
    """Response schema for incident clearing operations."""

    pass


class ErrorResponse(BaseModel):
    """Response schema for API error responses."""

    detail: str = Field(..., description="Detailed error description message")


# ============================================================================
# Asynchronous Job Schemas
# ============================================================================


class AsyncScanJobResponse(BaseModel):
    """Response schema for queuing an asynchronous document scan job."""

    job_id: str = Field(..., description="Unique background scan job identifier")
    status: str = Field(..., description="Initial job status (queued)")
    status_url: str = Field(
        ..., description="Relative endpoint URL to poll for job status"
    )
    message: str = Field(..., description="Status description message")


class AsyncScanJobCancelResponse(BaseModel):
    """Response schema for cancelling an asynchronous document scan job."""

    job_id: str = Field(..., description="Unique background scan job identifier")
    status: str = Field(..., description="Updated job status (cancelled)")
    message: str = Field(..., description="Cancellation confirmation message")


class AsyncScanStatusResponse(BaseModel):
    """Response schema for checking the status of an asynchronous scan job."""

    job_id: str = Field(..., description="Unique background scan job identifier")
    status: str = Field(
        ...,
        description="Current job status: queued, processing, completed, failed, or cancelled",
    )
    progress_percent: int = Field(
        default=0, ge=0, le=100, description="Scan progress percentage (0-100)"
    )
    stage: str = Field(default="", description="Current processing stage description")
    filename: str = Field(
        ..., description="Filename of uploaded document being scanned"
    )
    created_at: str = Field(
        ..., description="ISO 8601 UTC timestamp when job was created"
    )
    completed_at: str | None = Field(
        default=None, description="ISO 8601 UTC timestamp when job finished"
    )
    result: SimilarityCheckResponse | None = Field(
        default=None, description="Detailed scan results when completed"
    )
    error: str | None = Field(
        default=None, description="Error message if scan job failed"
    )


class MetricSample(BaseModel):
    """Schema for an individual Prometheus metric sample.

    One entry of a family's ``metrics`` list. A family expands into several
    samples: a counter contributes ``<name>_total`` and ``<name>_created``, a
    histogram one ``<name>_bucket`` per boundary plus ``_sum`` and ``_count``.
    ``name`` is therefore the sample's own suffixed name, not the family's.
    """

    name: str = Field(
        ..., description="Sample name, including any type suffix (e.g. _total, _bucket)"
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Label key/value pairs identifying this sample; empty when unlabelled",
    )
    value: float = Field(..., description="Current numeric value of the sample")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "documents_total",
                "labels": {"status": "success"},
                "value": 10.0,
            }
        }
    )


class MetricFamily(BaseModel):
    """Schema for one Prometheus metric family in the /metrics/json payload.

    The endpoint returns a mapping of family name to this model, mirroring what
    ``src.core.metrics.generate_metrics_json()`` builds from the Prometheus text
    exposition format.
    """

    name: str = Field(..., description="Metric family name, without type suffixes")
    type: str = Field(
        ...,
        description="Prometheus metric type: counter, gauge, histogram, summary, or untyped",
    )
    help: str = Field(..., description="Documentation string describing the metric")
    metrics: List[MetricSample] = Field(
        default_factory=list, description="Individual samples belonging to this family"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "documents",
                "type": "counter",
                "help": "Cumulative number of documents ingested since process start.",
                "metrics": [
                    {"name": "documents_total", "labels": {}, "value": 10.0},
                    {
                        "name": "documents_created",
                        "labels": {},
                        "value": 1693356000.0,
                    },
                ],
            }
        }
    )


# ============================================================================
# Example Paginated Response for Documents
# ============================================================================


# Example document schema (can be extended based on actual document model)
class DocumentSchema(BaseModel):
    """Schema for a document resource."""

    id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Document filename")
    upload_date: str = Field(..., description="Upload timestamp")
    file_size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type of the document")
    user_id: str | None = Field(None, description="ID of the user who uploaded")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "doc_123456",
                "filename": "report_2024.pdf",
                "upload_date": "2024-01-01T00:00:00Z",
                "file_size": 1048576,
                "content_type": "application/pdf",
                "user_id": "user_789",
            }
        }
    )


# Document paginated response type alias for convenience
DocumentPaginatedResponse = PaginatedResponse[DocumentSchema]


class DocumentsListResponse(PaginatedResponse[DocumentSchema]):
    """
    Paginated response specifically for document lists.

    This extends the generic PaginatedResponse with document-specific
    fields or custom behavior if needed.
    """

    pass


# ============================================================================
# Batch Analysis History Schemas
# ============================================================================


class BatchRunCreateResponse(BaseModel):
    """Response schema for creating a batch analysis run."""

    run_id: int = Field(..., description="Unique batch run identifier")
    status: str = Field(..., description="Initial run status")
    message: str = Field(..., description="Status description message")


class BatchRunSummary(BaseModel):
    """Compact schema for a batch run in list views."""

    run_id: int = Field(..., description="Unique batch run identifier")
    started_at: str = Field(..., description="ISO 8601 timestamp of run start")
    completed_at: str | None = Field(
        default=None, description="ISO 8601 timestamp of completion"
    )
    status: str = Field(
        ..., description="Run status: running, completed, failed, cancelled"
    )
    trigger_source: str = Field(default="manual", description="What triggered the run")
    documents_scanned: int = Field(
        default=0, description="Number of documents processed"
    )
    documents_flagged: int = Field(default=0, description="Number of documents flagged")
    avg_similarity: float = Field(default=0.0, description="Average similarity score")
    max_similarity: float = Field(default=0.0, description="Peak similarity score")
    threshold_used: float = Field(default=0.75, description="Similarity threshold")
    duration_ms: int | None = Field(
        default=None, description="Run duration in milliseconds"
    )
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )
    created_by: str | None = Field(
        default=None, description="User who initiated the run"
    )


class BatchRunListResponse(BaseModel):
    """Paginated list of batch runs."""

    runs: list[BatchRunSummary] = Field(
        default_factory=list, description="List of batch runs"
    )
    page: int = Field(..., ge=1, description="Current page number")
    per_page: int = Field(..., ge=1, description="Items per page")
    total_items: int = Field(..., ge=0, description="Total number of runs")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(default=False, description="Whether there is a next page")
    has_previous: bool = Field(
        default=False, description="Whether there is a previous page"
    )


class BatchDocumentResult(BaseModel):
    """Schema for a single document result within a batch run."""

    id: int = Field(..., description="Result record ID")
    run_id: int = Field(..., description="Parent batch run ID")
    document_name: str = Field(..., description="Document filename")
    similarity_score: float = Field(default=0.0, description="Similarity score")
    severity: str = Field(default="none", description="Severity level")
    flagged: int = Field(default=0, description="Whether flagged (0 or 1)")
    matched_docs: list[str] = Field(
        default_factory=list, description="Matched document names"
    )
    processing_ms: int | None = Field(
        default=None, description="Processing time in ms"
    )


class BatchRunDetailResponse(BaseModel):
    """Detailed response for a single batch run including documents."""

    run: BatchRunSummary = Field(..., description="Batch run metadata")
    documents: list[BatchDocumentResult] = Field(
        default_factory=list, description="Document results"
    )
    severity_distribution: dict[str, int] = Field(
        default_factory=dict, description="Severity counts"
    )


class BatchTimelineEventResponse(BaseModel):
    """Schema for a timeline event in the audit trail."""

    event_id: int = Field(..., description="Event identifier")
    run_id: int | None = Field(default=None, description="Associated batch run")
    event_type: str = Field(..., description="Event type classification")
    severity: str = Field(default="info", description="Severity level")
    message: str = Field(..., description="Human-readable event description")
    metadata: Any | None = Field(default=None, description="Additional event data")
    created_at: str = Field(..., description="ISO 8601 timestamp")


class BatchAlertResponse(BaseModel):
    """Schema for an alert notification."""

    alert_id: int = Field(..., description="Alert identifier")
    run_id: int | None = Field(default=None, description="Associated batch run")
    alert_type: str = Field(..., description="Alert type classification")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Detailed alert message")
    is_read: int = Field(default=0, description="Read status (0 or 1)")
    created_at: str = Field(..., description="ISO 8601 timestamp")


class BatchHistorySummaryResponse(BaseModel):
    """Aggregated summary statistics for batch analysis history."""

    total_runs: int = Field(default=0, description="Total number of batch runs")
    completed_runs: int = Field(default=0, description="Successfully completed runs")
    failed_runs: int = Field(default=0, description="Failed runs")
    success_rate: float = Field(default=0.0, description="Success rate percentage")
    total_documents_scanned: int = Field(
        default=0, description="Total documents scanned"
    )
    total_documents_flagged: int = Field(
        default=0, description="Total documents flagged"
    )
    avg_similarity: float = Field(
        default=0.0, description="Average similarity across runs"
    )
    avg_duration_ms: int = Field(default=0, description="Average run duration in ms")
    last_run_at: str | None = Field(
        default=None, description="Timestamp of most recent run"
    )


class BatchTrendDataResponse(BaseModel):
    """Daily trend data point for analytics charts."""

    scan_date: str = Field(..., description="Date (YYYY-MM-DD)")
    total_runs: int = Field(default=0, description="Number of runs on this date")
    total_docs_scanned: int | None = Field(
        default=0, description="Documents scanned"
    )
    total_docs_flagged: int | None = Field(
        default=0, description="Documents flagged"
    )
    avg_similarity: float | None = Field(
        default=0.0, description="Average similarity"
    )
    peak_similarity: float | None = Field(default=0.0, description="Peak similarity")
    avg_duration_ms: float | None = Field(
        default=0.0, description="Average duration in ms"
    )


# ============================================================================
# Document Versioning Schemas
# ============================================================================


class DocumentVersionSnapshotResponse(BaseModel):
    """Response schema for a document version snapshot."""

    document_hash: str = Field(..., description="SHA-256 content hash")
    user_id: str = Field(..., description="User who uploaded")
    assignment_id: str = Field(..., description="Assignment identifier")
    filename: str = Field(default="untitled", description="Document filename")
    content_length: int = Field(default=0, description="Character count")
    word_count: int = Field(default=0, description="Word count")
    version_number: int = Field(default=1, description="Version sequence number")
    parent_hash: str | None = Field(default=None, description="Parent version hash")
    similarity_to_parent: float | None = Field(
        default=None, description="Similarity to parent"
    )
    created_at: str = Field(..., description="ISO 8601 creation timestamp")


class DocumentVersionListResponse(BaseModel):
    """Paginated list of document version snapshots."""

    items: list[DocumentVersionSnapshotResponse] = Field(default_factory=list)


# ============================================================================
# Anomaly Detection Schemas
# ============================================================================


class AnomalyScanResponse(BaseModel):
    """Response schema for an anomaly scan."""

    id: int = Field(..., description="Scan identifier")
    scan_type: str = Field(default="full", description="Scan type")
    status: str = Field(default="running", description="Scan status")
    documents_scanned: int = Field(default=0, description="Documents processed")
    anomalies_found: int = Field(default=0, description="Anomalies detected")
    started_at: str = Field(..., description="ISO 8601 start timestamp")
    completed_at: str | None = Field(
        default=None, description="Completion timestamp"
    )
    triggered_by: str = Field(default="system", description="Who triggered the scan")
    error_message: str | None = Field(default=None, description="Error if failed")


class AnomalyScanListResponse(BaseModel):
    """Paginated list of anomaly scans."""

    items: list[AnomalyScanResponse] = Field(default_factory=list)
    page: int = Field(..., ge=1)
    per_page: int = Field(..., ge=1)
    total: int = Field(..., ge=0)
    total_pages: int = Field(..., ge=0)


# ============================================================================
# Document Health Scoring Schemas
# ============================================================================


class HealthDimensionScore(BaseModel):
    """Score for a single health dimension."""

    name: str = Field(..., description="Dimension name")
    score: float = Field(..., description="Score 0-100")
    weight: float = Field(..., description="Weight in composite score")
    weighted: float = Field(..., description="Weighted contribution")
    details: str = Field(default="", description="Human-readable detail")


class DocumentHealthScoreResponse(BaseModel):
    """Response schema for a document health score."""

    id: int = Field(..., description="Score record ID")
    filename: str = Field(..., description="Document filename")
    overall_score: float = Field(..., description="Composite health score 0-100")
    grade: str = Field(..., description="Letter grade (A+, A, B+, B, C, D, F)")
    dimensions: list[HealthDimensionScore] = Field(
        default_factory=list, description="Per-dimension scores"
    )
    checked_at: str = Field(..., description="ISO 8601 timestamp of check")
    gate_passed: bool = Field(
        default=True, description="Whether document passed quality gate"
    )
    gate_reason: str = Field(default="", description="Quality gate decision reason")


class DocumentHealthListResponse(BaseModel):
    """Paginated list of document health scores."""

    scores: list[DocumentHealthScoreResponse] = Field(
        default_factory=list, description="Health scores"
    )
    page: int = Field(..., ge=1, description="Current page")
    per_page: int = Field(..., ge=1, description="Items per page")
    total_items: int = Field(..., ge=0, description="Total score records")
    total_pages: int = Field(..., ge=0, description="Total pages")
    has_next: bool = Field(default=False)
    has_previous: bool = Field(default=False)


# ============================================================================
# Similarity Heatmap & Clustering Schemas
# ============================================================================


class HeatmapSnapshotResponse(BaseModel):
    """Response schema for a similarity heatmap snapshot."""

    snapshot_id: int = Field(..., description="Snapshot identifier")
    labels: list[str] | None = Field(default=None, description="Document labels")
    matrix: list[list[float]] | None = Field(
        default=None, description="NxN similarity matrix"
    )
    document_count: int = Field(..., description="Number of documents")
    min_similarity: float = Field(..., description="Minimum pairwise similarity")
    max_similarity: float = Field(..., description="Maximum pairwise similarity")
    mean_similarity: float = Field(..., description="Average pairwise similarity")
    computed_at: str = Field(..., description="ISO 8601 computation timestamp")
    computed_by: str | None = Field(
        default=None, description="User who triggered computation"
    )
    notes: str | None = Field(default=None, description="Optional notes")
    hotspots_found: int | None = Field(
        default=None, description="Number of hotspots detected"
    )


class HeatmapSnapshotListResponse(BaseModel):
    """Paginated list of heatmap snapshots."""

    snapshots: list[HeatmapSnapshotResponse] = Field(default_factory=list)
    page: int = Field(..., ge=1)
    per_page: int = Field(..., ge=1)
    total_items: int = Field(..., ge=0)
    total_pages: int = Field(..., ge=0)
    has_next: bool = Field(default=False)
    has_previous: bool = Field(default=False)


class DocumentVersionLineageItem(BaseModel):
    """A single version entry in a lineage."""

    document_hash: str = Field(...)
    version_number: int = Field(...)
    filename: str = Field(default="untitled")
    word_count: int = Field(default=0)
    similarity_to_parent: float | None = Field(default=None)
    created_at: str = Field(...)


class DocumentVersionLineageResponse(BaseModel):
    """Response for a version lineage."""

    user_id: str = Field(...)
    assignment_id: str = Field(...)
    versions: list[DocumentVersionLineageItem] = Field(default_factory=list)
    total: int = Field(default=0)


class DocumentVersionDiffResponse(BaseModel):
    """Response for a pairwise version diff."""

    parent_hash: str = Field(...)
    child_hash: str = Field(...)
    similarity: float = Field(default=0.0)
    added_words: int = Field(default=0)
    removed_words: int = Field(default=0)
    changed_words: int = Field(default=0)
    jaccard_index: float = Field(default=0.0)
    computed_at: str = Field(...)


class DocumentVersionSummaryResponse(BaseModel):
    """Aggregate versioning statistics."""

    total_versions: int = Field(default=0)
    total_lineages: int = Field(default=0)
    total_diffs: int = Field(default=0)
    avg_similarity: float = Field(default=0.0)
    avg_versions_per_document: float = Field(default=0.0)
    unique_users: int = Field(default=0)


class DocumentVersionTrendPoint(BaseModel):
    """A single point in a similarity trend."""

    from_version: int = Field(...)
    to_version: int = Field(...)
    similarity: float | None = Field(default=None)
    added_words: int | None = Field(default=None)
    removed_words: int | None = Field(default=None)
    created_at: str = Field(...)


class DocumentVersionTrendResponse(BaseModel):
    """Similarity trend across versions."""

    user_id: str = Field(...)
    assignment_id: str = Field(...)
    trend: list[DocumentVersionTrendPoint] = Field(default_factory=list)
    total_points: int = Field(default=0)


class DocumentVersionMostRevisedItem(BaseModel):
    """Document with most revisions."""

    assignment_id: str = Field(...)
    user_id: str = Field(...)
    total_versions: int = Field(...)
    avg_similarity: float = Field(default=0.0)
    last_created: str = Field(...)


class DocumentVersionMostRevisedResponse(BaseModel):
    """Response for most-revised documents."""

    documents: list[DocumentVersionMostRevisedItem] = Field(default_factory=list)


class AnomalyAlertResponse(BaseModel):
    """Response schema for an anomaly alert."""

    id: int = Field(..., description="Alert identifier")
    scan_id: int | None = Field(default=None, description="Parent scan")
    anomaly_type: str = Field(..., description="Anomaly type")
    severity: str = Field(default="info", description="Severity level")
    title: str = Field(..., description="Alert title")
    description: str = Field(default="", description="Detailed description")
    confidence: float = Field(default=0.0, description="Detection confidence 0-1")
    affected_docs: list[str] = Field(
        default_factory=list, description="Affected documents"
    )
    evidence: dict[str, Any] = Field(default_factory=dict, description="Evidence data")
    is_acknowledged: bool = Field(default=False, description="Acknowledgement status")
    is_resolved: bool = Field(default=False, description="Resolution status")
    notes: str = Field(default="", description="Analyst notes")
    detected_at: str = Field(..., description="ISO 8601 detection timestamp")
    acknowledged_at: str | None = Field(default=None)
    resolved_at: str | None = Field(default=None)


class AnomalyAlertListResponse(BaseModel):
    """Paginated list of anomaly alerts."""

    items: list[AnomalyAlertResponse] = Field(default_factory=list)
    page: int = Field(..., ge=1)
    per_page: int = Field(..., ge=1)
    total: int = Field(..., ge=0)
    total_pages: int = Field(..., ge=0)
    has_next: bool = Field(default=False)
    has_previous: bool = Field(default=False)


class AnomalySummaryResponse(BaseModel):
    """Aggregate anomaly detection statistics."""

    total_alerts: int = Field(default=0)
    unacknowledged: int = Field(default=0)
    unresolved: int = Field(default=0)
    critical_unresolved: int = Field(default=0)
    avg_confidence: float = Field(default=0.0)
    total_scans: int = Field(default=0)
    completed_scans: int = Field(default=0)


class AnomalySeverityDistResponse(BaseModel):
    """Severity distribution."""

    distribution: dict[str, int] = Field(default_factory=dict)


class AnomalyConfigResponse(BaseModel):
    """Anomaly detection configuration."""

    z_score_threshold: float = Field(default=2.5)
    cluster_min_size: int = Field(default=3)
    cluster_similarity: float = Field(default=0.85)
    outlier_percentile: float = Field(default=95.0)
    collusion_threshold: float = Field(default=0.80)
    template_threshold: float = Field(default=0.75)
    enable_statistical: int = Field(default=1)
    enable_cluster: int = Field(default=1)
    enable_pattern: int = Field(default=1)
    enable_collusion: int = Field(default=1)
    updated_at: str | None = Field(default=None)


class HealthScoreSummaryResponse(BaseModel):
    """Aggregate health score statistics."""

    total_scored: int = Field(default=0, description="Total documents scored")
    avg_score: float = Field(default=0.0, description="Average health score")
    min_score: float = Field(default=0.0, description="Minimum score")
    max_score: float = Field(default=0.0, description="Maximum score")
    passed_gate: int = Field(default=0, description="Documents that passed gate")
    failed_gate: int = Field(default=0, description="Documents that failed gate")
    pass_rate: float = Field(default=0.0, description="Pass rate percentage")
    grade_distribution: dict[str, int] = Field(
        default_factory=dict, description="Grade counts"
    )
    last_checked_at: str | None = Field(
        default=None, description="Last check timestamp"
    )


class HealthGateConfigResponse(BaseModel):
    """Quality gate configuration."""

    min_score: float = Field(default=60.0, description="Minimum passing score")
    min_grade: str = Field(default="D", description="Minimum passing grade")
    enabled: bool = Field(default=True, description="Whether gate is active")


class HealthGateCheckResponse(BaseModel):
    """Result of a quality gate check."""

    filename: str = Field(..., description="Document filename")
    passed: bool = Field(..., description="Whether document passed")
    reason: str = Field(..., description="Decision reason")
    overall_score: float = Field(..., description="Document score")
    grade: str = Field(..., description="Document grade")


class HealthDimensionAvgResponse(BaseModel):
    """Average scores per health dimension."""

    dimensions: dict[str, float] = Field(
        default_factory=dict, description="Dimension name → avg score"
    )


class ClusterInfo(BaseModel):
    """Single cluster in a clustering result."""

    cluster_id: int = Field(..., description="Cluster identifier")
    documents: list[str] = Field(
        default_factory=list, description="Filenames in cluster"
    )
    centroid_score: float = Field(..., description="Average intra-cluster similarity")
    size: int = Field(..., description="Number of documents")


class ClusteringResultResponse(BaseModel):
    """Response schema for a clustering result."""

    result_id: int = Field(..., description="Result identifier")
    num_clusters: int = Field(..., description="Total clusters formed")
    silhouette_score: float = Field(
        ..., description="Silhouette quality score (-1 to 1)"
    )
    linkage_method: str = Field(..., description="Linkage method used")
    distance_threshold: float = Field(..., description="Distance cutoff")
    clusters: list[ClusterInfo] = Field(
        default_factory=list, description="Cluster details"
    )
    document_assignments: dict[str, int] = Field(
        default_factory=dict, description="Doc → cluster mapping"
    )
    computed_at: str = Field(..., description="ISO 8601 timestamp")


class HotspotResponse(BaseModel):
    """Response schema for a similarity hotspot."""

    hotspot_id: int = Field(..., description="Hotspot identifier")
    snapshot_id: int | None = Field(default=None, description="Associated snapshot")
    doc_a: str = Field(..., description="First document")
    doc_b: str = Field(..., description="Second document")
    similarity: float = Field(..., description="Similarity score")
    severity: str = Field(default="warning", description="Severity level")
    created_at: str = Field(..., description="Creation timestamp")
    is_resolved: int = Field(default=0, description="Resolution status")


class HotspotListResponse(BaseModel):
    """Response for hotspot listing."""

    hotspots: list[HotspotResponse] = Field(default_factory=list)
    total: int = Field(default=0)


class HotspotSummaryResponse(BaseModel):
    """Summary statistics for hotspots."""

    total_hotspots: int = Field(default=0)
    unresolved: int = Field(default=0)
    critical_unresolved: int = Field(default=0)
    avg_similarity: float = Field(default=0.0)


# ============================================================================
# Utility Functions
# ============================================================================


def create_paginated_response(
    items: list[T], page: int, per_page: int, total_items: int
) -> PaginatedResponse[T]:
    """
    Create a paginated response from items and pagination data.

    This is a convenience function for creating paginated responses.

    Args:
        items: List of items on the current page
        page: Current page number (1-indexed)
        per_page: Number of items per page
        total_items: Total number of items across all pages

    Returns:
        PaginatedResponse instance

    Example:
        >>> items = [{"id": 1, "name": "Doc 1"}]
        >>> response = create_paginated_response(items, 1, 20, 100)
        >>> response.page
        1
        >>> response.total_pages
        5
    """
    return PaginatedResponse.from_pagination_data(
        items=items, page=page, per_page=per_page, total_items=total_items
    )


def paginated_response_to_dict(response: PaginatedResponse[T]) -> dict[str, Any]:
    """
    Convert a PaginatedResponse to a dictionary format.

    Args:
        response: PaginatedResponse instance

    Returns:
        Dictionary representation

    Example:
        >>> response = create_paginated_response([], 1, 20, 0)
        >>> paginated_response_to_dict(response)
        {
            'items': [],
            'page': 1,
            'per_page': 20,
            'total_items': 0,
            'total_pages': 0,
            'has_next': False,
            'has_previous': False
        }
    """
    return response.model_dump()


def create_error_response(
    detail: str, error_code: str | None = None
) -> ErrorResponse:
    """
    Create a standardized error response.

    Args:
        detail: Error message detail
        error_code: Optional error code for debugging

    Returns:
        ErrorResponse instance
    """
    # If error code is provided, include it in the detail message
    if error_code:
        detail = f"[{error_code}] {detail}"

    return ErrorResponse(detail=detail)


# ============================================================================
# Export all schemas for easy importing
# ============================================================================

__all__ = [
    # Authentication
    "LoginResponse",
    "RefreshRequest",
    "TokenResponse",
    "RevokeRequest",
    "RevokeResponse",
    # Health
    "HealthCheckResponse",
    "HealthzResponse",
    "StatusResponse",
    # Pagination
    "PaginationParams",
    "PaginationMeta",
    "PaginatedResponse",
    "CursorPaginationParams",
    "CursorPaginatedResponse",
    "DocumentSchema",
    "DocumentPaginatedResponse",
    "DocumentsListResponse",
    "create_paginated_response",
    "paginated_response_to_dict",
    # Plagiarism
    "FlaggedChunkMatch",
    "MatchedDocument",
    "SimilarityCheckResponse",
    "DocumentUploadResponse",
    "ScanTextRequest",
    # Admin
    "ClearDataResponse",
    "CorpusStatsResponse",
    "IncidentResponse",
    # Batch Analysis History
    "BatchRunCreateResponse",
    "BatchRunSummary",
    "BatchRunListResponse",
    "BatchDocumentResult",
    "BatchRunDetailResponse",
    "BatchTimelineEventResponse",
    "BatchAlertResponse",
    "BatchHistorySummaryResponse",
    "BatchTrendDataResponse",
    # Document Versioning
    "DocumentVersionSnapshotResponse",
    "DocumentVersionListResponse",
    "DocumentVersionLineageResponse",
    "DocumentVersionDiffResponse",
    "DocumentVersionSummaryResponse",
    "DocumentVersionTrendResponse",
    "DocumentVersionMostRevisedResponse",
    # Anomaly Detection
    "AnomalyScanResponse",
    "AnomalyScanListResponse",
    "AnomalyAlertResponse",
    "AnomalyAlertListResponse",
    "AnomalySummaryResponse",
    "AnomalySeverityDistResponse",
    "AnomalyConfigResponse",
    # Document Health Scoring
    "HealthDimensionScore",
    "DocumentHealthScoreResponse",
    "DocumentHealthListResponse",
    "HealthScoreSummaryResponse",
    "HealthGateConfigResponse",
    "HealthGateCheckResponse",
    "HealthDimensionAvgResponse",
    # Similarity Heatmap & Clustering
    "HeatmapSnapshotResponse",
    "HeatmapSnapshotListResponse",
    "ClusterInfo",
    "ClusteringResultResponse",
    "HotspotResponse",
    "HotspotListResponse",
    "HotspotSummaryResponse",
    # Errors
    "ErrorResponse",
    "create_error_response",
    # Async Jobs
    "AsyncScanJobResponse",
    "AsyncScanJobCancelResponse",
    "AsyncScanStatusResponse",
]
