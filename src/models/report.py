"""Data contract for reports built from supplied analysis results.

The report package does not look up analysis IDs or manufacture sample results.
Callers supply document names, a square similarity matrix and optional matches in
``ReportRequest.analysis_data``. Scores use the closed interval [0, 1].
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class ReportFormat(str, Enum):
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"


class ReportType(str, Enum):
    SUMMARY = "summary"
    DETAILED = "detailed"
    COMPARISON = "comparison"
    EXECUTIVE = "executive"
    BATCH = "batch"


class ReportStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportConfig:
    include_heatmap: bool = True
    include_matches: bool = True
    include_summary_stats: bool = True
    max_matches: int = 100
    highlight_threshold: float = 0.8

    def __post_init__(self):
        if not isinstance(self.max_matches, int) or self.max_matches < 0:
            raise ValueError("max_matches must be a nonnegative integer")
        if not 0 <= self.highlight_threshold <= 1:
            raise ValueError("highlight_threshold must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReportRequest:
    analysis_id: str
    analysis_data: dict[str, Any]
    document_ids: list[str] = field(default_factory=list)
    title: str = ""
    description: str = ""
    report_type: ReportType = ReportType.DETAILED
    format: ReportFormat = ReportFormat.HTML
    config: ReportConfig = field(default_factory=ReportConfig)
    include_details: bool = True

    def __post_init__(self):
        self.format = ReportFormat(self.format)
        self.report_type = ReportType(self.report_type)
        if self.config is None:
            self.config = ReportConfig()


@dataclass
class Report:
    title: str
    description: str = ""
    report_type: ReportType = ReportType.DETAILED
    format: ReportFormat = ReportFormat.HTML
    id: str = field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = field(default_factory=datetime.now)
    status: ReportStatus = ReportStatus.PENDING
    progress: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    document_names: list[str] = field(default_factory=list)
    similarity_matrix: list[list[float]] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)
    file_path: str | None = None
    file_size: int = 0
    error: str | None = None

    def update_progress(self, progress: int) -> None:
        if not 0 <= progress <= 100:
            raise ValueError("progress must be between zero and 100")
        self.progress = progress

    def mark_completed(self, file_path: str, file_size: int) -> None:
        self.file_path = file_path
        self.file_size = file_size
        self.status = ReportStatus.COMPLETED
        self.progress = 100
        self.error = None

    def mark_failed(self, error: str) -> None:
        self.status = ReportStatus.FAILED
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            generated_at=self.generated_at.isoformat(),
            status=self.status.value,
            report_type=self.report_type.value,
            format=self.format.value,
        )
        return result


@dataclass
class ReportResponse:
    success: bool
    message: str
    report_id: str | None = None
    file_path: str | None = None
    file_size: int = 0
    download_url: str | None = None
    format: str | None = None
    record_count: int = 0
    report: Report | None = None
    error: str | None = None
