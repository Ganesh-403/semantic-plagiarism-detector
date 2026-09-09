"""Shared similarity analysis input, configuration and result records."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from uuid import uuid4
from src.models.document import DocumentStatus


class MatchSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SimilarityType(str, Enum):
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class SimilarityConfig:
    lexical_weight: float = 0.3
    semantic_weight: float = 0.7
    lexical_threshold: float = 0.3
    semantic_threshold: float = 0.4
    hybrid_threshold: float = 0.5
    use_stopwords: bool = True
    chunk_size: int = 100
    overlap_size: int = 20
    max_chunks: int = 1000

    def to_dict(self):
        return asdict(self)


@dataclass
class MatchResult:
    source_document: str = ""
    target_document: str = ""
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_score: float = 0.0
    matched_text: str = ""
    severity: MatchSeverity = MatchSeverity.NONE
    metadata: dict = field(default_factory=dict)

    def get_severity(self):
        return (
            MatchSeverity.HIGH
            if self.hybrid_score >= 0.8
            else MatchSeverity.MEDIUM
            if self.hybrid_score >= 0.6
            else MatchSeverity.LOW
            if self.hybrid_score >= 0.4
            else MatchSeverity.NONE
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class AnalysisResult:
    source_document_id: str = ""
    target_document_ids: list[str] = field(default_factory=list)
    analysis_type: SimilarityType = SimilarityType.HYBRID
    id: str = field(default_factory=lambda: str(uuid4()))
    matches: list[MatchResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.PENDING
    processing_time_ms: float = 0.0
    error_message: str = ""

    def add_match(self, match):
        self.matches.append(match)

    def mark_completed(self):
        self.status = DocumentStatus.COMPLETED

    def mark_failed(self, message=""):
        self.status = DocumentStatus.FAILED
        self.error_message = message

    def to_dict(self):
        return asdict(self)


@dataclass
class DocumentPair:
    source_content: str = ""
    target_content: str = ""
    source_id: str = ""
    target_id: str = ""

    def is_valid(self):
        return bool(self.source_content.strip() and self.target_content.strip())


@dataclass
class AnalysisStatistics:
    total_comparisons: int = 0
    average_similarity: float = 0.0
    max_similarity: float = 0.0
    min_similarity: float = 0.0
    high_similarity_count: int = 0

    def compute(self, matches):
        scores = [m.hybrid_score for m in matches]
        self.total_comparisons = len(scores)
        self.average_similarity = sum(scores) / len(scores) if scores else 0.0
        self.max_similarity = max(scores, default=0.0)
        self.min_similarity = min(scores, default=0.0)
        self.high_similarity_count = sum(
            m.get_severity() == MatchSeverity.HIGH for m in matches
        )
        return self

    def to_dict(self):
        return asdict(self)
