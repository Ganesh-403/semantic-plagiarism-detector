from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class DictLikeModel(BaseModel):
    """Base Pydantic model with support for dictionary-like subscripting and iteration."""

    def __getitem__(self, item: str) -> any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def get(self, item: str, default: any = None) -> any:
        return getattr(self, item, default)

    def keys(self) -> any:
        return self.model_fields.keys()

    def __iter__(self) -> any:
        return iter(self.model_fields.keys())


class User(DictLikeModel):
    """Pydantic model representing a User record DTO."""
    id: Optional[int] = None
    username: str
    role: str
    is_active: bool = True
    version: int = 1



class Document(DictLikeModel):
    """Pydantic model representing a Document metadata DTO."""
    filename: str
    file_hash: str
    upload_date: str
    class_section: Optional[str] = None
    student_name: Optional[str] = None
    assignment_title: Optional[str] = None
    pdf_author: Optional[str] = None
    pdf_creation_date: Optional[str] = None
    pdf_title: Optional[str] = None
    detected_language: Optional[str] = None
    deleted_at: Optional[str] = None


class MatchResult(DictLikeModel):
    """Pydantic model representing a similarity PlagiarismIncident DTO."""
    incident_id: Optional[str] = None
    document_a: str
    document_b: str
    similarity_score: float
    severity_rank: Optional[str] = None
    review_status: str = "Pending"
    date_flagged: Optional[str] = None
    last_seen: Optional[str] = None
    threshold_at_time_of_flag: Optional[float] = None

    @property
    def doc_a(self) -> str:
        return self.document_a

    @property
    def doc_b(self) -> str:
        return self.document_b

    @property
    def similarity(self) -> float:
        return self.similarity_score
