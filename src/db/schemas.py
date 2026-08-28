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

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class DictLikeModel(BaseModel):
    """Base Pydantic model with support for dictionary-like subscripting and iteration."""

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def keys(self) -> Any:
        return self.model_fields.keys()

    def __iter__(self) -> Any:
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

    incident_id: Optional[int] = None
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
