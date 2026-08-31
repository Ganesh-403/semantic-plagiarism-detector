"""
src/models/annotations.py
-------------------------
Pydantic models and schemas for the real-time collaborative annotation system.

Defines the data structures for highlights, comments, and WebSocket
messages exchanged between clients and the server during collaborative
document review sessions.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List, Any
from pydantic import BaseModel, Field, field_validator


class AnnotationType(str, Enum):
    """Enumeration of supported annotation types."""

    HIGHLIGHT = "highlight"
    COMMENT = "comment"
    RESOLVE = "resolve"
    DELETE = "delete"


class AnnotationColor(str, Enum):
    """Standardized color palette for highlights to ensure UI consistency."""

    YELLOW = "#fef08a"
    GREEN = "#bbf7d0"
    BLUE = "#bfdbfe"
    PINK = "#fbcfe8"
    ORANGE = "#fed7aa"


class HighlightData(BaseModel):
    """Represents a text highlight selection within a document."""

    start_index: int = Field(
        ..., ge=0, description="Starting character index (inclusive)."
    )
    end_index: int = Field(..., gt=0, description="Ending character index (exclusive).")
    color: AnnotationColor = Field(
        default=AnnotationColor.YELLOW, description="Highlight color."
    )
    text_snippet: str = Field(
        ..., max_length=500, description="The actual text that was highlighted."
    )

    @field_validator("end_index")
    @classmethod
    def validate_indices(cls, v: int, info: Any) -> int:
        """Ensure end_index is strictly greater than start_index."""
        # Note: Pydantic v2 validation context
        if "start_index" in info.data and v <= info.data["start_index"]:
            raise ValueError("end_index must be greater than start_index")
        return v


class CommentData(BaseModel):
    """Represents a comment attached to a specific highlight or document section."""

    content: str = Field(
        ..., min_length=1, max_length=2000, description="The comment text."
    )
    parent_annotation_id: Optional[str] = Field(
        None, description="ID of the parent annotation if this is a reply."
    )


class AnnotationCreate(BaseModel):
    """Schema for creating a new annotation via REST or WebSocket."""

    document_id: str = Field(
        ..., description="Unique identifier of the document being reviewed."
    )
    user_id: str = Field(..., description="ID of the user creating the annotation.")
    username: str = Field(..., description="Display name of the user.")
    type: AnnotationType
    highlight: Optional[HighlightData] = None
    comment: Optional[CommentData] = None

    @field_validator("highlight", "comment", mode="before")
    @classmethod
    def validate_payload_presence(cls, v: Any, info: Any) -> Any:
        """Ensure the appropriate payload is present for the annotation type."""
        ann_type = info.data.get("type")
        if ann_type == AnnotationType.HIGHLIGHT and not info.data.get("highlight"):
            raise ValueError("Highlight data is required for HIGHLIGHT annotations.")
        if ann_type == AnnotationType.COMMENT and not info.data.get("comment"):
            raise ValueError("Comment data is required for COMMENT annotations.")
        return v


class AnnotationRecord(AnnotationCreate):
    """Full database record for an annotation, including generated IDs and timestamps."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_resolved: bool = False


class WebSocketMessageType(str, Enum):
    """Types of messages broadcast over the WebSocket connection."""

    NEW_ANNOTATION = "new_annotation"
    UPDATE_ANNOTATION = "update_annotation"
    DELETE_ANNOTATION = "delete_annotation"
    CURSOR_MOVE = "cursor_move"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    ERROR = "error"


class WebSocketMessage(BaseModel):
    """Standardized envelope for all WebSocket messages."""

    type: WebSocketMessageType
    payload: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    room_id: str = Field(
        ..., description="The document/room ID this message belongs to."
    )
    sender_id: str = Field(..., description="ID of the user who triggered the event.")
