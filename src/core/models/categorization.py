"""
src/core/models/categorization.py
---------------------------------
Domain models for document categorization, tagging, and classification.

This module extracts dataclasses and domain logic that were previously
embedded directly in the UI routing layer (streamlit_app.py). By moving
these models to the core domain layer, we enforce a clean separation of
concerns between the view/routing layer and the business logic.

Issue #2782: Extract Domain Models from streamlit_app.py.
Issue #2812: Add HTML badge generation for low-confidence visual indicators.
"""

from __future__ import annotations

import re
import html
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TagSource(str, Enum):
    """Enumeration of tag origin sources."""

    MANUAL = "manual"
    AI_GENERATED = "ai_generated"
    IMPORTED = "imported"
    RULE_BASED = "rule_based"


class TagCategory(str, Enum):
    """Standardized categories for document tags."""

    SUBJECT = "subject"
    TOPIC = "topic"
    GRADE_LEVEL = "grade_level"
    DOCUMENT_TYPE = "document_type"
    CUSTOM = "custom"
@dataclass
class DocumentTag:
    """Represents a semantic tag assigned to a document or text chunk.

    This model captures not just the tag name, but also its origin,
    confidence level (for AI-generated tags), and categorization.

    Attributes:
        name: The display name of the tag (e.g., "Machine Learning").
        source: How the tag was generated (manual, AI, etc.).
        confidence: Confidence score between 0.0 and 1.0. Defaults to 1.0
                   for manually created tags.
        category: The semantic category of the tag.
        created_at: Timestamp of tag creation.
        metadata: Optional dictionary for additional tag properties.
    """

    name: str
    source: TagSource = TagSource.MANUAL
    confidence: float = 1.0
    category: TagCategory = TagCategory.CUSTOM
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate tag attributes after initialization."""
        # Clean and normalize the tag name
        self.name = self._normalize_name(self.name)

        # Validate confidence bounds
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Tag confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        # Ensure source and category are proper Enum instances
        if isinstance(self.source, str):
            self.source = TagSource(self.source.lower())
        if isinstance(self.category, str):
            self.category = TagCategory(self.category.lower())

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize tag name by stripping whitespace and converting to title case.

        Removes special characters except hyphens and underscores, and
        ensures consistent casing for database storage and UI display.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Tag name cannot be empty or None")

        # Strip leading/trailing whitespace
        cleaned = name.strip()

        # Remove invalid characters (keep alphanumeric, spaces, hyphens, underscores)
        cleaned = re.sub(r"[^\w\s\-]", "", cleaned)

        # Collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned)

        if not cleaned:
            raise ValueError("Tag name contains only invalid characters")

        return cleaned.title()

    def is_low_confidence(self, threshold: float = 0.6) -> bool:
        """Check if the tag confidence is below the verification threshold.

        Args:
            threshold: The minimum confidence level required. Defaults to 0.6.

        Returns:
            True if confidence is below threshold, False otherwise.
        """
        return self.confidence < threshold

    def get_css_classes(self) -> str:
        """Generate CSS class string for UI rendering based on tag properties.

        Returns space-separated CSS classes that can be applied to HTML
        elements to style the tag according to its source and confidence.
        """
        classes = ["tag-badge"]

        # Add source-specific classes
        classes.append(f"tag-source-{self.source.value}")

        # Add confidence-based classes
        if self.is_low_confidence():
            classes.append("tag-low-confidence")
        elif self.confidence >= 0.9:
            classes.append("tag-high-confidence")

        return " ".join(classes)

    def get_html_badge(self, show_confidence: bool = True) -> str:
        """Generate a raw HTML badge string for the tag.

        This method provides the core HTML structure, while the Streamlit
        renderer (app/components/tag_renderer.py) handles CSS injection
        and UI-specific formatting.

        Args:
            show_confidence: Whether to include the confidence percentage.

        Returns:
            HTML string for the tag badge.
        """
        safe_name = html.escape(self.name)
        classes = self.get_css_classes()

        inner_html = ""
        if self.is_low_confidence():
            inner_html += '<span class="tag-warning-icon">⚠️</span>'

        inner_html += safe_name

        if show_confidence and self.source == TagSource.AI_GENERATED:
            inner_html += f" ({int(self.confidence * 100)}%)"

        return f'<span class="{classes}">{inner_html}</span>'

    def to_dict(self) -> dict:
        """Serialize the tag to a dictionary for JSON storage or API responses."""
        data = asdict(self)
        # Convert datetime to ISO format string
        data["created_at"] = self.created_at.isoformat()
        # Convert Enums to their string values
        data["source"] = self.source.value
        data["category"] = self.category.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> DocumentTag:
        """Deserialize a dictionary into a DocumentTag instance.

        Handles conversion of ISO datetime strings and string enums
        back into their proper Python types.
        """
        # Parse datetime if present
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        return cls(**data)

    def __hash__(self) -> int:
        """Make DocumentTag hashable for use in sets and as dict keys."""
        return hash((self.name.lower(), self.source, self.category))

    def __eq__(self, other: object) -> bool:
        """Define equality based on name, source, and category."""
        if not isinstance(other, DocumentTag):
            return False
        return (
            self.name.lower() == other.name.lower()
            and self.source == other.source
            and self.category == other.category
        )


@dataclass
class TagCollection:
    """A collection of DocumentTags with utility methods for filtering and grouping.

    Provides a higher-level abstraction over a simple list of tags, enabling
    easy filtering by source, category, or confidence level.
    """

    tags: List[DocumentTag] = field(default_factory=list)

    def add(self, tag: DocumentTag) -> None:
        """Add a tag to the collection, preventing duplicates."""
        if tag not in self.tags:
            self.tags.append(tag)
        else:
            logger.debug("Tag '%s' already exists in collection. Skipping.", tag.name)

    def remove(self, tag_name: str) -> bool:
        """Remove a tag by name. Returns True if removed, False if not found."""
        initial_count = len(self.tags)
        self.tags = [t for t in self.tags if t.name.lower() != tag_name.lower()]
        return len(self.tags) < initial_count

    def filter_by_source(self, source: TagSource) -> List[DocumentTag]:
        """Return all tags matching the specified source."""
        return [t for t in self.tags if t.source == source]

    def filter_by_category(self, category: TagCategory) -> List[DocumentTag]:
        """Return all tags matching the specified category."""
        return [t for t in self.tags if t.category == category]

    def get_low_confidence_tags(self, threshold: float = 0.6) -> List[DocumentTag]:
        """Return all tags with confidence below the specified threshold."""
        return [t for t in self.tags if t.is_low_confidence(threshold)]

    def get_ai_generated_tags(self) -> List[DocumentTag]:
        """Return all tags generated by the AI model."""
        return self.filter_by_source(TagSource.AI_GENERATED)

    def to_list(self) -> List[dict]:
        """Serialize the entire collection to a list of dictionaries."""
        return [tag.to_dict() for tag in self.tags]

    @classmethod
    def from_list(cls, data: List[dict]) -> TagCollection:
        """Deserialize a list of dictionaries into a TagCollection."""
        tags = [DocumentTag.from_dict(d) for d in data]
        return cls(tags=tags)

    def __len__(self) -> int:
        return len(self.tags)

    def __iter__(self):
        return iter(self.tags)
