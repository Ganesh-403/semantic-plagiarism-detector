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

"""
tests/core/models/test_categorization.py
----------------------------------------
Comprehensive unit tests for the document categorization domain models.

Verifies dataclass validation, serialization, normalization, and
collection utility methods.
"""

from datetime import datetime

import pytest

from src.core.models.categorization import (
    DocumentTag,
    TagCategory,
    TagCollection,
    TagSource,
)


class TestDocumentTagValidation:
    """Test suite for DocumentTag initialization and validation."""

    def test_basic_initialization(self):
        """Verify basic tag creation with default values."""
        tag = DocumentTag(name="Machine Learning")

        assert tag.name == "Machine Learning"
        assert tag.source == TagSource.MANUAL
        assert tag.confidence == 1.0
        assert tag.category == TagCategory.CUSTOM

    def test_name_normalization_strips_whitespace(self):
        """Verify leading/trailing whitespace is stripped from tag names."""
        tag = DocumentTag(name="   Artificial Intelligence   ")
        assert tag.name == "Artificial Intelligence"

    def test_name_normalization_title_case(self):
        """Verify tag names are converted to title case."""
        tag = DocumentTag(name="natural language processing")
        assert tag.name == "Natural Language Processing"

    def test_name_normalization_removes_special_chars(self):
        """Verify special characters (except hyphens/underscores) are removed."""
        tag = DocumentTag(name="C++ Programming!")
        assert tag.name == "C Programming"

    def test_name_normalization_collapses_spaces(self):
        """Verify multiple spaces are collapsed into a single space."""
        tag = DocumentTag(name="Data    Science")
        assert tag.name == "Data Science"

    def test_empty_name_raises_value_error(self):
        """Verify empty or None names raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DocumentTag(name="")

        with pytest.raises(ValueError, match="cannot be empty"):
            DocumentTag(name=None)

    def test_invalid_only_name_raises_value_error(self):
        """Verify names with only special characters raise ValueError."""
        with pytest.raises(ValueError, match="only invalid characters"):
            DocumentTag(name="!!!@@@###")

    def test_confidence_bounds_validation(self):
        """Verify confidence must be between 0.0 and 1.0."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            DocumentTag(name="Test", confidence=1.5)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            DocumentTag(name="Test", confidence=-0.1)

    def test_string_enum_conversion(self):
        """Verify string inputs for source/category are converted to Enums."""
        tag = DocumentTag(name="Test", source="ai_generated", category="subject")

        assert tag.source == TagSource.AI_GENERATED
        assert tag.category == TagCategory.SUBJECT


class TestDocumentTagMethods:
    """Test suite for DocumentTag utility methods."""

    def test_is_low_confidence_true(self):
        """Verify is_low_confidence returns True for scores below threshold."""
        tag = DocumentTag(name="Test", confidence=0.4)
        assert tag.is_low_confidence() is True
        assert tag.is_low_confidence(threshold=0.5) is True

    def test_is_low_confidence_false(self):
        """Verify is_low_confidence returns False for scores above threshold."""
        tag = DocumentTag(name="Test", confidence=0.8)
        assert tag.is_low_confidence() is False

    def test_get_css_classes_manual_high_confidence(self):
        """Verify CSS classes for manual, high-confidence tags."""
        tag = DocumentTag(name="Test", source=TagSource.MANUAL, confidence=0.95)
        classes = tag.get_css_classes()

        assert "tag-badge" in classes
        assert "tag-source-manual" in classes
        assert "tag-high-confidence" in classes
        assert "tag-low-confidence" not in classes

    def test_get_css_classes_ai_low_confidence(self):
        """Verify CSS classes for AI-generated, low-confidence tags."""
        tag = DocumentTag(name="Test", source=TagSource.AI_GENERATED, confidence=0.3)
        classes = tag.get_css_classes()

        assert "tag-source-ai_generated" in classes
        assert "tag-low-confidence" in classes

    def test_to_dict_serialization(self):
        """Verify to_dict correctly serializes Enums and datetimes."""
        tag = DocumentTag(
            name="Test",
            source=TagSource.AI_GENERATED,
            category=TagCategory.TOPIC,
            confidence=0.85,
        )
        data = tag.to_dict()

        assert data["name"] == "Test"
        assert data["source"] == "ai_generated"
        assert data["category"] == "topic"
        assert data["confidence"] == 0.85
        assert isinstance(data["created_at"], str)  # ISO format string

    def test_from_dict_deserialization(self):
        """Verify from_dict correctly deserializes strings back to Enums/dates."""
        data = {
            "name": "Test",
            "source": "ai_generated",
            "category": "topic",
            "confidence": 0.85,
            "created_at": "2024-01-01T12:00:00",
        }
        tag = DocumentTag.from_dict(data)

        assert tag.source == TagSource.AI_GENERATED
        assert tag.category == TagCategory.TOPIC
        assert isinstance(tag.created_at, datetime)

    def test_hash_and_equality(self):
        """Verify tags can be used in sets and compared for equality."""
        tag1 = DocumentTag(name="Test", source=TagSource.MANUAL)
        tag2 = DocumentTag(name="test", source=TagSource.MANUAL)  # Case insensitive
        tag3 = DocumentTag(name="Test", source=TagSource.AI_GENERATED)

        # tag1 and tag2 should be equal (case-insensitive name match)
        assert tag1 == tag2
        assert hash(tag1) == hash(tag2)

        # tag1 and tag3 should not be equal (different source)
        assert tag1 != tag3

        # Should work in sets
        tag_set = {tag1, tag2, tag3}
        assert len(tag_set) == 2


class TestTagCollection:
    """Test suite for the TagCollection utility class."""

    def test_add_and_len(self):
        """Verify adding tags updates the collection length."""
        collection = TagCollection()
        collection.add(DocumentTag(name="Tag1"))
        collection.add(DocumentTag(name="Tag2"))

        assert len(collection) == 2

    def test_add_prevents_duplicates(self):
        """Verify adding the same tag twice doesn't create duplicates."""
        collection = TagCollection()
        tag = DocumentTag(name="Tag1")

        collection.add(tag)
        collection.add(tag)

        assert len(collection) == 1

    def test_remove_existing_tag(self):
        """Verify removing an existing tag returns True and updates collection."""
        collection = TagCollection(tags=[DocumentTag(name="Tag1")])

        assert collection.remove("Tag1") is True
        assert len(collection) == 0

    def test_remove_nonexistent_tag(self):
        """Verify removing a nonexistent tag returns False."""
        collection = TagCollection(tags=[DocumentTag(name="Tag1")])

        assert collection.remove("Tag2") is False
        assert len(collection) == 1

    def test_filter_by_source(self):
        """Verify filtering by source returns correct subset."""
        collection = TagCollection(
            tags=[
                DocumentTag(name="Manual", source=TagSource.MANUAL),
                DocumentTag(name="AI", source=TagSource.AI_GENERATED),
                DocumentTag(name="AI2", source=TagSource.AI_GENERATED),
            ]
        )

        ai_tags = collection.filter_by_source(TagSource.AI_GENERATED)
        assert len(ai_tags) == 2
        assert all(t.source == TagSource.AI_GENERATED for t in ai_tags)

    def test_get_low_confidence_tags(self):
        """Verify low confidence filtering works correctly."""
        collection = TagCollection(
            tags=[
                DocumentTag(name="High", confidence=0.9),
                DocumentTag(name="Low", confidence=0.3),
                DocumentTag(name="Med", confidence=0.7),
            ]
        )

        low_tags = collection.get_low_confidence_tags(threshold=0.6)
        assert len(low_tags) == 1
        assert low_tags[0].name == "Low"
