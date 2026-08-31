"""
tests/db/test_annotations_db.py
-------------------------------
Unit tests for the annotations database layer.
"""

import pytest
from datetime import datetime
from pathlib import Path

from src.db.annotations_db import (
    initialize_annotations_db,
    create_annotation,
    get_annotations_for_document,
    resolve_annotation,
    delete_annotation,
)
from src.models.annotations import (
    AnnotationRecord,
    AnnotationType,
    HighlightData,
    CommentData,
    AnnotationColor,
)


@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary database for testing."""
    db_path = tmp_path / "test_annotations.db"
    initialize_annotations_db(db_path)
    return db_path


class TestAnnotationsDB:
    """Test suite for annotation CRUD operations."""

    def test_create_and_retrieve_highlight(self, temp_db):
        """Verify a highlight annotation can be created and retrieved."""
        record = AnnotationRecord(
            document_id="doc_123",
            user_id="user_1",
            username="Alice",
            type=AnnotationType.HIGHLIGHT,
            highlight=HighlightData(
                start_index=10,
                end_index=20,
                color=AnnotationColor.YELLOW,
                text_snippet="sample text",
            ),
        )

        assert create_annotation(record, db_path=temp_db) is True

        annotations = get_annotations_for_document("doc_123", db_path=temp_db)
        assert len(annotations) == 1
        assert annotations[0].id == record.id
        assert annotations[0].highlight.text_snippet == "sample text"

    def test_create_and_retrieve_comment(self, temp_db):
        """Verify a comment annotation can be created and retrieved."""
        record = AnnotationRecord(
            document_id="doc_123",
            user_id="user_2",
            username="Bob",
            type=AnnotationType.COMMENT,
            comment=CommentData(content="This looks suspicious."),
        )

        assert create_annotation(record, db_path=temp_db) is True

        annotations = get_annotations_for_document("doc_123", db_path=temp_db)
        assert len(annotations) == 1
        assert annotations[0].comment.content == "This looks suspicious."

    def test_resolve_annotation(self, temp_db):
        """Verify an annotation can be marked as resolved."""
        record = AnnotationRecord(
            document_id="doc_123",
            user_id="user_1",
            username="Alice",
            type=AnnotationType.COMMENT,
            comment=CommentData(content="Check this."),
        )
        create_annotation(record, db_path=temp_db)

        assert resolve_annotation(record.id, db_path=temp_db) is True

        annotations = get_annotations_for_document("doc_123", db_path=temp_db)
        assert annotations[0].is_resolved is True

    def test_delete_annotation(self, temp_db):
        """Verify an annotation can be deleted."""
        record = AnnotationRecord(
            document_id="doc_123",
            user_id="user_1",
            username="Alice",
            type=AnnotationType.COMMENT,
            comment=CommentData(content="Delete me."),
        )
        create_annotation(record, db_path=temp_db)

        assert delete_annotation(record.id, db_path=temp_db) is True

        annotations = get_annotations_for_document("doc_123", db_path=temp_db)
        assert len(annotations) == 0

    def test_get_annotations_empty_document(self, temp_db):
        """Verify retrieving annotations for a non-existent document returns empty list."""
        annotations = get_annotations_for_document("non_existent_doc", db_path=temp_db)
        assert annotations == []
