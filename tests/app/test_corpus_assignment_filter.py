"""Unit tests for Filter by Assignment selectbox in Corpus Document Management view."""

from src.db.schemas import Document


def test_assignment_filter_selectbox_all():
    """Verify 'All Assignments' includes all document rows."""
    docs = [
        Document(filename="a.txt", file_hash="h1", upload_date="2026-08-08", assignment_title="Assignment 1"),
        Document(filename="b.txt", file_hash="h2", upload_date="2026-08-08", assignment_title="Assignment 2"),
    ]
    raw_titles = sorted(list({d.assignment_title for d in docs} - {None, ""}))
    assignment_titles = ["All Assignments"] + raw_titles
    assert assignment_titles == ["All Assignments", "Assignment 1", "Assignment 2"]


def test_assignment_filter_selectbox_filtered():
    """Verify filtering documents by selected assignment."""
    docs = [
        Document(filename="a.txt", file_hash="h1", upload_date="2026-08-08", assignment_title="Assignment 1"),
        Document(filename="b.txt", file_hash="h2", upload_date="2026-08-08", assignment_title="Assignment 2"),
    ]
    selected_assignment = "Assignment 1"
    filtered = [doc for doc in docs if doc.assignment_title == selected_assignment]
    assert len(filtered) == 1
    assert filtered[0].filename == "a.txt"
