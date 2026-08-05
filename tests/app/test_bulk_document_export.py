"""Unit tests for Bulk Document Export Zip Builder in app/streamlit_app.py (#1507)."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.db.corpus_db import (
    add_chunks,
    add_document,
    configure_db_path,
    init_corpus_db,
)
from src.utils.bulk_export import create_documents_bulk_zip_archive


@pytest.fixture
def temp_corpus_db(tmp_path):
    """Set up isolated SQLite corpus database with sample documents and text chunks."""
    db_path = tmp_path / "test_corpus_export.db"
    configure_db_path(db_path)
    init_corpus_db()

    # Add sample document 1
    add_document(
        filename="essay_01.pdf",
        file_hash="hash111",
        student_name="Alice Smith",
        assignment_title="Ethics Essay",
        class_section="CS101",
    )
    add_chunks(
        [
            (
                1,
                "essay_01.pdf",
                0,
                "Academic integrity is fundamental in higher education.",
                np.zeros(128, dtype=np.float32),
            ),
            (
                2,
                "essay_01.pdf",
                1,
                "Plagiarism violates core ethical guidelines.",
                np.zeros(128, dtype=np.float32),
            ),
        ]
    )

    # Add sample document 2
    add_document(
        filename="assignment_2.docx",
        file_hash="hash222",
        student_name="Bob Jones",
        assignment_title="Research Paper",
        class_section="CS101",
    )
    add_chunks(
        [
            (
                3,
                "assignment_2.docx",
                0,
                "Artificial intelligence model evaluation framework.",
                np.zeros(128, dtype=np.float32),
            ),
        ]
    )

    yield db_path


def test_create_documents_bulk_zip_archive(temp_corpus_db):
    """Verify zip creation for selected documents containing contents and manifest CSV."""
    selected_files = ["essay_01.pdf", "assignment_2.docx"]
    zip_bytes = create_documents_bulk_zip_archive(selected_files)

    assert isinstance(zip_bytes, bytes)
    assert len(zip_bytes) > 0

    # Inspect zip contents
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert "essay_01.pdf" in namelist
        assert "assignment_2.docx" in namelist
        assert "export_manifest.csv" in namelist

        # Verify content of essay_01.pdf
        doc1_content = zf.read("essay_01.pdf").decode("utf-8")
        assert "Academic integrity is fundamental" in doc1_content
        assert "Plagiarism violates core ethical guidelines." in doc1_content

        # Verify manifest content
        manifest_csv = zf.read("export_manifest.csv").decode("utf-8-sig")
        df_manifest = pd.read_csv(io.StringIO(manifest_csv))
        assert len(df_manifest) == 2
        assert "essay_01.pdf" in df_manifest["filename"].values
        assert "assignment_2.docx" in df_manifest["filename"].values


def test_create_documents_bulk_zip_empty(temp_corpus_db):
    """Verify zip archive creation with an empty file list."""
    zip_bytes = create_documents_bulk_zip_archive([])
    assert isinstance(zip_bytes, bytes)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        assert len(zf.namelist()) == 0


def test_create_documents_bulk_zip_single_file(temp_corpus_db):
    """Verify zip archive creation with a single document."""
    zip_bytes = create_documents_bulk_zip_archive(["essay_01.pdf"])
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert "essay_01.pdf" in namelist
        assert "export_manifest.csv" in namelist
        assert "assignment_2.docx" not in namelist
