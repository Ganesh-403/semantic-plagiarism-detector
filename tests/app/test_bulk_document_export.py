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

"""Unit tests for Bulk Document Export Zip Builder in app/streamlit_app.py (#1507)."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.db.corpus_db import add_chunks, add_document, configure_db_path, init_corpus_db
from src.utils.bulk_export import (
    create_bulk_export_zip,
    create_documents_bulk_zip_archive,
)


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
    """Verify zip creation for selected documents containing contents and manifest CSV with hierarchy preserved."""
    selected_files = ["essay_01.pdf", "assignment_2.docx"]
    zip_bytes = create_bulk_export_zip(selected_files, preserve_hierarchy=True)

    assert isinstance(zip_bytes, bytes)
    assert len(zip_bytes) > 0

    # Inspect zip contents - structured as {class_section}/{assignment_title}/{filename}
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert "CS101/Ethics_Essay/essay_01.pdf" in namelist
        assert "CS101/Research_Paper/assignment_2.docx" in namelist
        assert "export_manifest.csv" in namelist

        # Verify content of essay_01.pdf
        doc1_content = zf.read("CS101/Ethics_Essay/essay_01.pdf").decode("utf-8")
        assert "Academic integrity is fundamental" in doc1_content
        assert "Plagiarism violates core ethical guidelines." in doc1_content

        # Verify manifest content
        manifest_csv = zf.read("export_manifest.csv").decode("utf-8-sig")
        df_manifest = pd.read_csv(io.StringIO(manifest_csv))
        assert len(df_manifest) == 2
        assert "essay_01.pdf" in df_manifest["filename"].values
        assert "assignment_2.docx" in df_manifest["filename"].values


def test_create_bulk_export_zip_flattened(temp_corpus_db):
    """Verify zip creation with preserve_hierarchy=False flattens files into root."""
    selected_files = ["essay_01.pdf", "assignment_2.docx"]
    zip_bytes = create_bulk_export_zip(selected_files, preserve_hierarchy=False)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert "essay_01.pdf" in namelist
        assert "assignment_2.docx" in namelist
        assert "export_manifest.csv" in namelist


def test_create_documents_bulk_zip_empty(temp_corpus_db):
    """Verify zip archive creation with an empty file list."""
    zip_bytes = create_bulk_export_zip([])
    assert isinstance(zip_bytes, bytes)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        assert len(zf.namelist()) == 0


def test_create_documents_bulk_zip_single_file(temp_corpus_db):
    """Verify zip archive creation with a single document."""
    zip_bytes = create_documents_bulk_zip_archive(
        ["essay_01.pdf"], preserve_hierarchy=False
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert "essay_01.pdf" in namelist
        assert "export_manifest.csv" in namelist
        assert "assignment_2.docx" not in namelist
