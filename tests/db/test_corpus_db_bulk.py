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
tests/db/test_corpus_db_bulk.py
--------------------------------
Unit test suite for bulk document metadata insertion in src/db/corpus_db.py.
Includes parametrized tests for diverse metadata combinations (Issue #3412).
"""

import sqlite3

import pytest

from src.db.corpus_db import (
    add_documents_bulk,
    clear_all_data,
    get_all_documents,
    init_corpus_db,
)


@pytest.fixture(autouse=True)
def setup_teardown(mock_db):
    # mock_db patches the corpus DB path to an isolated temp file (Issue #2263)
    init_corpus_db()
    clear_all_data()
    yield
    clear_all_data()


def test_add_documents_bulk_success():
    docs = [
        {
            "filename": "bulk_doc_1.pdf",
            "file_hash": "hash_bulk_1",
            "class_section": "Class A",
            "student_name": "Alice",
            "assignment_title": "HW1",
            "detected_language": "en",
        },
        {
            "filename": "bulk_doc_2.pdf",
            "file_hash": "hash_bulk_2",
            "class_section": "Class B",
            "student_name": "Bob",
            "assignment_title": "HW2",
            "detected_language": "es",
        },
        {
            "filename": "bulk_doc_3.pdf",
            "file_hash": "hash_bulk_3",
            "class_section": "Class A",
            "student_name": "Charlie",
            "assignment_title": "HW1",
        },
    ]

    success_count = add_documents_bulk(docs)
    assert success_count == 3

    all_docs = get_all_documents()
    assert len(all_docs) == 3
    filenames = [d["filename"] for d in all_docs]
    assert "bulk_doc_1.pdf" in filenames
    assert "bulk_doc_2.pdf" in filenames
    assert "bulk_doc_3.pdf" in filenames

    # Find docs and assert language
    doc1 = next(d for d in all_docs if d["filename"] == "bulk_doc_1.pdf")
    doc2 = next(d for d in all_docs if d["filename"] == "bulk_doc_2.pdf")
    doc3 = next(d for d in all_docs if d["filename"] == "bulk_doc_3.pdf")
    assert doc1["detected_language"] == "en"
    assert doc2["detected_language"] == "es"
    assert doc3["detected_language"] is None


def test_add_documents_bulk_duplicate_ignore():
    # Test that inserting duplicate filenames/hashes does not fail but ignores them
    docs = [
        {"filename": "dup_1.pdf", "file_hash": "hash_dup_1"},
        {"filename": "dup_2.pdf", "file_hash": "hash_dup_2"},
    ]
    assert add_documents_bulk(docs) == 2

    # Try inserting same docs plus one new
    docs_with_dups = [
        {"filename": "dup_1.pdf", "file_hash": "hash_dup_1"},
        {"filename": "dup_3.pdf", "file_hash": "hash_dup_3"},
    ]
    # Because of INSERT OR IGNORE, dup_1 is skipped, dup_3 is inserted.
    success_count = add_documents_bulk(docs_with_dups)
    assert success_count == 1

    all_docs = get_all_documents()
    assert len(all_docs) == 3


def test_add_documents_bulk_empty():
    assert add_documents_bulk([]) == 0
    assert len(get_all_documents()) == 0


def test_add_documents_bulk_missing_fields():
    # Missing filename is rejected upfront with an IntegrityError (not swallowed)
    docs = [{"file_hash": "hash123"}]  # filename is missing

    with pytest.raises(sqlite3.IntegrityError):
        add_documents_bulk(docs)
    assert len(get_all_documents()) == 0


# 10 Diverse Metadata Test Cases for Parametrized Testing
DIVERSE_METADATA_CASES = [
    # 1. Japanese Unicode student name and assignment title
    {
        "filename": "japanese_essay_2026.pdf",
        "file_hash": "hash_jp_001",
        "class_section": "クラス101",
        "student_name": "山田太郎",
        "assignment_title": "日本語の論文_最終版",
        "pdf_author": "山田太郎",
        "pdf_title": "日本語論文",
        "detected_language": "ja",
    },
    # 2. Cyrillic Unicode student name and section
    {
        "filename": "cyrillic_analysis.pdf",
        "file_hash": "hash_cyr_002",
        "class_section": "Секция А-1",
        "student_name": "Александр Пушкин",
        "assignment_title": "Евгений Онегин - Анализ",
        "pdf_author": "Пушкин",
        "pdf_title": "Литературный анализ",
        "detected_language": "ru",
    },
    # 3. Arabic Unicode student name and class section
    {
        "filename": "arabic_report.pdf",
        "file_hash": "hash_ar_003",
        "class_section": "قسم العلوم والفيزياء",
        "student_name": "محمد بن طارق",
        "assignment_title": "تقرير الفيزياء الفلكية",
        "pdf_author": "محمد طارق",
        "pdf_title": "التقرير النهائي",
        "detected_language": "ar",
    },
    # 4. French Accents in student name and title
    {
        "filename": "french_bioethics.pdf",
        "file_hash": "hash_fr_004",
        "class_section": "Équipe Biologie #3",
        "student_name": "René François Élizabeth",
        "assignment_title": "Thèse de Bioéthique & Écologie",
        "pdf_author": "René François",
        "pdf_title": "Thèse Bioéthique",
        "detected_language": "fr",
    },
    # 5. Empty assignment title string
    {
        "filename": "empty_assignment_title.pdf",
        "file_hash": "hash_empty_assign_005",
        "class_section": "CS-101",
        "student_name": "John Doe",
        "assignment_title": "",
        "detected_language": "en",
    },
    # 6. Long filename (250+ characters)
    {
        "filename": "long_filename_" + "x" * 230 + "_v1.pdf",
        "file_hash": "hash_long_fn_006",
        "class_section": "Section Alpha",
        "student_name": "Alice Smith",
        "assignment_title": "Filename Test Assignment",
        "detected_language": "en",
    },
    # 7. Special characters & SQL quotes in student name and title
    {
        "filename": "special_chars_doc.pdf",
        "file_hash": "hash_special_chars_007",
        "class_section": "SEC-404 & <Dev>",
        "student_name": 'O\'Connor-Smith, Jr. & "Co."',
        "assignment_title": "HW #1: AI / Machine Learning (100% Final!)",
        "pdf_author": "Author & Co. 'Special'",
        "pdf_title": "Paper: 'Deep Learning & Neural Nets'",
        "detected_language": "en",
    },
    # 8. Minimal metadata (optional fields set to None or omitted)
    {
        "filename": "minimal_metadata_only.pdf",
        "file_hash": "hash_minimal_008",
    },
    # 9. Rich PDF metadata (author, title, creation date, tags)
    {
        "filename": "rich_pdf_meta.pdf",
        "file_hash": "hash_rich_pdf_009",
        "class_section": "MATH-301",
        "student_name": "Alan Turing",
        "assignment_title": "On Computable Numbers",
        "pdf_author": "Dr. Alan M. Turing",
        "pdf_creation_date": "1936-11-12T00:00:00Z",
        "pdf_title": "On Computable Numbers with an Application to the Entscheidungsproblem",
        "tags": "math,computation,logic",
        "detected_language": "en",
    },
    # 10. Long student name and long assignment title
    {
        "filename": "long_fields_record.pdf",
        "file_hash": "hash_long_fields_010",
        "class_section": "SEC-" + "Y" * 40,
        "student_name": "Sir " + "Alexander " * 10,
        "assignment_title": "Comprehensive Review of " + "Algorithms " * 10,
        "detected_language": "en",
    },
]


@pytest.mark.parametrize("doc_metadata", DIVERSE_METADATA_CASES)
def test_add_documents_bulk_parametrized(doc_metadata):
    """Parametrized test verifying document insertion and retrieval for diverse metadata combinations.

    Validates unicode student names (Japanese, Cyrillic, Arabic, French accents),
    empty assignments, long filenames, special characters, and minimal records.
    Asserts all fields are stored and retrieved with exact field values.
    """
    inserted_count = add_documents_bulk([doc_metadata])
    assert inserted_count == 1

    all_docs = get_all_documents()
    assert len(all_docs) == 1

    stored = all_docs[0]
    assert stored["filename"] == doc_metadata["filename"]
    assert stored["file_hash"] == doc_metadata["file_hash"]

    # Verify optional metadata fields match inputs exactly
    assert stored.get("class_section") == doc_metadata.get("class_section")
    assert stored.get("student_name") == doc_metadata.get("student_name")
    assert stored.get("assignment_title") == doc_metadata.get("assignment_title")
    assert stored.get("pdf_author") == doc_metadata.get("pdf_author")
    assert stored.get("pdf_title") == doc_metadata.get("pdf_title")
    assert stored.get("detected_language") == doc_metadata.get("detected_language")


def test_add_documents_bulk_all_ten_batch():
    """Verify inserting all 10 diverse metadata records in a single bulk call."""
    inserted_count = add_documents_bulk(DIVERSE_METADATA_CASES)
    assert inserted_count == len(DIVERSE_METADATA_CASES)

    all_docs = get_all_documents()
    assert len(all_docs) == len(DIVERSE_METADATA_CASES)

    retrieved_by_hash = {d["file_hash"]: d for d in all_docs}

    for input_doc in DIVERSE_METADATA_CASES:
        h = input_doc["file_hash"]
        assert h in retrieved_by_hash
        stored = retrieved_by_hash[h]

        assert stored["filename"] == input_doc["filename"]
        assert stored.get("class_section") == input_doc.get("class_section")
        assert stored.get("student_name") == input_doc.get("student_name")
        assert stored.get("assignment_title") == input_doc.get("assignment_title")
        assert stored.get("pdf_author") == input_doc.get("pdf_author")
        assert stored.get("pdf_title") == input_doc.get("pdf_title")
        assert stored.get("detected_language") == input_doc.get("detected_language")
