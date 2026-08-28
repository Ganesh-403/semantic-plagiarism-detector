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
tests/db/test_citation_db.py
---------------------------
Tests for citation database operations, duplicate insertion tracking,
and Jaccard similarity edge cases.
"""

import os

import pytest

from src.db.citation_db import (
    add_document_citations,
    compute_citation_jaccard,
    init_citation_db,
)
from src.db.corpus_db import _DB_PATH


@pytest.fixture(autouse=True)
def setup_teardown_db():
    # Setup test database
    if os.path.exists(_DB_PATH):
        os.remove(_DB_PATH)

    init_citation_db()
    yield
    # Teardown test database
    if os.path.exists(_DB_PATH):
        try:
            os.remove(_DB_PATH)
        except PermissionError:
            pass


def test_add_document_citations_duplicate_count():
    doc_name = "test_doc.pdf"
    citations = [
        {
            "hash": "hash123",
            "author": "Smith",
            "year": "2023",
            "title": "A Great Paper",
            "raw_text": "Smith, 2023, A Great Paper",
        }
    ]

    # First insert should return 1
    added = add_document_citations(doc_name, citations)
    assert added == 1

    # Second insert of the exact same citation should return 0
    added_duplicate = add_document_citations(doc_name, citations)
    assert added_duplicate == 0


def test_jaccard_both_empty():
    """Test test_jaccard_both_empty — assert returns 0.0 when both sets are empty."""
    score = compute_citation_jaccard("doc_empty_1.pdf", "doc_empty_2.pdf")
    assert score == 0.0


def test_jaccard_one_empty():
    """Test test_jaccard_one_empty — assert returns 0.0 when one set has citations and the other is empty."""
    citations = [
        {
            "hash": "hash1",
            "author": "Author A",
            "year": "2023",
            "title": "Paper One",
            "raw_text": "Ref 1",
        }
    ]
    add_document_citations("doc_populated.pdf", citations)

    score = compute_citation_jaccard("doc_populated.pdf", "doc_empty.pdf")
    assert score == 0.0


def test_jaccard_identical_sets():
    """Test test_jaccard_identical_sets — assert returns 1.0 for identical citation sets."""
    citations = [
        {
            "hash": "hash1",
            "author": "Author A",
            "year": "2023",
            "title": "Paper One",
            "raw_text": "Ref 1",
        },
        {
            "hash": "hash2",
            "author": "Author B",
            "year": "2024",
            "title": "Paper Two",
            "raw_text": "Ref 2",
        },
    ]
    add_document_citations("doc_a.pdf", citations)
    add_document_citations("doc_b.pdf", citations)

    score = compute_citation_jaccard("doc_a.pdf", "doc_b.pdf")
    assert score == 1.0


def test_jaccard_partial_overlap():
    """Test test_jaccard_partial_overlap — assert returns expected fraction for overlapping sets."""
    cit1 = {
        "hash": "hash1",
        "author": "Author A",
        "year": "2023",
        "title": "Paper One",
        "raw_text": "Ref 1",
    }
    cit2 = {
        "hash": "hash2",
        "author": "Author B",
        "year": "2024",
        "title": "Paper Two",
        "raw_text": "Ref 2",
    }
    cit3 = {
        "hash": "hash3",
        "author": "Author C",
        "year": "2025",
        "title": "Paper Three",
        "raw_text": "Ref 3",
    }

    add_document_citations("doc_a.pdf", [cit1, cit2])
    add_document_citations("doc_b.pdf", [cit2, cit3])

    score = compute_citation_jaccard("doc_a.pdf", "doc_b.pdf")
    # Intersection = {hash2} (1), Union = {hash1, hash2, hash3} (3) -> 1/3 ≈ 0.333333
    assert abs(score - (1 / 3)) < 1e-6
