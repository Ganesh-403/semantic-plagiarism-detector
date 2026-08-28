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
tests/core/test_citation_network.py
-----------------------------------
Unit tests for the citation extraction and network analysis engine.
"""

import pytest

from src.core.citation_extractor import (
    Citation,
    compute_jaccard_similarity,
    extract_citations,
)
from src.db.citation_graph_db import (
    get_document_citation_keys,
    ingest_citations,
    initialize_citation_db,
)


class TestCitationExtractor:
    """Test suite for regex-based citation parsing."""

    def test_extract_apa_format(self):
        """Verify APA format citations are parsed correctly."""
        text = "Smith, J. (2020). The art of plagiarism. Journal of AI, 12(3), 45-60."
        citations = extract_citations(text)

        assert len(citations) == 1
        assert citations[0].authors == "Smith, J."
        assert citations[0].year == "2020"
        assert "plagiarism" in citations[0].title.lower()
        assert citations[0].format_detected == "APA"

    def test_extract_mla_format(self):
        """Verify MLA format citations are parsed correctly."""
        text = 'Doe, Jane. "Detecting Ghostwriting." Tech Review, 2021.'
        citations = extract_citations(text)

        assert len(citations) == 1
        assert citations[0].authors == "Doe, Jane"
        assert citations[0].year == "2021"
        assert citations[0].format_detected == "MLA"

    def test_jaccard_similarity_identical(self):
        """Verify identical sets have a Jaccard similarity of 1.0."""
        set_a = {"cite_1", "cite_2"}
        set_b = {"cite_1", "cite_2"}
        assert compute_jaccard_similarity(set_a, set_b) == 1.0

    def test_jaccard_similarity_disjoint(self):
        """Verify disjoint sets have a Jaccard similarity of 0.0."""
        set_a = {"cite_1"}
        set_b = {"cite_2"}
        assert compute_jaccard_similarity(set_a, set_b) == 0.0


class TestCitationGraphDB:
    """Test suite for the citation graph database."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        db_path = tmp_path / "test_cite.db"
        initialize_citation_db(db_path)
        return db_path

    def test_ingest_and_retrieve(self, temp_db):
        """Verify citations can be ingested and retrieved."""
        cite = Citation(
            raw_text="Test",
            authors="Author",
            year="2023",
            title="Title",
            source="Source",
            format_detected="APA",
        )

        ingest_citations("doc_1", [cite], db_path=temp_db)
        keys = get_document_citation_keys("doc_1", db_path=temp_db)

        assert len(keys) == 1
        assert cite.get_normalized_key() in keys
