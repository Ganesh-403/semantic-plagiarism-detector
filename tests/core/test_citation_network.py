"""
tests/core/test_citation_network.py
-----------------------------------
Unit tests for the citation extraction and network analysis engine.
"""

import pytest
from src.core.citation_extractor import (
    extract_citations,
    compute_jaccard_similarity,
    Citation,
)
from src.db.citation_graph_db import (
    initialize_citation_db,
    ingest_citations,
    get_document_citation_keys,
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
