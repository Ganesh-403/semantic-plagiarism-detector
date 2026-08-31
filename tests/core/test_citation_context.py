"""
tests/core/test_citation_context.py
-----------------------------------
Unit tests for Citation Context and Semantic Alignment Analysis.
"""

import pytest
from src.core.citation_context_analyzer import (
    extract_citation_contexts,
    map_citations_to_references,
)
from src.core.semantic_citation_aligner import (
    compute_tf_idf_cosine_similarity,
    analyze_citation_alignment,
)


class TestCitationContextAnalyzer:
    """Test suite for citation context extraction."""

    def test_extract_numeric_citations(self):
        """Verify numeric citations like [1] are extracted."""
        text = "This is a sentence [1]. Another sentence."
        contexts = extract_citation_contexts(text)
        assert len(contexts) == 1
        assert contexts[0]["citation_id"] == "1"

    def test_extract_author_year_citations(self):
        """Verify author-year citations like (Smith, 2020) are extracted."""
        text = "As shown by (Smith, 2020), this is true."
        contexts = extract_citation_contexts(text)
        assert len(contexts) == 1
        assert "Smith" in contexts[0]["citation_id"]

    def test_map_citations_to_references(self):
        """Verify contexts are mapped to reference abstracts."""
        contexts = [{"citation_id": "1", "context_text": "Text"}]
        refs = {"1": "Abstract text"}
        mapped = map_citations_to_references(contexts, refs)
        assert mapped[0]["reference_abstract"] == "Abstract text"


class TestSemanticCitationAligner:
    """Test suite for semantic alignment scoring."""

    def test_cosine_similarity_identical(self):
        """Verify identical texts have similarity 1.0."""
        text = "The cat sat on the mat."
        sim = compute_tf_idf_cosine_similarity(text, text)
        assert sim == 1.0

    def test_cosine_similarity_disjoint(self):
        """Verify disjoint texts have low similarity."""
        text_a = "The cat sat on the mat."
        text_b = "Quantum physics is complex."
        sim = compute_tf_idf_cosine_similarity(text_a, text_b)
        assert sim < 0.2

    def test_analyze_citation_alignment_bluffing(self):
        """Verify mismatched context and abstract are flagged as bluffing."""
        mapped = [
            {
                "citation_id": "1",
                "context_text": "The cat sat on the mat.",
                "reference_abstract": "Quantum physics is complex.",
            }
        ]
        results = analyze_citation_alignment(mapped, bluffing_threshold=0.2)
        assert results[0]["is_bluffing"] is True
