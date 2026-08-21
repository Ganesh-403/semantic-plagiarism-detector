"""
Tests for AI-Powered Plagiarism Scoring Engine.

Comprehensive test suite covering scoring components, fingerprinting,
severity classification, and batch processing.
"""

import pytest
import numpy as np
from src.core.ai_scoring_engine import (
    AIScoringEngine,
    ContentFingerprinter,
    ScoringConfig,
    PlagiarismScore,
    SeverityLevel,
    ScoringMethod,
    ContentFingerprint,
)


class TestContentFingerprinter:
    """Tests for content fingerprinting."""

    def setup_method(self):
        self.config = ScoringConfig(shingle_size=3, minhash_num_perm=32)
        self.fingerprinter = ContentFingerprinter(self.config)

    def test_create_fingerprint(self):
        """Test fingerprint creation."""
        fp = self.fingerprinter.create_fingerprint(
            "This is a test document with some content.", "test.txt"
        )
        assert isinstance(fp, ContentFingerprint)
        assert fp.doc_name == "test.txt"
        assert len(fp.shingles) > 0
        assert len(fp.minhash_signature) == 32

    def test_identical_texts(self):
        """Test identical texts produce similar fingerprints."""
        text = "Machine learning is a branch of artificial intelligence that focuses on building systems."
        fp1 = self.fingerprinter.create_fingerprint(text, "a.txt")
        fp2 = self.fingerprinter.create_fingerprint(text, "b.txt")
        sim = self.fingerprinter.compare_fingerprints(fp1, fp2)
        assert sim == 1.0

    def test_different_texts(self):
        """Test different texts produce lower similarity."""
        fp1 = self.fingerprinter.create_fingerprint(
            "Machine learning algorithms analyze large datasets.", "a.txt"
        )
        fp2 = self.fingerprinter.create_fingerprint(
            "Cooking recipes for delicious pasta dishes.", "b.txt"
        )
        sim = self.fingerprinter.compare_fingerprints(fp1, fp2)
        assert sim < 0.5

    def test_empty_text(self):
        """Test empty text handling."""
        fp = self.fingerprinter.create_fingerprint("", "empty.txt")
        assert len(fp.shingles) == 0

    def test_near_duplicates(self):
        """Test near-duplicate detection."""
        base = (
            "This is a long document about artificial intelligence and machine learning. "
            * 5
        )
        modified = base + " Some additional content added at the end."
        fp1 = self.fingerprinter.create_fingerprint(base, "a.txt")
        fp2 = self.fingerprinter.create_fingerprint(modified, "b.txt")
        sim = self.fingerprinter.compare_fingerprints(fp1, fp2)
        assert sim > 0.7


class TestScoringComponents:
    """Tests for individual scoring components."""

    def setup_method(self):
        self.engine = AIScoringEngine(ScoringConfig(enable_fingerprint=False))

    def test_semantic_score(self):
        """Test semantic scoring."""
        score = self.engine.compute_semantic_score(
            "Machine learning algorithms process large datasets efficiently.",
            "Machine learning systems analyze big data effectively.",
        )
        assert 0 <= score.score <= 1
        assert score.method == "semantic"
        assert score.confidence > 0

    def test_lexical_score(self):
        """Test lexical scoring."""
        score = self.engine.compute_lexical_score(
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox leaps over the lazy dog.",
        )
        assert 0 <= score.score <= 1
        assert score.method == "lexical"

    def test_structural_score(self):
        """Test structural scoring."""
        text_a = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        text_b = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        score = self.engine.compute_structural_score(text_a, text_b)
        assert 0 <= score.score <= 1
        assert score.method == "structural"

    def test_statistical_score(self):
        """Test statistical scoring."""
        text_a = "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks."
        text_b = "Machine learning is a branch of artificial intelligence. Neural networks enable deep learning."
        score = self.engine.compute_statistical_score(text_a, text_b)
        assert 0 <= score.score <= 1
        assert score.method == "statistical"

    def test_identical_texts_high_score(self):
        """Test identical texts produce high scores."""
        text = "Artificial intelligence transforms industries through automation and data analysis."
        semantic = self.engine.compute_semantic_score(text, text)
        lexical = self.engine.compute_lexical_score(text, text)
        assert semantic.score > 0.9
        assert lexical.score > 0.9


class TestEnsembleScoring:
    """Tests for ensemble scoring."""

    def setup_method(self):
        self.engine = AIScoringEngine(ScoringConfig(enable_fingerprint=False))

    def test_ensemble_combines_components(self):
        """Test ensemble scoring combines all components."""
        text_a = "This is a test document about machine learning and artificial intelligence."
        text_b = "This is a test document about machine learning and artificial intelligence."
        score = self.engine.score_documents(text_a, text_b)
        assert isinstance(score, PlagiarismScore)
        assert score.overall_score > 0.8
        assert score.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]

    def test_different_documents_lower_score(self):
        """Test different documents produce lower scores."""
        text_a = "Machine learning algorithms analyze large datasets for pattern recognition."
        text_b = "Cooking pasta requires boiling water and adding salt for flavor."
        score = self.engine.score_documents(text_a, text_b)
        assert score.overall_score < 0.5


class TestSeverityClassification:
    """Tests for severity classification."""

    def setup_method(self):
        self.engine = AIScoringEngine()

    def test_severity_levels(self):
        """Test all severity levels."""
        assert self.engine.determine_severity(0.0) == SeverityLevel.CLEAN
        assert self.engine.determine_severity(0.3) == SeverityLevel.LOW
        assert self.engine.determine_severity(0.5) == SeverityLevel.MODERATE
        assert self.engine.determine_severity(0.7) == SeverityLevel.HIGH
        assert self.engine.determine_severity(0.9) == SeverityLevel.CRITICAL

    def test_boundary_values(self):
        """Test threshold boundaries."""
        assert self.engine.determine_severity(0.20) == SeverityLevel.LOW
        assert self.engine.determine_severity(0.40) == SeverityLevel.MODERATE
        assert self.engine.determine_severity(0.60) == SeverityLevel.HIGH
        assert self.engine.determine_severity(0.80) == SeverityLevel.CRITICAL


class TestBatchScoring:
    """Tests for batch scoring."""

    def setup_method(self):
        self.engine = AIScoringEngine(ScoringConfig(enable_fingerprint=False))

    def test_batch_score(self):
        """Test batch scoring of multiple documents."""
        docs = {
            "doc1.txt": "Machine learning is a subset of AI.",
            "doc2.txt": "Deep learning is a subset of machine learning.",
            "doc3.txt": "Cooking pasta requires boiling water.",
        }
        scores = self.engine.batch_score(docs)
        assert len(scores) == 3
        assert all(isinstance(s, PlagiarismScore) for s in scores)
        assert scores[0].overall_score >= scores[-1].overall_score

    def test_batch_sorted_by_score(self):
        """Test batch results are sorted by score."""
        docs = {
            "a.txt": "Artificial intelligence and machine learning.",
            "b.txt": "Artificial intelligence and machine learning.",
            "c.txt": "Cooking recipes for dinner.",
        }
        scores = self.engine.batch_score(docs)
        for i in range(len(scores) - 1):
            assert scores[i].overall_score >= scores[i + 1].overall_score


class TestResultSerialization:
    """Tests for result serialization."""

    def test_score_to_dict(self):
        """Test score to_dict."""
        engine = AIScoringEngine(ScoringConfig(enable_fingerprint=False))
        score = engine.score_documents("Test text A", "Test text B")
        d = score.to_dict()
        assert "doc_a" in d
        assert "overall_score" in d
        assert "severity" in d
        assert isinstance(d["components"], list)


class TestConfiguration:
    """Tests for configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = ScoringConfig()
        assert "semantic" in config.weights
        assert config.shingle_size == 5
        assert config.enable_fingerprint is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ScoringConfig(shingle_size=8, enable_fingerprint=False)
        assert config.shingle_size == 8
        assert config.enable_fingerprint is False
