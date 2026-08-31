"""
Unit tests for the Hybrid Similarity Analysis Pipeline
"""

import pytest
import os
import tempfile
from pathlib import Path
import json

from src.models.similarity import (
    MatchResult, MatchSeverity, AnalysisResult,
    SimilarityConfig, SimilarityType, DocumentPair
)
from src.analysis.similarity_metrics import SimilarityMetrics
from src.analysis.lexical_analyzer import LexicalAnalyzer
from src.analysis.semantic_analyzer import SemanticAnalyzer
from src.analysis.hybrid_analyzer import HybridAnalyzer


class TestSimilarityMetrics:
    """Test similarity metrics calculations."""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        result = SimilarityMetrics.cosine_similarity(vec1, vec2)
        assert result == 1.0

        vec2 = [0.0, 1.0, 0.0]
        result = SimilarityMetrics.cosine_similarity(vec1, vec2)
        assert result == 0.0

        vec1 = [1.0, 1.0, 0.0]
        vec2 = [1.0, 1.0, 0.0]
        result = SimilarityMetrics.cosine_similarity(vec1, vec2)
        assert result == 1.0

    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        set1 = {1, 2, 3}
        set2 = {1, 2, 4}
        result = SimilarityMetrics.jaccard_similarity(set1, set2)
        assert result == 0.5

        set1 = {1, 2, 3}
        set2 = {4, 5, 6}
        result = SimilarityMetrics.jaccard_similarity(set1, set2)
        assert result == 0.0

        set1 = {1, 2, 3}
        set2 = {1, 2, 3}
        result = SimilarityMetrics.jaccard_similarity(set1, set2)
        assert result == 1.0

    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation."""
        result = SimilarityMetrics.levenshtein_distance("kitten", "sitting")
        assert result == 3

        result = SimilarityMetrics.levenshtein_distance("hello", "hello")
        assert result == 0

        result = SimilarityMetrics.levenshtein_distance("", "hello")
        assert result == 5

    def test_levenshtein_similarity(self):
        """Test Levenshtein similarity ratio."""
        result = SimilarityMetrics.levenshtein_similarity("kitten", "sitting")
        assert 0.5 < result < 0.6

        result = SimilarityMetrics.levenshtein_similarity("hello", "hello")
        assert result == 1.0

        result = SimilarityMetrics.levenshtein_similarity("", "hello")
        assert result == 0.0

    def test_ngram_similarity(self):
        """Test n-gram similarity calculation."""
        result = SimilarityMetrics.ngram_similarity("hello world", "hello world", n=3)
        assert result == 1.0

        result = SimilarityMetrics.ngram_similarity("hello world", "goodbye world", n=3)
        assert 0.3 < result < 0.5

        result = SimilarityMetrics.ngram_similarity("hello", "", n=3)
        assert result == 0.0

    def test_lcs_similarity(self):
        """Test LCS similarity calculation."""
        result = SimilarityMetrics.lcs_similarity("hello", "hello")
        assert result == 1.0

        result = SimilarityMetrics.lcs_similarity("hello", "hallo")
        assert 0.8 < result < 0.9

        result = SimilarityMetrics.lcs_similarity("abc", "def")
        assert result == 0.0

    def test_dice_similarity(self):
        """Test Dice similarity calculation."""
        set1 = {1, 2, 3, 4}
        set2 = {3, 4, 5, 6}
        result = SimilarityMetrics.dice_similarity(set1, set2)
        assert result == 0.5

    def test_overlap_coefficient(self):
        """Test overlap coefficient calculation."""
        set1 = {1, 2, 3, 4}
        set2 = {3, 4, 5, 6}
        result = SimilarityMetrics.overlap_coefficient(set1, set2)
        assert result == 0.5

    def test_combine_scores(self):
        """Test score combination."""
        scores = [0.8, 0.6, 0.9]
        result = SimilarityMetrics.combine_scores(scores)
        expected = (0.8 + 0.6 + 0.9) / 3
        assert result == expected

        weights = [0.5, 0.3, 0.2]
        result = SimilarityMetrics.combine_scores(scores, weights)
        expected = 0.8 * 0.5 + 0.6 * 0.3 + 0.9 * 0.2
        assert result == expected

    def test_confidence_score(self):
        """Test confidence score calculation."""
        result = SimilarityMetrics.confidence_score(0.8, 5)
        assert 0.8 <= result <= 0.85

        result = SimilarityMetrics.confidence_score(0.5, 0)
        assert result == 0.5


class TestLexicalAnalyzer:
    """Test lexical analyzer."""

    def setup_method(self):
        self.analyzer = LexicalAnalyzer()

    def test_preprocess(self):
        """Test text preprocessing."""
        text = "Hello World! This is a test."
        result = self.analyzer._preprocess(text)
        assert "hello" in result
        assert "world" in result
        assert "test" in result

        text = "123 Testing!@#$ Special chars."
        result = self.analyzer._preprocess(text)
        assert "testing" in result
        assert "special" in result
        assert "chars" in result

    def test_chunk_text(self):
        """Test text chunking."""
        text = " ".join(["word"] * 100)
        chunks = self.analyzer._chunk_text(text)
        assert len(chunks) == 1

        text = " ".join(["word"] * 500)
        chunks = self.analyzer._chunk_text(text)
        assert len(chunks) > 1

    def test_compare_documents_identical(self):
        """Test comparing identical documents."""
        text = "This is a test document for plagiarism detection."
        score, matches = self.analyzer.compare_documents(text, text)
        assert score >= 0.9

    def test_compare_documents_different(self):
        """Test comparing different documents."""
        doc1 = "This is about artificial intelligence and machine learning."
        doc2 = "This is about cooking recipes and food preparation."
        score, matches = self.analyzer.compare_documents(doc1, doc2)
        assert score < 0.5

    def test_compare_documents_partial(self):
        """Test comparing partially similar documents."""
        doc1 = "The quick brown fox jumps over the lazy dog."
        doc2 = "The quick brown fox jumps over the lazy dog. This is extra text."
        score, matches = self.analyzer.compare_documents(doc1, doc2)
        assert score >= 0.7

    def test_batch_compare(self):
        """Test batch comparison."""
        source = "This is the source document."
        targets = [
            {'id': 'doc1', 'content': 'This is the source document.'},
            {'id': 'doc2', 'content': 'This is something completely different.'}
        ]
        results = self.analyzer.batch_compare(source, targets)
        assert len(results) == 2
        assert results[0]['score'] >= results[1]['score']


class TestSemanticAnalyzer:
    """Test semantic analyzer."""

    def setup_method(self):
        self.analyzer = SemanticAnalyzer()

    def test_chunk_sentences(self):
        """Test sentence chunking."""
        text = "Hello world. How are you? I am fine."
        sentences = self.analyzer._chunk_sentences(text)
        assert len(sentences) == 3

    def test_chunk_paragraphs(self):
        """Test paragraph chunking."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        paragraphs = self.analyzer._chunk_paragraphs(text)
        assert len(paragraphs) == 3

    def test_model_info(self):
        """Test model info."""
        info = self.analyzer.get_model_info()
        assert 'loaded' in info
        assert 'model_name' in info


class TestHybridAnalyzer:
    """Test hybrid analyzer."""

    def setup_method(self):
        self.analyzer = HybridAnalyzer()
        self.test_doc1 = "This is a test document for plagiarism detection."
        self.test_doc2 = "This is a test document for plagiarism detection."
        self.test_doc3 = "This is something completely different."

    def test_analyze_pair_identical(self):
        """Test analyzing identical documents."""
        result = self.analyzer.analyze_pair(
            self.test_doc1, self.test_doc2,
            "source1", "target1"
        )
        assert len(result.matches) == 1
        assert result.matches[0].hybrid_score >= 0.8
        assert result.matches[0].severity == MatchSeverity.HIGH
        assert result.summary['threshold_met'] is True

    def test_analyze_pair_different(self):
        """Test analyzing different documents."""
        result = self.analyzer.analyze_pair(
            self.test_doc1, self.test_doc3,
            "source1", "target2"
        )
        assert len(result.matches) == 1
        assert result.matches[0].hybrid_score < 0.5
        assert result.matches[0].severity == MatchSeverity.NONE
        assert result.summary['threshold_met'] is False

    def test_analyze_batch(self):
        """Test batch analysis."""
        targets = [
            {'id': 'doc1', 'content': self.test_doc2},
            {'id': 'doc2', 'content': self.test_doc3}
        ]
        result = self.analyzer.analyze_batch(self.test_doc1, targets)
        assert len(result.matches) == 2
        assert result.matches[0].hybrid_score >= result.matches[1].hybrid_score

    def test_analyze_with_threshold(self):
        """Test analysis with custom threshold."""
        threshold_met, result = self.analyzer.analyze_with_threshold(
            self.test_doc1, self.test_doc2, threshold=0.7
        )
        assert threshold_met is True

        threshold_met, result = self.analyzer.analyze_with_threshold(
            self.test_doc1, self.test_doc3, threshold=0.7
        )
        assert threshold_met is False

    def test_get_recommendations(self):
        """Test recommendation generation."""
        recommendations = self.analyzer.get_recommendations(
            self.test_doc1, self.test_doc2
        )
        assert 'recommendations' in recommendations
        assert 'action_required' in recommendations
        assert recommendations['action_required'] is True

        recommendations = self.analyzer.get_recommendations(
            self.test_doc1, self.test_doc3
        )
        assert 'recommendations' in recommendations
        assert recommendations['action_required'] is False

    def test_analysis_summary(self):
        """Test analysis summary generation."""
        result = self.analyzer.analyze_pair(
            self.test_doc1, self.test_doc2,
            "source1", "target1"
        )
        summary = self.analyzer.get_analysis_summary(result)
        assert 'analysis_id' in summary
        assert 'source_document' in summary
        assert 'target_documents' in summary
        assert 'matches_count' in summary
        assert 'processing_time_ms' in summary


class TestMatchResult:
    """Test match result class."""

    def test_get_severity(self):
        """Test severity calculation."""
        match = MatchResult(hybrid_score=0.9)
        assert match.get_severity() == MatchSeverity.HIGH

        match = MatchResult(hybrid_score=0.7)
        assert match.get_severity() == MatchSeverity.MEDIUM

        match = MatchResult(hybrid_score=0.5)
        assert match.get_severity() == MatchSeverity.LOW

        match = MatchResult(hybrid_score=0.3)
        assert match.get_severity() == MatchSeverity.NONE

    def test_to_dict(self):
        """Test conversion to dictionary."""
        match = MatchResult(
            source_document="source1",
            target_document="target1",
            lexical_score=0.8,
            semantic_score=0.9,
            hybrid_score=0.85,
            matched_text="This is a test"
        )
        result = match.to_dict()
        assert 'source_document' in result
        assert 'target_document' in result
        assert 'lexical_score' in result
        assert 'semantic_score' in result
        assert 'hybrid_score' in result
        assert 'severity' in result


class TestAnalysisResult:
    """Test analysis result class."""

    def test_add_match(self):
        """Test adding matches."""
        result = AnalysisResult()
        match = MatchResult()
        result.add_match(match)
        assert len(result.matches) == 1

    def test_mark_completed(self):
        """Test marking as completed."""
        result = AnalysisResult()
        result.mark_completed()
        assert result.status.value == "completed"

    def test_mark_failed(self):
        """Test marking as failed."""
        result = AnalysisResult()
        result.mark_failed()
        assert result.status.value == "failed"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AnalysisResult(
            source_document_id="source1",
            target_document_ids=["target1"]
        )
        result.add_match(MatchResult())
        result.mark_completed()
        dict_result = result.to_dict()
        assert 'id' in dict_result
        assert 'source_document_id' in dict_result
        assert 'target_document_ids' in dict_result
        assert 'matches' in dict_result
        assert 'status' in dict_result


class TestSimilarityConfig:
    """Test similarity configuration."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SimilarityConfig()
        result = config.to_dict()
        assert 'lexical_weight' in result
        assert 'semantic_weight' in result
        assert 'lexical_threshold' in result
        assert 'semantic_threshold' in result
        assert 'hybrid_threshold' in result

    def test_default_values(self):
        """Test default configuration values."""
        config = SimilarityConfig()
        assert config.lexical_weight == 0.3
        assert config.semantic_weight == 0.7
        assert config.lexical_threshold == 0.3
        assert config.semantic_threshold == 0.4
        assert config.hybrid_threshold == 0.5


class TestDocumentPair:
    """Test document pair class."""

    def test_is_valid(self):
        """Test document pair validation."""
        pair = DocumentPair(
            source_content="Hello",
            target_content="World"
        )
        assert pair.is_valid() is True

        pair = DocumentPair(
            source_content="",
            target_content="World"
        )
        assert pair.is_valid() is False

        pair = DocumentPair(
            source_content="Hello",
            target_content=""
        )
        assert pair.is_valid() is False