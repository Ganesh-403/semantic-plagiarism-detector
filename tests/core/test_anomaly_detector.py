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
Tests for Plagiarism Anomaly Detection Engine.

Comprehensive test suite covering statistical analysis, cluster detection,
pattern analysis, and anomaly classification.
"""

import numpy as np

from src.core.anomaly_detector import (
    Anomaly,
    AnomalyConfig,
    AnomalyDetector,
    AnomalyResult,
    AnomalySeverity,
    AnomalyType,
    ClusterAnalyzer,
    PatternAnalyzer,
    StatisticalAnalyzer,
)


class TestStatisticalAnalyzer:
    """Tests for statistical analysis."""

    def setup_method(self):
        self.config = AnomalyConfig(z_score_threshold=2.0)
        self.analyzer = StatisticalAnalyzer(self.config)

    def test_z_score_normal(self):
        """Test Z-score with normal data."""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        anomalies = self.analyzer.z_score_analysis(scores)
        assert len(anomalies) == 0

    def test_z_score_with_outlier(self):
        """Test Z-score detects outlier."""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.95]
        anomalies = self.analyzer.z_score_analysis(scores)
        assert len(anomalies) >= 1

    def test_z_score_insufficient_data(self):
        """Test Z-score with insufficient data."""
        scores = [0.5, 0.6]
        anomalies = self.analyzer.z_score_analysis(scores)
        assert len(anomalies) == 0

    def test_iqr_analysis(self):
        """Test IQR analysis."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2.0]
        anomalies = self.analyzer.iqr_analysis(scores)
        assert len(anomalies) >= 1

    def test_percentile_analysis(self):
        """Test percentile computation."""
        scores = list(np.random.rand(100))
        percentiles = self.analyzer.percentile_analysis(scores)
        assert "p50" in percentiles
        assert "p95" in percentiles

    def test_distribution_anomaly(self):
        """Test distribution shape detection."""
        scores = [0.1] * 90 + [0.9] * 10
        anomaly = self.analyzer.detect_distribution_anomaly(scores)
        assert anomaly is not None
        assert anomaly.anomaly_type == AnomalyType.STATISTICAL


class TestClusterAnalyzer:
    """Tests for cluster analysis."""

    def setup_method(self):
        self.config = AnomalyConfig(
            cluster_min_size=2, cluster_similarity_threshold=0.8
        )
        self.analyzer = ClusterAnalyzer(self.config)

    def test_find_clusters(self):
        """Test cluster finding."""
        doc_names = ["a", "b", "c", "d"]
        sim_matrix = np.array(
            [
                [1.0, 0.9, 0.3, 0.3],
                [0.9, 1.0, 0.3, 0.3],
                [0.3, 0.3, 1.0, 0.9],
                [0.3, 0.3, 0.9, 1.0],
            ]
        )
        clusters = self.analyzer.find_similarity_clusters(doc_names, sim_matrix)
        assert len(clusters) >= 1

    def test_no_clusters(self):
        """Test when no clusters exist."""
        doc_names = ["a", "b", "c"]
        sim_matrix = np.array(
            [
                [1.0, 0.3, 0.3],
                [0.3, 1.0, 0.3],
                [0.3, 0.3, 1.0],
            ]
        )
        clusters = self.analyzer.find_similarity_clusters(doc_names, sim_matrix)
        assert len(clusters) == 0

    def test_collusion_detection(self):
        """Test collusion cluster detection."""
        clusters = [{"documents": ["a", "b", "c"], "size": 3, "avg_similarity": 0.92}]
        anomalies = self.analyzer.detect_collusion_clusters(clusters)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.COLLUSION


class TestPatternAnalyzer:
    """Tests for pattern analysis."""

    def setup_method(self):
        self.config = AnomalyConfig(pattern_min_length=3)
        self.analyzer = PatternAnalyzer(self.config)

    def test_find_repeated_phrases(self):
        """Test repeated phrase detection."""
        docs = {
            "doc1": "the cat sat on the mat the dog ran the cat sat",
            "doc2": "the cat sat on the mat the bird flew the cat sat",
        }
        repeated = self.analyzer.find_repeated_phrases(docs, min_length=3)
        assert len(repeated) > 0

    def test_copy_pattern_detection(self):
        """Test copy pattern detection."""
        docs = {
            "doc1": "the quick brown fox jumps over the lazy dog quickly",
            "doc2": "the quick brown fox jumps over the lazy dog quickly",
        }
        anomalies = self.analyzer.detect_copy_patterns(docs)
        assert len(anomalies) >= 1

    def test_no_copy_patterns(self):
        """Test when no copy patterns exist."""
        docs = {
            "doc1": "machine learning algorithms process data efficiently",
            "doc2": "cooking recipes for delicious pasta dishes tonight",
        }
        anomalies = self.analyzer.detect_copy_patterns(docs)
        assert len(anomalies) == 0


class TestAnomalyDetector:
    """Tests for the main anomaly detector."""

    def setup_method(self):
        self.config = AnomalyConfig(
            z_score_threshold=2.0,
            cluster_min_size=2,
            cluster_similarity_threshold=0.8,
            enable_statistical=True,
            enable_cluster=True,
            enable_pattern=True,
        )
        self.detector = AnomalyDetector(self.config)

    def test_basic_detection(self):
        """Test basic anomaly detection."""
        docs = {
            "doc1": "Machine learning is a subset of artificial intelligence.",
            "doc2": "Machine learning is a subset of artificial intelligence.",
            "doc3": "Cooking pasta requires boiling water and adding salt.",
        }
        result = self.detector.detect(docs)
        assert isinstance(result, AnomalyResult)
        assert result.summary["documents_analyzed"] == 3

    def test_with_similarity_matrix(self):
        """Test with similarity matrix."""
        docs = {"a": "text a", "b": "text b", "c": "text c", "d": "text d"}
        sim_matrix = np.array(
            [
                [1.0, 0.9, 0.3, 0.3],
                [0.9, 1.0, 0.3, 0.3],
                [0.3, 0.3, 1.0, 0.9],
                [0.3, 0.3, 0.9, 1.0],
            ]
        )
        result = self.detector.detect(docs, similarity_matrix=sim_matrix)
        assert isinstance(result, AnomalyResult)

    def test_with_similarity_scores(self):
        """Test with similarity scores."""
        docs = {"a": "text a", "b": "text b"}
        scores = [0.95, 0.3, 0.3, 0.5, 0.5, 0.5]
        result = self.detector.detect(docs, similarity_scores=scores)
        assert isinstance(result, AnomalyResult)

    def test_recommendations_generated(self):
        """Test recommendations are generated."""
        docs = {"a": "text a", "b": "text b"}
        result = self.detector.detect(docs)
        assert isinstance(result.recommendations, list)


class TestAnomalyDataClasses:
    """Tests for data classes."""

    def test_anomaly_to_dict(self):
        """Test anomaly serialization."""
        anomaly = Anomaly(
            anomaly_id="T-001",
            anomaly_type=AnomalyType.PATTERN,
            severity=AnomalySeverity.HIGH,
            title="Test",
            description="Test desc",
            affected_documents=["a.pdf"],
            confidence=0.9,
            evidence={"key": "val"},
            detected_at="2026-01-01T00:00:00Z",
        )
        d = anomaly.to_dict()
        assert d["anomaly_id"] == "T-001"
        assert d["anomaly_type"] == "pattern"

    def test_result_to_dict(self):
        """Test result serialization."""
        result = AnomalyResult(
            anomalies=[],
            summary={},
            statistics={},
            recommendations=[],
            processing_time=0.5,
        )
        d = result.to_dict()
        assert "anomalies" in d
        assert d["processing_time"] == 0.5


class TestConfiguration:
    """Tests for configuration."""

    def test_default_config(self):
        """Test default config."""
        config = AnomalyConfig()
        assert config.z_score_threshold == 2.5
        assert config.cluster_min_size == 3
        assert config.enable_statistical is True

    def test_custom_config(self):
        """Test custom config."""
        config = AnomalyConfig(z_score_threshold=1.5, cluster_min_size=5)
        assert config.z_score_threshold == 1.5
        assert config.cluster_min_size == 5
