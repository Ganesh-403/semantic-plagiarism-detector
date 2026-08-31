"""Tests for src.core.document_cluster_analyzer."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.config import HIGH_SEVERITY, LOW_SEVERITY, MEDIUM_SEVERITY, SimilarityThresholds
from src.core.document_cluster_analyzer import (
    Cluster,
    ClusterAnalysisResult,
    ClusterMethod,
    ClusterRiskLevel,
    DocumentClusterAnalyzer,
    DocumentPair,
    _build_similarity_graph,
    _classify_cluster_risk,
    _compute_cluster_similarity_stats,
    _connected_components,
    cluster_by_agglomerative,
    cluster_by_threshold,
    compute_document_risk_scores,
    generate_risk_summary,
    quick_cluster,
)


# ---------------------------------------------------------------------------
# DocumentPair
# ---------------------------------------------------------------------------

class TestDocumentPair:
    def test_valid_pair(self):
        pair = DocumentPair(doc_a="a.pdf", doc_b="b.pdf", similarity=0.85)
        assert pair.doc_a == "a.pdf"
        assert pair.similarity == 0.85

    def test_canonical_key_ordering(self):
        pair = DocumentPair(doc_b="a.pdf", doc_a="z.pdf", similarity=0.5)
        key = pair.canonical_key()
        assert key == ("a.pdf", "z.pdf")

    def test_invalid_similarity_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            DocumentPair("a", "b", 1.5)
        with pytest.raises(ValueError, match="must be in"):
            DocumentPair("a", "b", -0.1)

    def test_to_dict(self):
        pair = DocumentPair("x.pdf", "y.pdf", 0.72)
        d = pair.to_dict()
        assert d["similarity"] == 0.72
        assert d["doc_a"] == "x.pdf"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestBuildSimilarityGraph:
    def test_basic_graph(self):
        names = ["a", "b", "c"]
        mat = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.2],
            [0.3, 0.2, 1.0],
        ])
        graph = _build_similarity_graph(names, mat, 0.5)
        assert ("b", 0.8) in graph["a"]
        assert len(graph["c"]) == 0

    def test_empty_graph(self):
        names = ["a", "b"]
        mat = np.array([[1.0, 0.1], [0.1, 1.0]])
        graph = _build_similarity_graph(names, mat, 0.5)
        assert len(graph["a"]) == 0
        assert len(graph["b"]) == 0


class TestConnectedComponents:
    def test_single_component(self):
        graph = {"a": [("b", 0.9)], "b": [("a", 0.9)], "c": []}
        comps = _connected_components(graph)
        assert len(comps) == 1
        assert set(comps[0]) == {"a", "b"}

    def test_multiple_components(self):
        graph = {
            "a": [("b", 0.8)],
            "b": [("a", 0.8)],
            "c": [("d", 0.9)],
            "d": [("c", 0.9)],
        }
        comps = _connected_components(graph)
        assert len(comps) == 2


class TestClusterByThreshold:
    def test_basic_clustering(self):
        names = ["a", "b", "c", "d"]
        mat = np.array([
            [1.0, 0.9, 0.2, 0.1],
            [0.9, 1.0, 0.3, 0.1],
            [0.2, 0.3, 1.0, 0.85],
            [0.1, 0.1, 0.85, 1.0],
        ])
        clusters = cluster_by_threshold(names, mat, 0.5)
        assert len(clusters) == 2
        cluster_sets = [set(c) for c in clusters]
        assert {"a", "b"} in cluster_sets
        assert {"c", "d"} in cluster_sets

    def test_no_clusters_below_threshold(self):
        names = ["a", "b"]
        mat = np.array([[1.0, 0.3], [0.3, 1.0]])
        clusters = cluster_by_threshold(names, mat, 0.5)
        assert len(clusters) == 0

    def test_single_cluster_all_connected(self):
        names = ["a", "b", "c"]
        mat = np.array([
            [1.0, 0.8, 0.7],
            [0.8, 1.0, 0.9],
            [0.7, 0.9, 1.0],
        ])
        clusters = cluster_by_threshold(names, mat, 0.5)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3


class TestClusterSimilarityStats:
    def test_basic_stats(self):
        docs = ["a", "b", "c"]
        idx = {"a": 0, "b": 1, "c": 2}
        mat = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0],
        ])
        avg, mx = _compute_cluster_similarity_stats(docs, idx, mat)
        assert abs(mx - 0.8) < 1e-6

    def test_single_document(self):
        avg, mx = _compute_cluster_similarity_stats(["a"], {"a": 0}, np.eye(1))
        assert avg == 1.0
        assert mx == 1.0


class TestClassifyClusterRisk:
    def test_critical_risk(self):
        thresholds = SimilarityThresholds()
        risk = _classify_cluster_risk(0.88, 0.95, 4, thresholds)
        assert risk == ClusterRiskLevel.CRITICAL

    def test_high_risk(self):
        thresholds = SimilarityThresholds()
        risk = _classify_cluster_risk(0.91, 0.92, 2, thresholds)
        assert risk == ClusterRiskLevel.HIGH

    def test_elevated_risk(self):
        thresholds = SimilarityThresholds()
        risk = _classify_cluster_risk(0.65, 0.68, 3, thresholds)
        assert risk == ClusterRiskLevel.ELEVATED

    def test_low_risk(self):
        thresholds = SimilarityThresholds()
        risk = _classify_cluster_risk(0.3, 0.4, 2, thresholds)
        assert risk == ClusterRiskLevel.LOW


# ---------------------------------------------------------------------------
# DocumentClusterAnalyzer
# ---------------------------------------------------------------------------

class TestDocumentClusterAnalyzer:
    def _make_matrix(self, n, high_pairs):
        mat = np.full((n, n), 0.3)
        np.fill_diagonal(mat, 1.0)
        for i, j in high_pairs:
            mat[i, j] = 0.92
            mat[j, i] = 0.92
        return mat

    def test_analyze_basic(self):
        names = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
        mat = self._make_matrix(4, [(0, 1), (2, 3)])
        analyzer = DocumentClusterAnalyzer(threshold=0.59)
        result = analyzer.analyze(names, mat)
        assert result.total_documents == 4
        assert result.total_clusters == 2

    def test_analyze_empty(self):
        analyzer = DocumentClusterAnalyzer()
        result = analyzer.analyze([], np.empty((0, 0)))
        assert result.total_clusters == 0

    def test_analyze_single_doc(self):
        analyzer = DocumentClusterAnalyzer()
        result = analyzer.analyze(["a.pdf"], np.eye(1))
        assert result.total_documents == 1

    def test_mismatched_shape_raises(self):
        analyzer = DocumentClusterAnalyzer()
        with pytest.raises(ValueError, match="does not match"):
            analyzer.analyze(["a", "b"], np.eye(3))

    def test_result_serialization(self):
        names = ["x.pdf", "y.pdf", "z.pdf"]
        mat = self._make_matrix(3, [(0, 1)])
        result = DocumentClusterAnalyzer().analyze(names, mat)
        d = result.to_dict()
        assert "clusters" in d
        assert "total_documents" in d
        assert isinstance(d["clusters"], list)

    def test_agglomerative_method(self):
        names = ["a", "b", "c", "d"]
        mat = self._make_matrix(4, [(0, 1), (2, 3)])
        result = DocumentClusterAnalyzer().analyze(
            names, mat, method=ClusterMethod.AGGLOMERATIVE
        )
        assert result.method == ClusterMethod.AGGLOMERATIVE

    def test_recommendations_generated(self):
        names = ["a", "b", "c"]
        mat = self._make_matrix(3, [(0, 1), (1, 2)])
        result = DocumentClusterAnalyzer(threshold=0.59).analyze(names, mat)
        assert len(result.recommendations) > 0

    def test_high_risk_cluster_detected(self):
        names = ["a", "b", "c", "d", "e"]
        mat = self._make_matrix(5, [(0, 1), (1, 2), (2, 0), (0, 3)])
        result = DocumentClusterAnalyzer(threshold=0.5).analyze(names, mat)
        high_risk = result.high_risk_clusters
        assert len(high_risk) > 0


# ---------------------------------------------------------------------------
# Quick cluster
# ---------------------------------------------------------------------------

class TestQuickCluster:
    def test_basic_quick_cluster(self):
        names = ["a", "b", "c"]
        mat = np.array([
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ])
        result = quick_cluster(names, mat, 0.5)
        assert len(result) == 1
        assert result[0]["size"] == 2

    def test_empty_quick_cluster(self):
        result = quick_cluster([], np.empty((0, 0)), 0.5)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Risk scores
# ---------------------------------------------------------------------------

class TestComputeDocumentRiskScores:
    def test_basic_risk_scores(self):
        names = ["a", "b", "c"]
        mat = np.array([
            [1.0, 0.95, 0.2],
            [0.95, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ])
        scores = compute_document_risk_scores(names, mat)
        assert scores["a"]["max_similarity"] == 0.95
        assert scores["a"]["most_similar_document"] == "b"
        assert scores["c"]["max_similarity"] == 0.3

    def test_empty_risk_scores(self):
        scores = compute_document_risk_scores([], np.empty((0, 0)))
        assert len(scores) == 0


class TestGenerateRiskSummary:
    def test_empty_summary(self):
        summary = generate_risk_summary({})
        assert summary["total_documents"] == 0

    def test_basic_summary(self):
        risk_scores = {
            "a.pdf": {"risk_score": 0.95, "severity": "High"},
            "b.pdf": {"risk_score": 0.3, "severity": "Low"},
        }
        summary = generate_risk_summary(risk_scores)
        assert summary["total_documents"] == 2
        assert summary["max_risk_score"] == 0.95
        assert "a.pdf" in summary["critical_documents"]
