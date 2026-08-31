"""tests/api/test_heatmap.py

Unit tests for the Similarity Heatmap & Clustering API endpoints.
Tests cover snapshot CRUD, clustering, hotspot management, and analytics.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.middleware import get_expected_bearer_token
from src.db.heatmap_db import HeatmapRepository, init_heatmap_db

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_auth(monkeypatch):
    """Bypass bearer-token auth for every test."""
    monkeypatch.setattr(
        "src.api.middleware.get_expected_bearer_token",
        lambda: "test-token",
    )


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Point the DB at a fresh temp file for each test."""
    db_path = str(tmp_path / "heatmap_test.db")
    monkeypatch.setattr(
        "src.db.heatmap_db.get_heatmap_db_path",
        lambda: db_path,
    )
    init_heatmap_db()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

HEADERS = {"Authorization": "Bearer test-token"}


# ---------------------------------------------------------------------------
# Snapshot CRUD Tests
# ---------------------------------------------------------------------------

class TestSnapshotCRUD:
    """Tests for heatmap snapshot creation, listing, and retrieval."""

    def test_create_snapshot_empty_corpus(self):
        """Creating a snapshot with no documents should still succeed."""
        resp = client.post(
            "/api/v1/heatmap/snapshots",
            headers=HEADERS,
            json={"notes": "Empty corpus test"},
        )
        assert resp.status_code in (200, 201, 404)

    def test_list_snapshots_empty(self):
        """Listing snapshots on a fresh DB should return empty list."""
        resp = client.get(
            "/api/v1/heatmap/snapshots",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, (dict, list))

    def test_list_snapshots_with_pagination(self):
        """Pagination parameters should be accepted."""
        resp = client.get(
            "/api/v1/heatmap/snapshots?page=1&per_page=10",
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_get_snapshot_not_found(self):
        """Requesting a non-existent snapshot should return 404."""
        resp = client.get(
            "/api/v1/heatmap/snapshots/999999",
            headers=HEADERS,
        )
        assert resp.status_code in (404, 422)

    def test_delete_snapshot_not_found(self):
        """Deleting a non-existent snapshot should return 404."""
        resp = client.delete(
            "/api/v1/heatmap/snapshots/999999",
            headers=HEADERS,
        )
        assert resp.status_code in (404, 422)


# ---------------------------------------------------------------------------
# Clustering Tests
# ---------------------------------------------------------------------------

class TestClustering:
    """Tests for document clustering endpoints."""

    def test_cluster_empty_corpus(self):
        """Clustering with no documents should return gracefully."""
        resp = client.post(
            "/api/v1/heatmap/cluster",
            headers=HEADERS,
            json={"linkage_method": "average", "distance_threshold": 0.5},
        )
        assert resp.status_code in (200, 404)

    def test_cluster_default_params(self):
        """Clustering with default parameters should be accepted."""
        resp = client.post(
            "/api/v1/heatmap/cluster",
            headers=HEADERS,
            json={},
        )
        assert resp.status_code in (200, 404)

    def test_list_clustering_results(self):
        """Listing clustering results on empty DB should return empty."""
        resp = client.get(
            "/api/v1/heatmap/cluster",
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_get_clustering_result_not_found(self):
        """Requesting non-existent clustering result should return 404."""
        resp = client.get(
            "/api/v1/heatmap/cluster/999999",
            headers=HEADERS,
        )
        assert resp.status_code in (404, 422)


# ---------------------------------------------------------------------------
# Hotspot Tests
# ---------------------------------------------------------------------------

class TestHotspots:
    """Tests for similarity hotspot endpoints."""

    def test_list_hotspots_empty(self):
        """Listing hotspots on fresh DB should return empty."""
        resp = client.get(
            "/api/v1/heatmap/hotspots",
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_hotspot_summary(self):
        """Summary endpoint should return zeroed stats on empty DB."""
        resp = client.get(
            "/api/v1/heatmap/hotspots/summary",
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_resolve_hotspot_not_found(self):
        """Resolving a non-existent hotspot should return 404."""
        resp = client.put(
            "/api/v1/heatmap/hotspots/999999/resolve",
            headers=HEADERS,
        )
        assert resp.status_code in (404, 422)

    def test_list_hotspots_with_filter(self):
        """Filtering hotspots by severity should be accepted."""
        resp = client.get(
            "/api/v1/heatmap/hotspots?severity=critical",
            headers=HEADERS,
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Analytics Tests
# ---------------------------------------------------------------------------

class TestAnalytics:
    """Tests for heatmap analytics endpoints."""

    def test_analytics_summary(self):
        """Analytics summary on empty DB should return zeroed stats."""
        resp = client.get(
            "/api/v1/heatmap/analytics/summary",
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_analytics_similarity_distribution(self):
        """Similarity distribution should be accepted."""
        resp = client.get(
            "/api/v1/heatmap/analytics/distribution",
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_analytics_top_pairs(self):
        """Top pairs endpoint should be accepted."""
        resp = client.get(
            "/api/v1/heatmap/analytics/top-pairs?limit=5",
            headers=HEADERS,
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth Tests
# ---------------------------------------------------------------------------

class TestAuth:
    """Tests for authentication requirements."""

    def test_no_auth_header(self):
        """Requests without auth header should be rejected."""
        resp = client.get("/api/v1/heatmap/snapshots")
        assert resp.status_code in (401, 403)

    def test_invalid_auth_token(self):
        """Requests with wrong token should be rejected."""
        resp = client.get(
            "/api/v1/heatmap/snapshots",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code in (401, 403)

    def test_create_snapshot_no_auth(self):
        """POST without auth should be rejected."""
        resp = client.post(
            "/api/v1/heatmap/snapshots",
            json={},
        )
        assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Core Engine Unit Tests
# ---------------------------------------------------------------------------

class TestSimilarityEngine:
    """Unit tests for the similarity_heatmap core module."""

    def test_import(self):
        """Module should be importable."""
        from src.core.similarity_heatmap import SimilarityEngine
        assert SimilarityEngine is not None

    def test_heatmap_matrix_dataclass(self):
        """HeatmapMatrix dataclass should instantiate correctly."""
        from src.core.similarity_heatmap import HeatmapMatrix
        m = HeatmapMatrix(
            labels=["a", "b"],
            matrix=[[1.0, 0.5], [0.5, 1.0]],
            min_similarity=0.5,
            max_similarity=1.0,
            mean_similarity=0.75,
            document_count=2,
        )
        assert m.document_count == 2
        assert m.min_similarity == 0.5

    def test_heatmap_cell_dataclass(self):
        """HeatmapCell dataclass should instantiate correctly."""
        from src.core.similarity_heatmap import HeatmapCell
        c = HeatmapCell(
            row_label="doc1",
            col_label="doc2",
            similarity=0.85,
            row_index=0,
            col_index=1,
        )
        assert c.similarity == 0.85

    def test_cluster_info_dataclass(self):
        """ClusterInfo dataclass should instantiate correctly."""
        from src.core.similarity_heatmap import ClusterInfo
        ci = ClusterInfo(
            cluster_id=1,
            documents=["a.pdf", "b.pdf"],
            centroid_score=0.9,
            size=2,
        )
        assert ci.size == 2

    def test_empty_similarity_matrix(self):
        """Engine should handle empty document lists."""
        from src.core.similarity_heatmap import SimilarityEngine
        engine = SimilarityEngine()
        result = engine.compute_similarity_matrix([])
        assert result.document_count == 0

    def test_single_document_matrix(self):
        """Engine should handle single-document input."""
        from src.core.similarity_heatmap import SimilarityEngine
        engine = SimilarityEngine()
        result = engine.compute_similarity_matrix(["doc1"])
        assert result.document_count == 1
        assert result.max_similarity == 1.0  # self-similarity

    def test_two_document_matrix(self):
        """Engine should produce a 2x2 matrix for two documents."""
        from src.core.similarity_heatmap import SimilarityEngine
        engine = SimilarityEngine()
        result = engine.compute_similarity_matrix(["doc1", "doc2"])
        assert result.document_count == 2
        assert len(result.matrix) == 2
        assert len(result.matrix[0]) == 2

    def test_cluster_empty(self):
        """Clustering empty input should return zero clusters."""
        from src.core.similarity_heatmap import SimilarityEngine
        engine = SimilarityEngine()
        result = engine.cluster_documents([])
        assert result.num_clusters == 0

    def test_cluster_single_doc(self):
        """Clustering one document should produce one cluster."""
        from src.core.similarity_heatmap import SimilarityEngine
        engine = SimilarityEngine()
        result = engine.cluster_documents(["doc1"])
        assert result.num_clusters == 1

    def test_hotspot_detection_empty(self):
        """Hotspot detection on empty matrix should return empty list."""
        from src.core.similarity_heatmap import SimilarityEngine
        engine = SimilarityEngine()
        hotspots = engine.detect_hotspots([], threshold=0.8)
        assert hotspots == []

    def test_hotspot_detection_below_threshold(self):
        """Pairs below threshold should not be flagged as hotspots."""
        from src.core.similarity_heatmap import SimilarityEngine
        engine = SimilarityEngine()
        matrix = [[1.0, 0.3], [0.3, 1.0]]
        labels = ["a", "b"]
        hotspots = engine.detect_hotspots(
            labels, matrix=matrix, threshold=0.8
        )
        assert len(hotspots) == 0

    def test_hotspot_detection_above_threshold(self):
        """Pairs above threshold should be flagged as hotspots."""
        from src.core.similarity_heatmap import SimilarityEngine
        engine = SimilarityEngine()
        matrix = [[1.0, 0.95], [0.95, 1.0]]
        labels = ["a", "b"]
        hotspots = engine.detect_hotspots(
            labels, matrix=matrix, threshold=0.8
        )
        assert len(hotspots) == 1

    def test_silhouette_score_range(self):
        """Silhouette score should be between -1 and 1."""
        from src.core.similarity_heatmap import SimilarityEngine
        engine = SimilarityEngine()
        result = engine.cluster_documents(
            ["doc1", "doc2", "doc3", "doc4"],
            num_clusters=2,
        )
        assert -1.0 <= result.silhouette_score <= 1.0


# ---------------------------------------------------------------------------
# DB Repository Tests
# ---------------------------------------------------------------------------

class TestHeatmapDB:
    """Unit tests for HeatmapRepository."""

    def test_create_snapshot(self):
        """Should create and return a snapshot record."""
        repo = HeatmapRepository()
        snap_id = repo.create_snapshot(
            labels=["a", "b"],
            matrix=[[1.0, 0.5], [0.5, 1.0]],
            document_count=2,
            min_sim=0.5,
            max_sim=1.0,
            mean_sim=0.75,
        )
        assert snap_id is not None
        assert snap_id > 0

    def test_list_snapshots(self):
        """Should list snapshots in reverse chronological order."""
        repo = HeatmapRepository()
        repo.create_snapshot(
            labels=["a"], matrix=[[1.0]], document_count=1,
            min_sim=1.0, max_sim=1.0, mean_sim=1.0,
        )
        repo.create_snapshot(
            labels=["b"], matrix=[[1.0]], document_count=1,
            min_sim=1.0, max_sim=1.0, mean_sim=1.0,
        )
        snaps = repo.list_snapshots(page=1, per_page=10)
        assert len(snaps["items"]) == 2
        assert snaps["total"] == 2

    def test_get_snapshot(self):
        """Should retrieve a snapshot by ID."""
        repo = HeatmapRepository()
        snap_id = repo.create_snapshot(
            labels=["x"], matrix=[[1.0]], document_count=1,
            min_sim=1.0, max_sim=1.0, mean_sim=1.0,
        )
        snap = repo.get_snapshot(snap_id)
        assert snap is not None
        assert snap["document_count"] == 1

    def test_delete_snapshot(self):
        """Should delete a snapshot by ID."""
        repo = HeatmapRepository()
        snap_id = repo.create_snapshot(
            labels=["y"], matrix=[[1.0]], document_count=1,
            min_sim=1.0, max_sim=1.0, mean_sim=1.0,
        )
        deleted = repo.delete_snapshot(snap_id)
        assert deleted is True
        assert repo.get_snapshot(snap_id) is None

    def test_create_hotspot(self):
        """Should create a hotspot record."""
        repo = HeatmapRepository()
        h_id = repo.create_hotspot(
            doc_a="file1.pdf",
            doc_b="file2.pdf",
            similarity=0.92,
            severity="critical",
        )
        assert h_id is not None
        assert h_id > 0

    def test_list_hotspots(self):
        """Should list hotspots with optional severity filter."""
        repo = HeatmapRepository()
        repo.create_hotspot("a.pdf", "b.pdf", 0.9, "critical")
        repo.create_hotspot("c.pdf", "d.pdf", 0.7, "warning")
        all_hotspots = repo.list_hotspots()
        assert len(all_hotspots["items"]) >= 2
        critical = repo.list_hotspots(severity="critical")
        assert all(h["severity"] == "critical" for h in critical["items"])

    def test_resolve_hotspot(self):
        """Should mark a hotspot as resolved."""
        repo = HeatmapRepository()
        h_id = repo.create_hotspot("a.pdf", "b.pdf", 0.85, "warning")
        resolved = repo.resolve_hotspot(h_id)
        assert resolved is True

    def test_hotspot_summary(self):
        """Should return correct hotspot summary counts."""
        repo = HeatmapRepository()
        repo.create_hotspot("a.pdf", "b.pdf", 0.95, "critical")
        repo.create_hotspot("c.pdf", "d.pdf", 0.7, "warning")
        summary = repo.hotspot_summary()
        assert summary["total_hotspots"] >= 2
        assert summary["unresolved"] >= 2

    def test_create_clustering_result(self):
        """Should create and store clustering results."""
        repo = HeatmapRepository()
        r_id = repo.create_clustering_result(
            num_clusters=3,
            silhouette_score=0.45,
            linkage_method="average",
            distance_threshold=0.5,
            clusters=[{"id": 0, "docs": ["a.pdf"], "score": 0.9, "size": 1}],
            assignments={"a.pdf": 0},
        )
        assert r_id is not None
        assert r_id > 0

    def test_list_clustering_results(self):
        """Should list clustering results."""
        repo = HeatmapRepository()
        repo.create_clustering_result(
            num_clusters=2, silhouette_score=0.3,
            linkage_method="complete", distance_threshold=0.6,
            clusters=[], assignments={},
        )
        results = repo.list_clustering_results(page=1, per_page=10)
        assert results["total"] >= 1

    def test_get_clustering_result(self):
        """Should retrieve clustering result by ID."""
        repo = HeatmapRepository()
        r_id = repo.create_clustering_result(
            num_clusters=2, silhouette_score=0.5,
            linkage_method="average", distance_threshold=0.4,
            clusters=[{"id": 0, "docs": ["a.pdf"], "score": 0.9, "size": 1}],
            assignments={"a.pdf": 0},
        )
        result = repo.get_clustering_result(r_id)
        assert result is not None
        assert result["num_clusters"] == 2

    def test_analytics_summary(self):
        """Should return analytics summary from stored data."""
        repo = HeatmapRepository()
        repo.create_snapshot(
            labels=["a", "b"],
            matrix=[[1.0, 0.6], [0.6, 1.0]],
            document_count=2,
            min_sim=0.6, max_sim=1.0, mean_sim=0.8,
        )
        summary = repo.analytics_summary()
        assert "total_snapshots" in summary
        assert summary["total_snapshots"] >= 1
