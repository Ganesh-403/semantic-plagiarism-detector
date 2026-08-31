"""tests/api/test_document_versions.py

Unit tests for the Document Versioning API endpoints.
Tests cover snapshot CRUD, lineage, diffs, and analytics.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.middleware import get_expected_bearer_token
from src.db.version_repo import DocumentSnapshotRepository, init_version_repo_db

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
    db_path = str(tmp_path / "version_test.db")
    monkeypatch.setattr(
        "src.db.version_repo.DEFAULT_DB_PATH",
        __import__("pathlib").Path(db_path),
    )
    init_version_repo_db(__import__("pathlib").Path(db_path))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

HEADERS = {"Authorization": "Bearer test-token"}


# ---------------------------------------------------------------------------
# Snapshot CRUD Tests
# ---------------------------------------------------------------------------

class TestSnapshotCRUD:
    """Tests for version snapshot creation, listing, and retrieval."""

    def test_register_version(self):
        """Registering a version should succeed."""
        resp = client.post(
            "/api/v1/versions/snapshots",
            headers=HEADERS,
            params={
                "user_id": "alice",
                "assignment_id": "essay-01",
                "filename": "draft.docx",
                "content_text": "Hello world, this is a test document.",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "document_hash" in data
        assert data["version_number"] == 1
        assert data["user_id"] == "alice"

    def test_register_multiple_versions(self):
        """Multiple versions should increment version numbers."""
        for i in range(3):
            resp = client.post(
                "/api/v1/versions/snapshots",
                headers=HEADERS,
                params={
                    "user_id": "bob",
                    "assignment_id": "lab-03",
                    "filename": f"draft-v{i+1}.docx",
                    "content_text": f"Version {i+1} content with unique words {i*100}",
                },
            )
            assert resp.status_code == 201

        # Check lineage
        resp = client.get(
            "/api/v1/versions/lineage/bob/lab-03",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3

    def test_list_snapshots(self):
        """Listing snapshots should return paginated results."""
        # Register one first
        client.post(
            "/api/v1/versions/snapshots",
            headers=HEADERS,
            params={
                "user_id": "carol",
                "assignment_id": "midterm",
                "filename": "essay.docx",
                "content_text": "Some content",
            },
        )
        resp = client.get(
            "/api/v1/versions/snapshots",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["total"] >= 1

    def test_list_snapshots_with_filter(self):
        """Filtering by user should return only matching snapshots."""
        client.post(
            "/api/v1/versions/snapshots",
            headers=HEADERS,
            params={
                "user_id": "dave",
                "assignment_id": "hw-01",
                "filename": "hw.docx",
                "content_text": "Homework content",
            },
        )
        resp = client.get(
            "/api/v1/versions/snapshots?user_id=dave",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert all(item["user_id"] == "dave" for item in data["items"])

    def test_get_snapshot(self):
        """Getting a snapshot by hash should return the record."""
        create_resp = client.post(
            "/api/v1/versions/snapshots",
            headers=HEADERS,
            params={
                "user_id": "eve",
                "assignment_id": "essay-02",
                "filename": "final.docx",
                "content_text": "Final version content",
            },
        )
        doc_hash = create_resp.json()["document_hash"]

        resp = client.get(
            f"/api/v1/versions/snapshots/{doc_hash}",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["document_hash"] == doc_hash

    def test_get_snapshot_not_found(self):
        """Getting a non-existent snapshot should return 404."""
        resp = client.get(
            "/api/v1/versions/snapshots/nonexistent_hash",
            headers=HEADERS,
        )
        assert resp.status_code == 404

    def test_delete_snapshot(self):
        """Deleting a snapshot should succeed."""
        create_resp = client.post(
            "/api/v1/versions/snapshots",
            headers=HEADERS,
            params={
                "user_id": "alice",
                "assignment_id": "temp-01",
                "filename": "temp.docx",
                "content_text": "Temporary content",
            },
        )
        doc_hash = create_resp.json()["document_hash"]

        resp = client.delete(
            f"/api/v1/versions/snapshots/{doc_hash}",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_delete_snapshot_not_found(self):
        """Deleting a non-existent snapshot should return 404."""
        resp = client.delete(
            "/api/v1/versions/snapshots/nonexistent",
            headers=HEADERS,
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Lineage Tests
# ---------------------------------------------------------------------------

class TestLineage:
    """Tests for version lineage endpoints."""

    def test_get_lineage(self):
        """Getting a lineage should return all versions in order."""
        for i in range(4):
            client.post(
                "/api/v1/versions/snapshots",
                headers=HEADERS,
                params={
                    "user_id": "alice",
                    "assignment_id": "thesis",
                    "filename": f"chapter{i+1}.docx",
                    "content_text": f"Chapter {i+1} content here",
                },
            )

        resp = client.get(
            "/api/v1/versions/lineage/alice/thesis",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 4
        assert data["user_id"] == "alice"
        assert data["assignment_id"] == "thesis"

    def test_list_lineages(self):
        """Listing lineages should return results."""
        # Create some versions
        for user in ["alice", "bob"]:
            client.post(
                "/api/v1/versions/snapshots",
                headers=HEADERS,
                params={
                    "user_id": user,
                    "assignment_id": "essay",
                    "filename": "draft.docx",
                    "content_text": f"{user}'s draft content",
                },
            )

        resp = client.get(
            "/api/v1/versions/lineage",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2

    def test_delete_lineage(self):
        """Deleting a lineage should remove all versions."""
        for i in range(3):
            client.post(
                "/api/v1/versions/snapshots",
                headers=HEADERS,
                params={
                    "user_id": "carol",
                    "assignment_id": "temp-lin",
                    "filename": f"v{i}.docx",
                    "content_text": f"Version {i}",
                },
            )

        resp = client.delete(
            "/api/v1/versions/lineage/carol/temp-lin",
            headers=HEADERS,
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Diff Tests
# ---------------------------------------------------------------------------

class TestDiffs:
    """Tests for version diff endpoints."""

    def test_register_and_get_diff(self):
        """Registering a diff should succeed, and retrieval should return it."""
        # Create two versions
        v1 = client.post(
            "/api/v1/versions/snapshots",
            headers=HEADERS,
            params={
                "user_id": "alice",
                "assignment_id": "diff-test",
                "filename": "v1.docx",
                "content_text": "First version content",
            },
        ).json()

        v2 = client.post(
            "/api/v1/versions/snapshots",
            headers=HEADERS,
            params={
                "user_id": "alice",
                "assignment_id": "diff-test",
                "filename": "v2.docx",
                "content_text": "Second version with changes",
            },
        ).json()

        # Register diff
        resp = client.post(
            "/api/v1/versions/diffs",
            headers=HEADERS,
            params={
                "parent_hash": v1["document_hash"],
                "child_hash": v2["document_hash"],
                "similarity": 0.72,
                "added_words": 150,
                "removed_words": 30,
                "changed_words": 45,
                "jaccard_index": 0.65,
            },
        )
        assert resp.status_code == 201

        # Get diff
        resp = client.get(
            f"/api/v1/versions/diffs/{v1['document_hash']}/{v2['document_hash']}",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["similarity"] == 0.72
        assert data["added_words"] == 150

    def test_get_diffs_for_version(self):
        """Getting all diffs for a version should return related diffs."""
        resp = client.get(
            "/api/v1/versions/diffs/some_hash/all",
            headers=HEADERS,
        )
        assert resp.status_code == 200

    def test_get_diff_not_found(self):
        """Getting a non-existent diff should return 404."""
        resp = client.get(
            "/api/v1/versions/diffs/aaa/bbb",
            headers=HEADERS,
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Analytics Tests
# ---------------------------------------------------------------------------

class TestAnalytics:
    """Tests for analytics endpoints."""

    def test_analytics_summary(self):
        """Summary should return valid statistics."""
        resp = client.get(
            "/api/v1/versions/analytics/summary",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_versions" in data
        assert "avg_similarity" in data
        assert "unique_users" in data

    def test_similarity_trend(self):
        """Trend endpoint should return trend data."""
        resp = client.get(
            "/api/v1/versions/analytics/trend/alice/essay",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "trend" in data

    def test_most_revised(self):
        """Most revised endpoint should return documents."""
        resp = client.get(
            "/api/v1/versions/analytics/most-revised?limit=5",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data

    def test_highest_drift(self):
        """Highest drift endpoint should return documents."""
        resp = client.get(
            "/api/v1/versions/analytics/highest-drift?limit=5",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data


# ---------------------------------------------------------------------------
# Auth Tests
# ---------------------------------------------------------------------------

class TestAuth:
    """Tests for authentication requirements."""

    def test_no_auth_header(self):
        """Requests without auth header should be rejected."""
        resp = client.get("/api/v1/versions/snapshots")
        assert resp.status_code in (401, 403)

    def test_invalid_auth_token(self):
        """Requests with wrong token should be rejected."""
        resp = client.get(
            "/api/v1/versions/snapshots",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code in (401, 403)

    def test_register_no_auth(self):
        """POST without auth should be rejected."""
        resp = client.post(
            "/api/v1/versions/snapshots",
            params={"user_id": "x", "assignment_id": "y", "content_text": "z"},
        )
        assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# DB Repository Unit Tests
# ---------------------------------------------------------------------------

class TestVersionRepoDB:
    """Unit tests for DocumentSnapshotRepository."""

    def test_register_and_get(self):
        """Should register and retrieve a snapshot."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_version_repo_db(db)
            repo = DocumentSnapshotRepository(db)

            result = repo.register_version(
                user_id="test_user",
                assignment_id="test_assignment",
                filename="test.docx",
                content_text="Test content here",
            )
            assert result["version_number"] == 1

            snap = repo.get_snapshot(result["document_hash"])
            assert snap is not None
            assert snap["word_count"] == 4

    def test_version_number_increments(self):
        """Version numbers should increment for the same user + assignment."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_version_repo_db(db)
            repo = DocumentSnapshotRepository(db)

            v1 = repo.register_version("u1", "a1", "f1", "First draft")
            v2 = repo.register_version("u1", "a1", "f2", "Second draft with changes")
            v3 = repo.register_version("u1", "a1", "f3", "Third draft final")

            assert v1["version_number"] == 1
            assert v2["version_number"] == 2
            assert v3["version_number"] == 3

    def test_list_versions_pagination(self):
        """Should paginate versions correctly."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_version_repo_db(db)
            repo = DocumentSnapshotRepository(db)

            for i in range(25):
                repo.register_version("u1", "a1", f"f{i}", f"Content {i}")

            page1 = repo.list_versions(page=1, per_page=10)
            assert len(page1["items"]) == 10
            assert page1["total"] == 25
            assert page1["has_next"] is True

            page3 = repo.list_versions(page=3, per_page=10)
            assert len(page3["items"]) == 5
            assert page3["has_next"] is False

    def test_lineage(self):
        """Should track and retrieve full lineage."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_version_repo_db(db)
            repo = DocumentSnapshotRepository(db)

            repo.register_version("u1", "a1", "v1", "Draft 1")
            repo.register_version("u1", "a1", "v2", "Draft 2")
            repo.register_version("u1", "a1", "v3", "Draft 3")

            lineage = repo.get_lineage("u1", "a1")
            assert len(lineage) == 3
            assert lineage[0]["version_number"] == 1
            assert lineage[2]["version_number"] == 3

    def test_register_diff(self):
        """Should store and retrieve diffs."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_version_repo_db(db)
            repo = DocumentSnapshotRepository(db)

            diff_id = repo.register_diff(
                parent_hash="hash1",
                child_hash="hash2",
                similarity=0.85,
                added_words=100,
                removed_words=20,
                changed_words=30,
                jaccard_index=0.6,
            )
            assert diff_id is not None

            diff = repo.get_diff("hash1", "hash2")
            assert diff is not None
            assert diff["similarity"] == 0.85

    def test_analytics_summary(self):
        """Should return correct analytics."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_version_repo_db(db)
            repo = DocumentSnapshotRepository(db)

            repo.register_version("u1", "a1", "v1", "Content one")
            repo.register_version("u1", "a1", "v2", "Content two modified")
            repo.register_version("u2", "a2", "v1", "Another document")

            repo.register_diff("hash_a", "hash_b", 0.75, 50, 10, 20, 0.5)

            summary = repo.analytics_summary()
            assert summary["total_versions"] == 3
            assert summary["unique_users"] == 2
            assert summary["total_diffs"] == 1

    def test_most_revised_documents(self):
        """Should return documents sorted by version count."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_version_repo_db(db)
            repo = DocumentSnapshotRepository(db)

            for i in range(5):
                repo.register_version("u1", "many", f"v{i}", f"Version {i}")
            for i in range(2):
                repo.register_version("u2", "few", f"v{i}", f"Version {i}")

            most = repo.most_revised_documents(limit=5)
            assert most[0]["total_versions"] == 5

    def test_delete_lineage(self):
        """Should delete all versions in a lineage."""
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            init_version_repo_db(db)
            repo = DocumentSnapshotRepository(db)

            for i in range(3):
                repo.register_version("u1", "del", f"v{i}", f"Content {i}")

            deleted = repo.delete_lineage("u1", "del")
            assert deleted is True

            remaining = repo.get_lineage("u1", "del")
            assert len(remaining) == 0
