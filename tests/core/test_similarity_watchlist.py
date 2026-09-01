"""Tests for src.core.similarity_watchlist."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.core.similarity_watchlist import (
    AlertTrigger,
    SimilarityWatchlistChecker,
    SimilarityWatchlistRepository,
    WatchlistAlert,
    WatchlistCheckResult,
    WatchlistEntry,
    WatchlistStatus,
    WatchlistSummary,
    WatchlistType,
    init_watchlist_db,
    quick_watch_document,
    quick_watch_pair,
)


@pytest.fixture
def tmp_db():
    """Provide a temporary database path and clean up after."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "watchlist.db")
        yield db_path


@pytest.fixture
def repo(tmp_db):
    return SimilarityWatchlistRepository(tmp_db)


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

class TestInitWatchlistDb:
    def test_creates_tables(self, tmp_db):
        init_watchlist_db(tmp_db)
        repo = SimilarityWatchlistRepository(tmp_db)
        entries = repo.get_all_entries()
        assert entries == []


# ---------------------------------------------------------------------------
# WatchlistEntry
# ---------------------------------------------------------------------------

class TestWatchlistEntry:
    def test_to_dict(self):
        entry = WatchlistEntry(
            entry_id=1,
            watchlist_type=WatchlistType.DOCUMENT,
            target="essay.pdf",
            label="Watch essay",
            status=WatchlistStatus.ACTIVE,
        )
        d = entry.to_dict()
        assert d["entry_id"] == 1
        assert d["watchlist_type"] == "document"
        assert d["status"] == "active"


# ---------------------------------------------------------------------------
# SimilarityWatchlistRepository
# ---------------------------------------------------------------------------

class TestRepository:
    def test_add_and_get_entry(self, repo):
        entry = WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="essay.pdf",
            label="Test entry",
            created_by="admin",
        )
        entry_id = repo.add_entry(entry)
        assert entry_id is not None

        fetched = repo.get_entry(entry_id)
        assert fetched is not None
        assert fetched.target == "essay.pdf"
        assert fetched.label == "Test entry"

    def test_get_nonexistent_entry(self, repo):
        assert repo.get_entry(99999) is None

    def test_get_all_entries(self, repo):
        for i in range(3):
            repo.add_entry(WatchlistEntry(
                watchlist_type=WatchlistType.DOCUMENT,
                target=f"doc{i}.pdf",
                created_by="admin",
            ))
        entries = repo.get_all_entries()
        assert len(entries) == 3

    def test_get_all_entries_filtered_by_status(self, repo):
        repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="a.pdf",
            status=WatchlistStatus.ACTIVE,
            created_by="admin",
        ))
        repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="b.pdf",
            status=WatchlistStatus.PAUSED,
            created_by="admin",
        ))
        active = repo.get_all_entries(status=WatchlistStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].target == "a.pdf"

    def test_update_entry_status(self, repo):
        entry_id = repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="a.pdf",
            created_by="admin",
        ))
        assert repo.update_entry_status(entry_id, WatchlistStatus.PAUSED)
        entry = repo.get_entry(entry_id)
        assert entry.status == WatchlistStatus.PAUSED

    def test_delete_entry(self, repo):
        entry_id = repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="a.pdf",
            created_by="admin",
        ))
        assert repo.delete_entry(entry_id)
        assert repo.get_entry(entry_id) is None

    def test_add_and_get_alert(self, repo):
        entry_id = repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="a.pdf",
            created_by="admin",
        ))
        alert = WatchlistAlert(
            entry_id=entry_id,
            triggered_by=AlertTrigger.NEW_SCAN,
            matched_document="b.pdf",
            similarity_score=0.92,
            severity="High",
            scan_timestamp="2025-01-01T00:00:00",
            created_at="2025-01-01T00:00:00",
        )
        alert_id = repo.add_alert(alert)
        assert alert_id is not None

        alerts = repo.get_alerts(entry_id=entry_id)
        assert len(alerts) == 1
        assert alerts[0].similarity_score == 0.92

    def test_acknowledge_alert(self, repo):
        entry_id = repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="a.pdf",
            created_by="admin",
        ))
        alert = WatchlistAlert(
            entry_id=entry_id,
            triggered_by=AlertTrigger.NEW_SCAN,
            matched_document="b.pdf",
            similarity_score=0.85,
            severity="Medium",
            scan_timestamp="2025-01-01T00:00:00",
            created_at="2025-01-01T00:00:00",
        )
        alert_id = repo.add_alert(alert)
        assert repo.acknowledge_alert(alert_id)
        alerts = repo.get_alerts(entry_id=entry_id)
        assert alerts[0].acknowledged is True

    def test_acknowledge_all_alerts(self, repo):
        entry_id = repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="a.pdf",
            created_by="admin",
        ))
        for _ in range(3):
            repo.add_alert(WatchlistAlert(
                entry_id=entry_id,
                triggered_by=AlertTrigger.NEW_SCAN,
                matched_document="b.pdf",
                similarity_score=0.8,
                severity="Medium",
                scan_timestamp="2025-01-01T00:00:00",
                created_at="2025-01-01T00:00:00",
            ))
        count = repo.acknowledge_all_alerts(entry_id=entry_id)
        assert count == 3

    def test_get_alert_count(self, repo):
        entry_id = repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="a.pdf",
            created_by="admin",
        ))
        repo.add_alert(WatchlistAlert(
            entry_id=entry_id,
            triggered_by=AlertTrigger.NEW_SCAN,
            matched_document="b.pdf",
            similarity_score=0.8,
            severity="Medium",
            scan_timestamp="2025-01-01T00:00:00",
            created_at="2025-01-01T00:00:00",
        ))
        count = repo.get_alert_count(entry_id=entry_id)
        assert count == 1

    def test_get_summary(self, repo):
        repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="a.pdf",
            status=WatchlistStatus.ACTIVE,
            created_by="admin",
        ))
        summary = repo.get_summary()
        assert summary.total_entries == 1
        assert summary.active_entries == 1

    def test_find_entries_for_document(self, repo):
        entry_id = repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="target.pdf",
            status=WatchlistStatus.ACTIVE,
            created_by="admin",
        ))
        matches = repo.find_entries_for_document("target.pdf")
        assert len(matches) == 1
        assert matches[0].entry_id == entry_id

    def test_find_entries_for_document_no_match(self, repo):
        repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="other.pdf",
            status=WatchlistStatus.ACTIVE,
            created_by="admin",
        ))
        matches = repo.find_entries_for_document("target.pdf")
        assert len(matches) == 0

    def test_metadata_roundtrip(self, repo):
        entry = WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="a.pdf",
            metadata={"course": "CS101", "semester": "Fall 2025"},
            created_by="admin",
        )
        entry_id = repo.add_entry(entry)
        fetched = repo.get_entry(entry_id)
        assert fetched.metadata["course"] == "CS101"


# ---------------------------------------------------------------------------
# SimilarityWatchlistChecker
# ---------------------------------------------------------------------------

class TestSimilarityWatchlistChecker:
    def test_check_document_match(self, tmp_db):
        repo = SimilarityWatchlistRepository(tmp_db)
        repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="source.pdf",
            status=WatchlistStatus.ACTIVE,
            created_by="admin",
        ))

        doc_names = ["source.pdf", "copy.pdf", "unrelated.pdf"]
        sim_matrix = np.array([
            [1.0, 0.95, 0.1],
            [0.95, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ])

        checker = SimilarityWatchlistChecker(repo)
        results = checker.check_scan_results(doc_names, sim_matrix)

        assert len(results) == 1
        assert len(results[0].matches) == 1
        assert results[0].matches[0].similarity_score == 0.95

    def test_check_no_match_below_threshold(self, tmp_db):
        repo = SimilarityWatchlistRepository(tmp_db)
        repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="source.pdf",
            status=WatchlistStatus.ACTIVE,
            created_by="admin",
        ))

        doc_names = ["source.pdf", "unrelated.pdf"]
        sim_matrix = np.array([[1.0, 0.2], [0.2, 1.0]])

        checker = SimilarityWatchlistChecker(repo)
        results = checker.check_scan_results(doc_names, sim_matrix)

        assert len(results) == 1
        assert len(results[0].matches) == 0

    def test_check_pair_match(self, tmp_db):
        repo = SimilarityWatchlistRepository(tmp_db)
        repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.PAIR,
            target="a.pdf|b.pdf",
            status=WatchlistStatus.ACTIVE,
            created_by="admin",
        ))

        doc_names = ["a.pdf", "b.pdf", "c.pdf"]
        sim_matrix = np.array([
            [1.0, 0.88, 0.3],
            [0.88, 1.0, 0.2],
            [0.3, 0.2, 1.0],
        ])

        checker = SimilarityWatchlistChecker(repo)
        results = checker.check_scan_results(doc_names, sim_matrix)

        assert len(results) == 1
        assert len(results[0].matches) == 1

    def test_custom_threshold(self, tmp_db):
        repo = SimilarityWatchlistRepository(tmp_db)
        repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="source.pdf",
            similarity_threshold=0.99,
            status=WatchlistStatus.ACTIVE,
            created_by="admin",
        ))

        doc_names = ["source.pdf", "copy.pdf"]
        sim_matrix = np.array([[1.0, 0.90], [0.90, 1.0]])

        checker = SimilarityWatchlistChecker(repo)
        results = checker.check_scan_results(doc_names, sim_matrix)

        assert len(results[0].matches) == 0  # 0.90 < 0.99

    def test_empty_scan(self, tmp_db):
        repo = SimilarityWatchlistRepository(tmp_db)
        repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="a.pdf",
            created_by="admin",
        ))
        checker = SimilarityWatchlistChecker(repo)
        results = checker.check_scan_results([], np.empty((0, 0)))
        assert len(results) == 0

    def test_no_active_entries(self, tmp_db):
        repo = SimilarityWatchlistRepository(tmp_db)
        repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="a.pdf",
            status=WatchlistStatus.PAUSED,
            created_by="admin",
        ))
        checker = SimilarityWatchlistChecker(repo)
        results = checker.check_scan_results(
            ["a.pdf", "b.pdf"],
            np.array([[1.0, 0.9], [0.9, 1.0]]),
        )
        assert len(results) == 0

    def test_document_not_in_scan(self, tmp_db):
        repo = SimilarityWatchlistRepository(tmp_db)
        repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="missing.pdf",
            status=WatchlistStatus.ACTIVE,
            created_by="admin",
        ))
        checker = SimilarityWatchlistChecker(repo)
        results = checker.check_scan_results(
            ["other.pdf"],
            np.eye(1),
        )
        assert len(results) == 1
        assert len(results[0].matches) == 0

    def test_alert_saved_to_db(self, tmp_db):
        repo = SimilarityWatchlistRepository(tmp_db)
        entry_id = repo.add_entry(WatchlistEntry(
            watchlist_type=WatchlistType.DOCUMENT,
            target="source.pdf",
            status=WatchlistStatus.ACTIVE,
            created_by="admin",
        ))
        doc_names = ["source.pdf", "copy.pdf"]
        sim_matrix = np.array([[1.0, 0.92], [0.92, 1.0]])

        checker = SimilarityWatchlistChecker(repo)
        checker.check_scan_results(doc_names, sim_matrix)

        alerts = repo.get_alerts(entry_id=entry_id)
        assert len(alerts) == 1
        assert alerts[0].matched_document == "copy.pdf"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    def test_quick_watch_document(self, tmp_db):
        entry_id = quick_watch_document(tmp_db, "essay.pdf", label="Monitor essay")
        repo = SimilarityWatchlistRepository(tmp_db)
        entry = repo.get_entry(entry_id)
        assert entry is not None
        assert entry.target == "essay.pdf"

    def test_quick_watch_pair(self, tmp_db):
        entry_id = quick_watch_pair(tmp_db, "a.pdf", "b.pdf")
        repo = SimilarityWatchlistRepository(tmp_db)
        entry = repo.get_entry(entry_id)
        assert entry.watchlist_type == WatchlistType.PAIR
        assert entry.target == "a.pdf|b.pdf"


# ---------------------------------------------------------------------------
# WatchlistSummary
# ---------------------------------------------------------------------------

class TestWatchlistSummary:
    def test_to_dict(self):
        summary = WatchlistSummary(
            total_entries=10,
            active_entries=8,
            paused_entries=1,
            resolved_entries=1,
            total_alerts=5,
            unacknowledged_alerts=3,
            entries_by_type={"document": 7, "pair": 3},
            alerts_by_severity={"High": 2, "Medium": 3},
            recent_alerts=[],
        )
        d = summary.to_dict()
        assert d["total_entries"] == 10
        assert d["entries_by_type"]["document"] == 7
