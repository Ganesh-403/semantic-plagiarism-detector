"""
tests/db/test_security_audit_pagination.py
------------------------------------------
Unit tests for audit event pagination logic (Issue #2732).
"""

import sqlite3

import pytest

from src.db.security_audit import get_audit_events_count, get_recent_audit_events


@pytest.fixture
def audit_db(tmp_path):
    """Create a temporary audit database with 50 sample events."""
    db_path = tmp_path / "test_audit.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE security_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                username TEXT,
                details TEXT
            )
        """
        )

        # Insert 50 events with sequential timestamps
        for i in range(50):
            ts = f"2024-01-01T{10 + (i // 60):02d}:{i % 60:02d}:00"
            conn.execute(
                "INSERT INTO security_audit_log (timestamp, event_type, username, details) VALUES (?, ?, ?, ?)",
                (
                    ts,
                    "login_success" if i % 2 == 0 else "login_failed",
                    f"user_{i % 5}",
                    f"Event {i}",
                ),
            )
        conn.commit()

    return str(db_path)


class TestGetRecentAuditEventsPagination:
    """Test suite for the offset/limit pagination logic."""

    def test_default_limit_and_offset(self, audit_db):
        """Verify default limit=20 and offset=0 returns first 20 events."""
        events = get_recent_audit_events(db_path=audit_db)
        assert len(events) == 20

    def test_custom_limit(self, audit_db):
        """Verify custom limit parameter restricts result count."""
        events = get_recent_audit_events(limit=10, db_path=audit_db)
        assert len(events) == 10

    def test_offset_skips_events(self, audit_db):
        """Verify offset parameter skips the correct number of events."""
        # Get first page
        page1 = get_recent_audit_events(limit=10, offset=0, db_path=audit_db)
        # Get second page
        page2 = get_recent_audit_events(limit=10, offset=10, db_path=audit_db)

        # Ensure no overlap
        page1_ids = {e["id"] for e in page1}
        page2_ids = {e["id"] for e in page2}

        assert len(page1_ids.intersection(page2_ids)) == 0

    def test_offset_beyond_total_returns_empty(self, audit_db):
        """Verify offset greater than total records returns empty list."""
        events = get_recent_audit_events(limit=10, offset=100, db_path=audit_db)
        assert len(events) == 0

    def test_negative_offset_defaults_to_zero(self, audit_db):
        """Verify negative offset is treated as 0."""
        events = get_recent_audit_events(limit=10, offset=-5, db_path=audit_db)
        assert len(events) == 10

    def test_zero_limit_defaults_to_twenty(self, audit_db):
        """Verify limit < 1 defaults to 20."""
        events = get_recent_audit_events(limit=0, db_path=audit_db)
        assert len(events) == 20

    def test_pagination_with_filters(self, audit_db):
        """Verify pagination works correctly when combined with filters."""
        # Count total failed logins
        total_failed = get_audit_events_count(
            event_type="login_failed", db_path=audit_db
        )

        # Paginate through failed logins
        page1 = get_recent_audit_events(
            limit=10, offset=0, event_type="login_failed", db_path=audit_db
        )

        assert len(page1) <= 10
        assert all(e["event_type"] == "login_failed" for e in page1)


class TestGetAuditEventsCount:
    """Test suite for the total count helper function."""

    def test_count_all_events(self, audit_db):
        """Verify count returns total number of events."""
        count = get_audit_events_count(db_path=audit_db)
        assert count == 50

    def test_count_with_event_filter(self, audit_db):
        """Verify count respects event_type filter."""
        count = get_audit_events_count(event_type="login_failed", db_path=audit_db)
        assert count == 25  # Half of 50

    def test_count_with_username_filter(self, audit_db):
        """Verify count respects username filter."""
        count = get_audit_events_count(username="user_0", db_path=audit_db)
        assert count == 10  # 50 events / 5 users
