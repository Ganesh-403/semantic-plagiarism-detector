"""
tests/db/test_auth_lockout.py
-----------------------------
Comprehensive unit tests for the account lockout mechanism (Issue #2704).

Verifies that accounts are temporarily locked after N failed login attempts
within a configured time window, and that successful logins are blocked
during the lockout period.
"""

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.db.auth import (
    MAX_FAILED_ATTEMPTS,
    authenticate_user,
    is_account_locked,
)
from src.db.security_audit import count_recent_failed_logins


@pytest.fixture
def mock_audit_db(tmp_path):
    """Create a temporary SQLite database with the security_audit_log table."""
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
        conn.commit()

    return str(db_path)


class TestCountRecentFailedLogins:
    """Test suite for the count_recent_failed_logins function."""

    def test_returns_zero_for_empty_username(self, mock_audit_db):
        """Verify empty username returns 0."""
        assert count_recent_failed_logins("", db_path=mock_audit_db) == 0
        assert count_recent_failed_logins(None, db_path=mock_audit_db) == 0

    def test_counts_failures_within_window(self, mock_audit_db):
        """Verify only failures within the time window are counted."""
        now = datetime.utcnow()

        with sqlite3.connect(mock_audit_db) as conn:
            # 3 failures within the last 5 minutes
            for i in range(3):
                ts = (now - timedelta(minutes=i)).isoformat()
                conn.execute(
                    "INSERT INTO security_audit_log (timestamp, event_type, username) VALUES (?, 'login_failed', 'alice')",
                    (ts,),
                )

            # 2 failures outside the 15-minute window (20 mins ago)
            old_ts = (now - timedelta(minutes=20)).isoformat()
            conn.execute(
                "INSERT INTO security_audit_log (timestamp, event_type, username) VALUES (?, 'login_failed', 'alice')",
                (old_ts,),
            )
            conn.commit()

        # Default window is 15 minutes, should only count the 3 recent ones
        count = count_recent_failed_logins("alice", db_path=mock_audit_db)
        assert count == 3

    def test_custom_window_minutes(self, mock_audit_db):
        """Verify custom window_minutes parameter is respected."""
        now = datetime.utcnow()

        with sqlite3.connect(mock_audit_db) as conn:
            # 1 failure 10 minutes ago
            ts = (now - timedelta(minutes=10)).isoformat()
            conn.execute(
                "INSERT INTO security_audit_log (timestamp, event_type, username) VALUES (?, 'login_failed', 'bob')",
                (ts,),
            )
            conn.commit()

        # 5 minute window should miss it
        assert (
            count_recent_failed_logins("bob", window_minutes=5, db_path=mock_audit_db)
            == 0
        )

        # 15 minute window should catch it
        assert (
            count_recent_failed_logins("bob", window_minutes=15, db_path=mock_audit_db)
            == 1
        )

    def test_ignores_non_failed_login_events(self, mock_audit_db):
        """Verify only 'login_failed' events are counted."""
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(mock_audit_db) as conn:
            conn.execute(
                "INSERT INTO security_audit_log (timestamp, event_type, username) VALUES (?, 'login_success', 'charlie')",
                (now,),
            )
            conn.execute(
                "INSERT INTO security_audit_log (timestamp, event_type, username) VALUES (?, 'password_changed', 'charlie')",
                (now,),
            )
            conn.commit()

        assert count_recent_failed_logins("charlie", db_path=mock_audit_db) == 0

    def test_case_insensitive_username_matching(self, mock_audit_db):
        """Verify username matching is case-insensitive."""
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(mock_audit_db) as conn:
            conn.execute(
                "INSERT INTO security_audit_log (timestamp, event_type, username) VALUES (?, 'login_failed', 'alice')",
                (now,),
            )
            conn.commit()

        # Query with uppercase should still match the lowercase DB entry
        # Note: The function lowercases the input before querying
        assert count_recent_failed_logins("ALICE", db_path=mock_audit_db) == 1


class TestIsAccountLocked:
    """Test suite for the is_account_locked function."""

    def test_not_locked_below_threshold(self, mock_audit_db):
        """Verify account is not locked when failures are below threshold."""
        with patch(
            "src.db.auth.count_recent_failed_logins",
            return_value=MAX_FAILED_ATTEMPTS - 1,
        ):
            assert is_account_locked("alice") is False

    def test_locked_at_threshold(self, mock_audit_db):
        """Verify account is locked when failures exactly hit the threshold."""
        with patch(
            "src.db.auth.count_recent_failed_logins", return_value=MAX_FAILED_ATTEMPTS
        ):
            assert is_account_locked("alice") is True

    def test_locked_above_threshold(self, mock_audit_db):
        """Verify account is locked when failures exceed the threshold."""
        with patch(
            "src.db.auth.count_recent_failed_logins",
            return_value=MAX_FAILED_ATTEMPTS + 5,
        ):
            assert is_account_locked("alice") is True

    def test_empty_username_never_locked(self):
        """Verify empty username always returns False."""
        assert is_account_locked("") is False
        assert is_account_locked(None) is False


class TestAuthenticateUserLockoutIntegration:
    def test_successful_login_when_not_locked(self, mock_db):
        from src.db import auth
        auth.add_user("lockout-user", "StrongPassword9!")
        assert auth.authenticate_user("lockout-user", "StrongPassword9!") is True
        assert auth.get_security_audit_logs(username="lockout-user", event_type="login_success")

    def test_failed_logins_trigger_lockout_before_password_verification(self, mock_db):
        from src.db import auth
        auth.add_user("lockout-user", "StrongPassword9!")
        for _ in range(MAX_FAILED_ATTEMPTS):
            assert auth.authenticate_user("lockout-user", "WrongPassword9!") is False
        assert auth.is_account_locked("lockout-user")
        with patch.object(auth, "_ph") as hasher:
            assert auth.authenticate_user("lockout-user", "StrongPassword9!") is False
        hasher.verify.assert_not_called()
        assert auth.get_security_audit_logs(username="lockout-user", event_type="login_blocked_lockout")

    def test_failed_login_logs_failure(self, mock_db):
        from src.db import auth
        assert auth.authenticate_user("nonexistent", "StrongPassword9!") is False
        assert auth.get_security_audit_logs(username="nonexistent", event_type="login_failed")
