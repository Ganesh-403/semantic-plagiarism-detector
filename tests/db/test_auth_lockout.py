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

from src.db.auth import MAX_FAILED_ATTEMPTS, authenticate_user, is_account_locked
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
    """Test suite for lockout integration in authenticate_user."""

    @patch("src.db.auth.get_user_by_username")
    @patch("src.db.auth.verify_password", return_value=True)
    def test_successful_login_when_not_locked(self, mock_verify, mock_get_user):
        """Verify successful login proceeds normally when account is not locked."""
        mock_get_user.return_value = {"username": "alice", "password_hash": "hash"}

        with patch("src.db.auth.is_account_locked", return_value=False):
            with patch("src.db.auth.log_security_event") as mock_log:
                result = authenticate_user("alice", "correct_password")

        assert result is True
        mock_verify.assert_called_once()
        # Verify success was logged
        mock_log.assert_any_call(
            event_type="login_success",
            username="alice",
            details="Successful authentication",
        )

    @patch("src.db.auth.get_user_by_username")
    @patch("src.db.auth.verify_password")
    def test_login_blocked_when_locked(self, mock_verify, mock_get_user):
        """Verify login is blocked and password is NOT checked when account is locked."""
        with patch("src.db.auth.is_account_locked", return_value=True):
            with patch("src.db.auth.log_security_event") as mock_log:
                result = authenticate_user("alice", "any_password")

        assert result is False
        # Password verification should NOT even be attempted
        mock_verify.assert_not_called()
        mock_get_user.assert_not_called()

        # Verify lockout event was logged
        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args[1]
        assert call_kwargs["event_type"] == "login_blocked_lockout"
        assert "lockout" in call_kwargs["details"].lower()

    @patch("src.db.auth.get_user_by_username", return_value=None)
    def test_failed_login_logs_failure(self, mock_get_user):
        """Verify failed login (user not found) logs a login_failed event."""
        with patch("src.db.auth.is_account_locked", return_value=False):
            with patch("src.db.auth.log_security_event") as mock_log:
                result = authenticate_user("nonexistent", "password")

        assert result is False
        mock_log.assert_called_once()
        assert mock_log.call_args[1]["event_type"] == "login_failed"
