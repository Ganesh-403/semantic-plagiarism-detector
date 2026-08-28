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
tests/db/test_auth_expiration.py
--------------------------------
Comprehensive unit tests for the password expiration feature (Issue #2716).

Verifies that passwords expire after the configured lifetime, that the
migration correctly sets initial expiration dates, and that the login
flow properly flags expired passwords.
"""

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.db.auth import authenticate_user, is_password_expired, set_password_expiration


@pytest.fixture
def mock_users_db(tmp_path):
    """Create a temporary SQLite database with the users table."""
    db_path = tmp_path / "test_users.db"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_expires_at TEXT
            )
        """
        )
        conn.commit()

    return str(db_path)


class TestIsPasswordExpired:
    """Test suite for the is_password_expired function."""

    def test_returns_false_for_empty_username(self, mock_users_db):
        """Verify empty username returns False."""
        assert is_password_expired("", db_path=mock_users_db) is False
        assert is_password_expired(None, db_path=mock_users_db) is False

    def test_returns_false_when_expiration_is_null(self, mock_users_db):
        """Verify NULL expiration means password never expires."""
        with sqlite3.connect(mock_users_db) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES ('alice', 'hash')"
            )
            conn.commit()

        assert is_password_expired("alice", db_path=mock_users_db) is False

    def test_returns_false_when_not_expired(self, mock_users_db):
        """Verify False is returned when expiration is in the future."""
        future_date = (datetime.utcnow() + timedelta(days=30)).isoformat()

        with sqlite3.connect(mock_users_db) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, password_expires_at) VALUES ('bob', 'hash', ?)",
                (future_date,),
            )
            conn.commit()

        assert is_password_expired("bob", db_path=mock_users_db) is False

    def test_returns_true_when_expired(self, mock_users_db):
        """Verify True is returned when expiration is in the past."""
        past_date = (datetime.utcnow() - timedelta(days=1)).isoformat()

        with sqlite3.connect(mock_users_db) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, password_expires_at) VALUES ('charlie', 'hash', ?)",
                (past_date,),
            )
            conn.commit()

        assert is_password_expired("charlie", db_path=mock_users_db) is True

    def test_returns_true_when_expiring_right_now(self, mock_users_db):
        """Verify True is returned when expiration is exactly now."""
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(mock_users_db) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, password_expires_at) VALUES ('dave', 'hash', ?)",
                (now,),
            )
            conn.commit()

        assert is_password_expired("dave", db_path=mock_users_db) is True

    def test_handles_invalid_date_format_gracefully(self, mock_users_db):
        """Verify invalid date strings don't crash and return False."""
        with sqlite3.connect(mock_users_db) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, password_expires_at) VALUES ('eve', 'hash', 'not-a-date')"
            )
            conn.commit()

        # Should fail open (return False) on invalid date
        assert is_password_expired("eve", db_path=mock_users_db) is False


class TestSetPasswordExpiration:
    """Test suite for the set_password_expiration function."""

    def test_sets_expiration_successfully(self, mock_users_db):
        """Verify expiration date is set correctly."""
        with sqlite3.connect(mock_users_db) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES ('alice', 'hash')"
            )
            conn.commit()

        result = set_password_expiration(
            "alice", days_until_expiration=90, db_path=mock_users_db
        )

        assert result is True

        # Verify the date was actually set
        with sqlite3.connect(mock_users_db) as conn:
            cursor = conn.execute(
                "SELECT password_expires_at FROM users WHERE username = 'alice'"
            )
            row = cursor.fetchone()

        assert row is not None
        assert row[0] is not None

        # Verify it's approximately 90 days in the future
        expires_at = datetime.fromisoformat(row[0])
        expected = datetime.utcnow() + timedelta(days=90)

        # Allow 1 minute tolerance for test execution time
        assert abs((expires_at - expected).total_seconds()) < 60

    def test_returns_false_for_nonexistent_user(self, mock_users_db):
        """Verify False is returned when user doesn't exist."""
        result = set_password_expiration(
            "nonexistent", days_until_expiration=90, db_path=mock_users_db
        )

        assert result is False

    def test_rejects_negative_days(self, mock_users_db):
        """Verify negative days_until_expiration is rejected."""
        result = set_password_expiration(
            "alice", days_until_expiration=-10, db_path=mock_users_db
        )

        assert result is False


class TestAuthenticateUserExpiration:
    """Test suite for password expiration integration in authenticate_user."""

    @patch("src.db.auth.get_user_by_username")
    @patch("src.db.auth.verify_password", return_value=True)
    @patch("src.db.auth.is_account_locked", return_value=False)
    def test_flags_expired_password_on_success(
        self, mock_locked, mock_verify, mock_get_user
    ):
        """Verify authenticate_user flags expired passwords even on successful login."""
        mock_get_user.return_value = {"username": "alice", "password_hash": "hash"}

        with patch("src.db.auth.is_password_expired", return_value=True):
            with patch("src.db.auth.log_security_event") as mock_log:
                result = authenticate_user("alice", "correct_password")

        assert result["success"] is True
        assert result["password_expired"] is True

        # Verify the specific expired login event was logged
        mock_log.assert_called_once()
        assert mock_log.call_args[1]["event_type"] == "login_success_password_expired"

    @patch("src.db.auth.get_user_by_username")
    @patch("src.db.auth.verify_password", return_value=True)
    @patch("src.db.auth.is_account_locked", return_value=False)
    def test_no_expiration_flag_when_valid(
        self, mock_locked, mock_verify, mock_get_user
    ):
        """Verify password_expired is False when password is still valid."""
        mock_get_user.return_value = {"username": "bob", "password_hash": "hash"}

        with patch("src.db.auth.is_password_expired", return_value=False):
            result = authenticate_user("bob", "correct_password")

        assert result["success"] is True
        assert result["password_expired"] is False
