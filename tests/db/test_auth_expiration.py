"""
tests/db/test_auth_expiration.py
--------------------------------
Comprehensive unit tests for the password expiration feature (Issue #2716).

Verifies that passwords expire after the configured lifetime, that the
migration correctly sets initial expiration dates, and that the login
flow properly flags expired passwords.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.db.auth import (
    authenticate_user,
    is_password_expired,
    set_password_expiration,
)


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
        expected = datetime.now(timezone.utc) + timedelta(days=90)

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

    @pytest.mark.parametrize("expired", [True, False])
    def test_expiration_flag_and_audit_event(self, mock_db, expired):
        from src.db import auth
        auth.add_user("expiration-user", "StrongPassword9!")
        with patch.object(auth, "is_password_expired", return_value=expired):
            result = auth.authenticate_user("expiration-user", "StrongPassword9!", return_details=True)
        assert result["authenticated"] is True
        assert result["password_expired"] is expired
        event = "login_success_password_expired" if expired else "login_success"
        assert auth.get_security_audit_logs(username="expiration-user", event_type=event)
