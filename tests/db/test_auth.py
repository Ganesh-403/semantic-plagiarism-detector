import sqlite3
import uuid

import pytest

from src.db.auth import (
    add_user,
    delete_user,
    disable_2fa,
    enable_2fa,
    get_2fa_status,
    get_active_users_count,
    get_user_active_status,
    get_user_role,
    get_user_theme,
    init_db,
    is_user_active,
    log_security_event,
    set_user_active_status,
    set_user_theme,
    update_password,
    get_security_audit_logs,
    get_security_audit_log_count,
    get_distinct_audit_event_types,
    verify_user,
    update_user_profile,
    get_all_users,
    format_user_created_date,
)
from src.errors import StaleDataException


@pytest.fixture(autouse=True)
def setup_test_db(mock_db):
    """Uses the mock_db fixture from conftest.py to isolate DB operations."""
    init_db()
    yield


def test_init_db():
    init_db()
    assert verify_user("admin", "Admin123!") is True
    assert verify_user("admin", "wrongpassword") is False


def test_add_user():
    user = f"user_{uuid.uuid4().hex[:8]}"
    add_user(user, "SecurePass123!")
    check = get_user_role(user)
    assert check is not None


def test_duplicate_user():
    user = f"user_{uuid.uuid4().hex[:8]}"
    add_user(user, "SecurePass123!")
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        add_user(user, "SecurePass123!")


def test_verify_user():
    user = f"user_{uuid.uuid4().hex[:8]}"
    add_user(user, "SecurePass123!")
    assert verify_user(user, "SecurePass123!") is True
    assert verify_user(user, "WrongPass123!") is False


def test_get_user_role():
    user = f"user_{uuid.uuid4().hex[:8]}"
    add_user(user, "password123")
    assert get_user_role(user) is not None
    assert get_user_role("non_existent_user_999") is None


def test_update_password():
    user = f"user_{uuid.uuid4().hex[:8]}"
    add_user(user, "password123")
    update_password(user, "new_secret_123!")
    assert verify_user(user, "new_secret_123!") is True


def test_delete_user():
    delete_user("hnsdf9")
    assert get_user_role("hnsdf9") is None


import unittest.mock as mock


@pytest.fixture
def mock_audit_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE security_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            username TEXT,
            timestamp DATETIME,
            details TEXT
        )
    """)
    conn.execute(
        "INSERT INTO security_audit_log (event_type, username, timestamp) VALUES ('login', 'alice', '2023-01-01 10:00:00')"
    )
    conn.execute(
        "INSERT INTO security_audit_log (event_type, username, timestamp) VALUES ('login', 'bob', '2023-01-02 10:00:00')"
    )
    conn.execute(
        "INSERT INTO security_audit_log (event_type, username, timestamp) VALUES ('logout', 'alice', '2023-01-03 10:00:00')"
    )
    conn.commit()

    with mock.patch("src.db.auth._connect", return_value=conn):
        yield conn
    conn.close()


def test_get_security_audit_logs_default(mock_audit_db):
    logs = get_security_audit_logs()
    assert len(logs) == 3
    # Order by timestamp DESC
    assert logs[0]["username"] == "alice"
    assert logs[0]["event_type"] == "logout"
    assert logs[2]["username"] == "alice"
    assert logs[2]["event_type"] == "login"


def test_get_security_audit_logs_pagination(mock_audit_db):
    logs = get_security_audit_logs(limit=1, offset=1)
    assert len(logs) == 1
    # 2nd in desc order is bob
    assert logs[0]["username"] == "bob"


def test_get_security_audit_logs_username_filter(mock_audit_db):
    logs = get_security_audit_logs(username="alice")
    assert len(logs) == 2
    assert logs[0]["event_type"] == "logout"
    assert logs[1]["event_type"] == "login"


def test_get_security_audit_logs_empty(mock_audit_db):
    logs = get_security_audit_logs(username="charlie")
    assert len(logs) == 0


def test_get_security_audit_logs_invalid_limit_offset(mock_audit_db):
    with pytest.raises(ValueError):
        get_security_audit_logs(limit=-1)
    with pytest.raises(ValueError):
        get_security_audit_logs(offset=-1)


def test_get_security_audit_logs_date_filter(mock_audit_db):
    logs = get_security_audit_logs(
        start_date="2023-01-02 00:00:00", end_date="2023-01-02 23:59:59"
    )
    assert len(logs) == 1
    assert logs[0]["username"] == "bob"


def test_get_security_audit_log_count(mock_audit_db):
    assert get_security_audit_log_count() == 3
    assert get_security_audit_log_count(username="alice") == 2
    assert get_security_audit_log_count(event_type="logout") == 1


def test_get_distinct_audit_event_types(mock_audit_db):
    events = get_distinct_audit_event_types()
    assert set(events) == {"login", "logout"}


def test_2fa_flow():
    username = f"user2fa_{uuid.uuid4().hex[:8]}"
    add_user(username, "pass1234567!")

    enabled, secret = get_2fa_status(username)
    assert enabled is False
    assert secret is None

    test_secret = "JBSWY3DPEHPK3PXP"
    enable_2fa(username, test_secret)

    enabled, secret = get_2fa_status(username)
    assert enabled is True
    assert secret == test_secret

    disable_2fa(username)

    enabled, secret = get_2fa_status(username)
    assert enabled is False
    assert secret is None

    delete_user(username)


def test_suspend_account():
    username = f"user_{uuid.uuid4().hex[:8]}"
    add_user(username, "password123!")

    # Verify default is active
    assert get_user_active_status(username) is True
    assert is_user_active(username) is True

    # Suspend user
    set_user_active_status(username, False)
    assert get_user_active_status(username) is False
    assert is_user_active(username) is False

    # Try suspending default 'admin' user (must raise ValueError)
    with pytest.raises(ValueError, match="The admin account cannot be suspended."):
        set_user_active_status("admin", False)

    # Reactivate user
    set_user_active_status(username, True)
    assert get_user_active_status(username) is True
    assert is_user_active(username) is True

    delete_user(username)


def test_sqlite_file_lock_exception(mock_db):
    """Test that acquiring an exclusive lock on SQLite database triggers a clean sqlite3.Error when attempting add_user."""
    import src.db.auth

    conn = sqlite3.connect(src.db.auth._DB_PATH, timeout=0.1)
    conn.execute("BEGIN EXCLUSIVE TRANSACTION")
    conn.execute("INSERT INTO users (username, password) VALUES ('lock_dummy', 'pass')")
    try:
        with pytest.raises(sqlite3.Error) as exc_info:
            add_user("locked_user", "password123!")
        assert "Failed to add user" in str(exc_info.value) or "locked" in str(
            exc_info.value
        )
    finally:
        conn.rollback()
        conn.close()


def test_user_theme(mock_db):
    """Test get and set theme for a user."""
    user = f"theme_user_{uuid.uuid4().hex[:8]}"
    add_user(user, "password123!")

    # Default should be light
    assert get_user_theme(user) == "light"

    # Set to dark
    set_user_theme(user, "dark")
    assert get_user_theme(user) == "dark"


def test_delete_user_removes_user_row_and_audit_log(mock_db):
    """delete_user() must remove the user row and associated security_audit_log entries."""
    import src.db.auth

    user = f"user_{uuid.uuid4().hex[:8]}"
    add_user(user, "password123")

    # Seed an audit log entry for this user
    log_security_event("password_change", user, "test entry")

    # Confirm the audit entry exists before deletion
    with sqlite3.connect(src.db.auth._DB_PATH) as conn:
        audit_before = conn.execute(
            "SELECT COUNT(*) FROM security_audit_log WHERE username = ?", (user,)
        ).fetchone()[0]
    assert audit_before >= 1

    delete_user(user)

    # User row must be gone
    assert get_user_role(user) is None

    # Audit log entries for the deleted user must also be removed
    with sqlite3.connect(src.db.auth._DB_PATH) as conn:
        audit_after = conn.execute(
            "SELECT COUNT(*) FROM security_audit_log WHERE username = ?", (user,)
        ).fetchone()[0]
    assert audit_after == 0


def test_delete_user_removes_matching_session_and_authorization_rows(mock_db):
    """delete_user() should remove matching session and authorization rows for the deleted user."""
    import src.db.auth

    user = f"user_{uuid.uuid4().hex[:8]}"
    add_user(user, "password123")

    with sqlite3.connect(src.db.auth._DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                session_state TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS authorization_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                token TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO user_sessions (username, session_state) VALUES (?, ?)",
            (user, '{"page": "dashboard"}'),
        )
        conn.execute(
            "INSERT INTO authorization_tokens (username, token) VALUES (?, ?)",
            (user, "token-for-user"),
        )
        conn.execute(
            "INSERT INTO user_sessions (username, session_state) VALUES (?, ?)",
            ("other_user", '{"page": "dashboard"}'),
        )
        conn.execute(
            "INSERT INTO authorization_tokens (username, token) VALUES (?, ?)",
            ("other_user", "token-for-other"),
        )
        conn.commit()

    delete_user(user)

    with sqlite3.connect(src.db.auth._DB_PATH) as conn:
        user_session_count = conn.execute(
            "SELECT COUNT(*) FROM user_sessions WHERE username = ?", (user,)
        ).fetchone()[0]
        user_token_count = conn.execute(
            "SELECT COUNT(*) FROM authorization_tokens WHERE username = ?", (user,)
        ).fetchone()[0]
        other_session_count = conn.execute(
            "SELECT COUNT(*) FROM user_sessions WHERE username = ?", ("other_user",)
        ).fetchone()[0]
        other_token_count = conn.execute(
            "SELECT COUNT(*) FROM authorization_tokens WHERE username = ?",
            ("other_user",),
        ).fetchone()[0]

    assert get_user_role(user) is None
    assert user_session_count == 0
    assert user_token_count == 0
    assert other_session_count == 1
    assert other_token_count == 1


# ──────────────────────────────────────────────────────────────────────────────
# format_user_created_date — issue #1049
# ──────────────────────────────────────────────────────────────────────────────


def test_connect_uses_fifteen_second_timeout():
    """Verify that _connect helper sets sqlite3 timeout to 15.0 seconds."""
    from unittest.mock import patch
    from src.db.auth import _connect, SQLITE_TIMEOUT

    assert SQLITE_TIMEOUT == 15.0

    with patch("sqlite3.connect") as mock_connect:
        _connect()
        mock_connect.assert_called_once()
        kwargs = mock_connect.call_args[1]
        assert kwargs.get("timeout") == 15.0


class TestFormatUserCreatedDate:
    """Tests for the format_user_created_date helper (issue #1049)."""

    def test_valid_iso_datetime_with_z(self):
        """Full ISO datetime with Z suffix must format correctly."""
        result = format_user_created_date("2026-07-28T14:30:00Z")
        assert result == "Jul 28, 2026"

    def test_valid_iso_datetime_without_z(self):
        """ISO datetime without Z suffix must format correctly."""
        result = format_user_created_date("2026-07-28T14:30:00")
        assert result == "Jul 28, 2026"

    def test_valid_date_only(self):
        """Date-only ISO string must format correctly."""
        result = format_user_created_date("2026-07-28")
        assert result == "Jul 28, 2026"

    def test_valid_space_separated(self):
        """Space-separated datetime (SQLite default) must format correctly."""
        result = format_user_created_date("2026-07-28 14:30:00")
        assert result == "Jul 28, 2026"

    def test_different_date(self):
        """Verify month and day mapping for a different date."""
        result = format_user_created_date("2025-01-05")
        assert result == "Jan 05, 2025"

    def test_empty_string_returns_unknown(self):
        """Empty string must return 'Unknown'."""
        assert format_user_created_date("") == "Unknown"

    def test_none_returns_unknown(self):
        """None input must return 'Unknown'."""
        assert format_user_created_date(None) == "Unknown"  # type: ignore[arg-type]

    def test_whitespace_only_returns_unknown(self):
        """Whitespace-only string must return 'Unknown'."""
        assert format_user_created_date("   ") == "Unknown"

    def test_invalid_string_returns_unknown(self):
        """Garbage input must return 'Unknown', not raise."""
        assert format_user_created_date("not-a-date") == "Unknown"

    def test_partial_invalid_returns_unknown(self):
        """Partially valid input must return 'Unknown'."""
        assert format_user_created_date("2026-13-45") == "Unknown"

    def test_returns_str_type(self):
        """Return type must always be str."""
        result = format_user_created_date("2026-07-28")
        assert isinstance(result, str)

    def test_non_string_input_returns_unknown(self):
        """Non-string input (int, list) must return 'Unknown'."""
        assert format_user_created_date(12345) == "Unknown"  # type: ignore[arg-type]
        assert format_user_created_date([]) == "Unknown"  # type: ignore[arg-type]


def test_get_active_users_count():
    """Verify get_active_users_count counts only active users."""
    # 1. Starting count should be 1 (the default seeded 'admin' is active)
    initial_count = get_active_users_count()
    assert initial_count == 1

    # 2. Add an active user
    user1 = f"active_{uuid.uuid4().hex[:8]}"
    add_user(user1, "SecurePass123!")
    assert get_active_users_count() == 2

    # 3. Add another user and suspend them
    user2 = f"suspended_{uuid.uuid4().hex[:8]}"
    add_user(user2, "SecurePass123!")
    set_user_active_status(user2, False)
    # The count should still be 2 because user2 is inactive
    assert get_active_users_count() == 2

    # 4. Reactivate user2
    set_user_active_status(user2, True)
    assert get_active_users_count() == 3

    # 5. Delete user1
    delete_user(user1)
    assert get_active_users_count() == 2

    delete_user(user2)


def test_update_user_profile_success():
    """Verify update_user_profile successfully updates role/active status and increments version."""
    user = f"profile_user_{uuid.uuid4().hex[:8]}"
    add_user(user, "SecurePass123!", "teacher")

    # Get initial version
    users = get_all_users()
    user_data = next(u for u in users if u["username"] == user)
    assert user_data["version"] == 1
    assert user_data["role"] == "teacher"
    assert user_data["is_active"] is True

    # Update profile
    update_user_profile(user, role="admin", is_active=False, expected_version=1)

    # Verify updates and version increment
    users = get_all_users()
    user_data = next(u for u in users if u["username"] == user)
    assert user_data["version"] == 2
    assert user_data["role"] == "admin"
    assert user_data["is_active"] is False


def test_update_user_profile_stale_version():
    """Verify update_user_profile raises StaleDataException on version mismatch."""
    user = f"profile_user_{uuid.uuid4().hex[:8]}"
    add_user(user, "SecurePass123!", "teacher")

    # Update profile with incorrect version
    with pytest.raises(StaleDataException) as exc_info:
        update_user_profile(user, role="admin", is_active=True, expected_version=2)
    assert "Conflict detected" in str(exc_info.value) or "Expected version" in str(
        exc_info.value
    )

    # Verify that the role remains 'teacher'
    assert get_user_role(user) == "teacher"


def test_update_user_profile_non_existent():
    """Verify update_user_profile raises ValueError for non-existent user."""
    with pytest.raises(ValueError, match="User not found."):
        update_user_profile(
            "non_existent_username_xyz",
            role="teacher",
            is_active=True,
            expected_version=1,
        )


def test_update_user_profile_admin_suspension_prevented():
    """Verify update_user_profile prevents suspending the admin account."""
    users = get_all_users()
    admin_data = next(u for u in users if u["username"] == "admin")
    admin_ver = admin_data["version"]

    with pytest.raises(ValueError, match="The admin account cannot be suspended."):
        update_user_profile(
            "admin", role="admin", is_active=False, expected_version=admin_ver
        )


def test_get_all_users_filters_by_role():
    """Verify get_all_users filters users by role when specified."""
    admin_user = f"admin_{uuid.uuid4().hex[:8]}"
    teacher_user = f"teacher_{uuid.uuid4().hex[:8]}"
    add_user(admin_user, "SecurePass123!", "admin")
    add_user(teacher_user, "SecurePass123!", "teacher")

    all_users = get_all_users()
    assert any(u["username"] == admin_user for u in all_users)
    assert any(u["username"] == teacher_user for u in all_users)

    admin_users = get_all_users(role="admin")
    assert all(u["role"] == "admin" for u in admin_users)
    assert any(u["username"] == admin_user for u in admin_users)
    assert not any(u["username"] == teacher_user for u in admin_users)

    teacher_users = get_all_users(role="teacher")
    assert all(u["role"] == "teacher" for u in teacher_users)
    assert any(u["username"] == teacher_user for u in teacher_users)
    assert not any(u["username"] == admin_user for u in teacher_users)


def test_get_all_users_role_no_matches():
    """Verify get_all_users returns an empty list when no user matches the role."""
    assert get_all_users(role="nonexistent_role") == []


def test_revoke_token_and_is_token_revoked():
    """Verify revoke_token stores token signature and is_token_revoked checks it correctly."""
    from src.db.auth import is_token_revoked, revoke_token

    test_token = f"token_{uuid.uuid4().hex}"
    assert is_token_revoked(test_token) is False

    revoke_token(test_token, details="Test revocation")
    assert is_token_revoked(test_token) is True

    # Empty token edge cases
    assert is_token_revoked("") is False
    assert is_token_revoked(None) is False  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        revoke_token("")


def test_password_history_validation_prevents_reuse_of_last_3_passwords(mock_db):
    """Verify update_password prevents reusing any of the last 3 passwords."""
    user = f"hist_user_{uuid.uuid4().hex[:8]}"
    pass1 = "Pass_111111!"
    pass2 = "Pass_222222!"
    pass3 = "Pass_333333!"
    pass4 = "Pass_444444!"

    # 1. Add user with pass1
    add_user(user, pass1)

    # 2. Update to pass2
    update_password(user, pass2)
    assert verify_user(user, pass2) is True

    # 3. Update to pass3
    update_password(user, pass3)
    assert verify_user(user, pass3) is True

    # 4. Attempting to reuse pass1, pass2, or pass3 must raise ValueError
    for forbidden_pass in (pass1, pass2, pass3):
        with pytest.raises(ValueError) as exc_info:
            update_password(user, forbidden_pass)
        assert "New password cannot be one of your last 3 passwords" in str(exc_info.value)

    # 5. Update to pass4 (succeeds)
    update_password(user, pass4)
    assert verify_user(user, pass4) is True

    # 6. Now pass1 is older than the last 3 passwords (which are pass4, pass3, pass2) -> updating to pass1 succeeds
    update_password(user, pass1)
    assert verify_user(user, pass1) is True


def test_get_recent_audit_events(mock_db):
    """Verify get_recent_audit_events returns recent audit entries ordered by timestamp DESC up to limit."""
    from src.db.auth import get_recent_audit_events, log_security_event

    log_security_event("login_success", "alice", "Alice logged in")
    log_security_event("login_failure", "bob", "Bob failed login")
    log_security_event("password_change", "charlie", "Charlie updated password")

    events = get_recent_audit_events(limit=2)
    assert len(events) == 2
    assert isinstance(events, list)
    assert isinstance(events[0], dict)

    # Validate keys in dictionary
    for event in events:
        assert "id" in event
        assert "event_type" in event
        assert "username" in event
        assert "timestamp" in event
        assert "details" in event

    # Default limit=20 returns all logged events
    all_recent = get_recent_audit_events(limit=20)
    assert len(all_recent) >= 3

    # Negative limit raises ValueError
    with pytest.raises(ValueError):
        get_recent_audit_events(limit=-5)


def test_password_change_required_flag(mock_db):
    """Verify set_password_change_required sets/clears the flag, and verify_user returns it."""
    from src.db.auth import set_password_change_required

    username = f"flaguser_{uuid.uuid4().hex[:8]}"
    password = "Secure_Pass123!"
    add_user(username, password)

    # 1. Standard call returns True
    assert verify_user(username, password) is True

    # 2. By default must_change_password is False (0)
    result = verify_user(username, password, return_details=True)
    assert isinstance(result, dict)
    assert result["authenticated"] is True
    assert result["must_change_password"] is False

    # 3. Admin sets must_change_password = True
    set_password_change_required(username, required=True)
    result = verify_user(username, password, return_details=True)
    assert isinstance(result, dict)
    assert result["authenticated"] is True
    assert result["must_change_password"] is True

    # 4. Clear the flag back to False
    set_password_change_required(username, required=False)
    result = verify_user(username, password, return_details=True)
    assert result["must_change_password"] is False

    # 5. Setting flag on non-existent user raises ValueError
    with pytest.raises(ValueError):
        set_password_change_required("nonexistent_user_xyz", required=True)

    # 6. Invalid credentials still return False (or dict with authenticated=False)
    assert verify_user(username, "WrongPassword!") is False
    assert verify_user(username, "WrongPassword!", return_details=True) == {"authenticated": False, "must_change_password": False}
