import sqlite3
import uuid

import pytest

from src.db.auth import (add_user, delete_user, disable_2fa, enable_2fa,
                         get_2fa_status, get_user_active_status, get_user_role,
                         get_user_theme, init_db, is_user_active,
                         set_user_active_status, set_user_theme,
                         update_password, verify_user)


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
    user = f"user_{uuid.uuid4().hex[:8]}"
    add_user(user, "password123")
    delete_user(user)
    assert get_user_role(user) is None


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
    conn = sqlite3.connect(mock_db)
    conn.execute("BEGIN EXCLUSIVE TRANSACTION")
    try:
        with pytest.raises(sqlite3.Error) as exc_info:
            add_user("locked_user", "password123!")
        assert "Failed to add user" in str(exc_info.value) or "locked" in str(exc_info.value)
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


def test_get_all_users():
    """Test that get_all_users returns User Pydantic model DTOs."""
    from src.db.schemas import User
    from src.db.auth import get_all_users
    users = get_all_users()
    assert len(users) >= 1
    assert all(isinstance(u, User) for u in users)
    admin_user = next(u for u in users if u["username"] == "admin")
    assert admin_user["role"] == "admin"
    assert admin_user.role == "admin"
    assert admin_user.get("is_active") is True

