import sqlite3
import uuid

import pytest

from src.db.auth import (add_user, delete_user, disable_2fa, enable_2fa,
                         get_2fa_status, get_user_active_status, get_user_role,
                         init_db, is_user_active, set_user_active_status,
                         update_password, verify_user)


@pytest.fixture(autouse=True)
def setup_test_db(mock_db):
    """Uses the mock_db fixture from conftest.py to isolate DB operations."""
    init_db()
    yield


# Calls the init_db function and then uses verify_user to check if default admin user created
def test_init_db():
    init_db()
    assert verify_user("admin", "admin123") is not False


# Adds new user via uuid and uses get_user_role to check if user added
def test_add_user():
    user = uuid.uuid4().hex
    add_user(user, "ac_123")
    check = get_user_role(user)
    assert check is not None


# Adds a user and then checks whether adding same user again raises exception
def test_duplicate_user():
    user = f"user_{uuid.uuid4().hex[:8]}"
    add_user(user, "password123")
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        add_user(user, "password123")


# Checks whether adding incorrect password returns False
def test_verify_user():
    user = f"user_{uuid.uuid4().hex[:8]}"
    add_user(user, "password123")
    assert verify_user(user, "password123") is True
    assert verify_user(user, "wrong_pass") is False


def test_get_user_role():
    user = f"user_{uuid.uuid4().hex[:8]}"
    add_user(user, "password123")
    assert get_user_role(user) is not None
    assert get_user_role("non_existent_user_999") is None


def test_update_password():
    user = f"user_{uuid.uuid4().hex[:8]}"
    add_user(user, "password123")
    update_password(user, "new_secret_123")
    assert verify_user(user, "new_secret_123") is not False


# Deletes a user and then verifies if it still exists
# No need to change the username as for each run since del is last operation and
# duplicate_user first it gets created and deleted for each run
def test_delete_user():
    delete_user("hnsdf9")
    assert get_user_role("hnsdf9") is None


def test_2fa_flow():
    username = "test2fauser"
    add_user(username, "pass123")

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
    add_user(username, "password123")

    # Verify default is active
    assert get_user_active_status(username) is True
    assert is_user_active(username) is True
    assert verify_user(username, "password123") is True

    # Suspend user
    set_user_active_status(username, False)
    assert get_user_active_status(username) is False
    assert is_user_active(username) is False
    assert verify_user(username, "password123") is False

    # Try suspending default 'admin' user (must raise ValueError)
    try:
        add_user("admin", "admin123", "admin")
    except ValueError:
        pass
    with pytest.raises(ValueError, match="The admin account cannot be suspended."):
        set_user_active_status("admin", False)

    # Reactivate user
    set_user_active_status(username, True)
    assert get_user_active_status(username) is True
    assert is_user_active(username) is True
    assert verify_user(username, "password123") is True

    delete_user(username)
    delete_user("admin")


def test_sqlite_file_lock_exception(mock_db):
    """Test that acquiring an exclusive lock on SQLite database triggers a clean sqlite3.Error when attempting add_user."""
    conn = sqlite3.connect(mock_db)
    conn.execute("BEGIN EXCLUSIVE TRANSACTION")
    try:
        with pytest.raises(sqlite3.Error) as exc_info:
            add_user("locked_user", "password123")
        assert "Failed to add user" in str(exc_info.value) or "locked" in str(
            exc_info.value
        )
    finally:
        conn.rollback()
        conn.close()


def test_user_theme(mock_db):
    """Test get and set theme for a user."""
    user = f"theme_user_{uuid.uuid4().hex[:8]}"
    add_user(user, "password123")
    
    from src.db.auth import get_user_theme, set_user_theme

    # Default should be light
    assert get_user_theme(user) == "light"
    
    # Set to dark
    set_user_theme(user, "dark")
    assert get_user_theme(user) == "dark"
    
    # Invalid themes should fallback to light
    set_user_theme(user, "purple")
    assert get_user_theme(user) == "light"

