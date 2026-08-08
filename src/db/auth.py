"""
src/db/auth.py
--------------
SQLite-backed authentication with Argon2 password hashing (via argon2-cffi),
automatic transparent migration from legacy bcrypt hashes, user login tracking,
and strong password complexity policies.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import sqlite3
from datetime import datetime as dt
from datetime import timezone

import bcrypt
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError

from src.core.app_config import AUTH_DB_PATH
from src.core.concurrency import with_sqlite_retry
from src.db.migrations import migrate_auth_database, table_exists
from src.errors import StaleDataException

logger = logging.getLogger(__name__)

_DB_PATH = os.path.abspath(str(AUTH_DB_PATH))

VALID_ROLES = {"admin", "teacher"}

SQLITE_TIMEOUT: float = 15.0

PASSWORD_COMPLEXITY_REGEX = re.compile(
    r"^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&_\-#^()+=\[\]{}|:<>,./~\\])[A-Za-z\d@$!%*?&_\-#^()+=\[\]{}|:<>,./~\\]{8,}$"
)

_ph = PasswordHasher()


def configure_db_path(db_path: str | os.PathLike) -> None:
    """Configure the SQLite database path used by the authentication module."""
    global _DB_PATH
    _DB_PATH = os.path.abspath(os.fspath(db_path))


def _connect() -> sqlite3.Connection:
    """Establish a connection to the SQLite database with configured timeout."""
    return sqlite3.connect(_DB_PATH, timeout=SQLITE_TIMEOUT, check_same_thread=False)


def log_security_event(
    event_type: str,
    username: str,
    details: str | None = None,
) -> None:
    """Record a security-relevant event in the security_audit_log table."""
    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO security_audit_log (event_type, username, timestamp, details)
                VALUES (?, ?, ?, ?)
                """,
                (event_type, username, timestamp, details),
            )
            conn.commit()
    except Exception as exc:
        logger.warning(
            "Failed to write security audit log entry [%s, %s]: %s",
            event_type,
            username,
            exc,
        )


def get_security_audit_logs(
    username: str | None = None,
    event_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Retrieve security audit log entries with limit, offset, and optional filters (username, event_type, start_date, end_date)."""
    if limit < 0 or offset < 0:
        raise ValueError("Limit and offset must be non-negative integers.")

    query = (
        "SELECT id, event_type, username, timestamp, details FROM security_audit_log"
    )
    params: list = []
    conditions: list[str] = []

    if username:
        conditions.append("username = ?")
        params.append(username.lower())
    if event_type:
        conditions.append("event_type = ?")
        params.append(event_type)
    if start_date:
        conditions.append("timestamp >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("timestamp <= ?")
        params.append(end_date)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    try:
        with _connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                {
                    "id": r[0],
                    "event_type": r[1],
                    "username": r[2],
                    "timestamp": r[3],
                    "details": r[4],
                }
                for r in rows
            ]
    except sqlite3.Error as e:
        logger.error(f"Failed to query security audit logs: {e}")
        return []


def get_security_audit_log_count(
    username: str | None = None,
    event_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> int:
    """Return total number of matching security audit log entries."""
    query = "SELECT COUNT(*) FROM security_audit_log"
    params: list = []
    conditions: list[str] = []

    if username:
        conditions.append("username = ?")
        params.append(username.lower())
    if event_type:
        conditions.append("event_type = ?")
        params.append(event_type)
    if start_date:
        conditions.append("timestamp >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("timestamp <= ?")
        params.append(end_date)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    try:
        with _connect() as conn:
            row = conn.execute(query, params).fetchone()
            return row[0] if row else 0
    except sqlite3.Error as e:
        logger.error(f"Failed to count security audit logs: {e}")
        return 0


def get_distinct_audit_event_types() -> list[str]:
    """Return a list of all distinct event_type values from security_audit_log."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT event_type FROM security_audit_log ORDER BY event_type"
            ).fetchall()
            return [r[0] for r in rows if r[0]]
    except sqlite3.Error:
        return []


def get_recent_audit_events(limit: int = 20) -> list[dict]:
    """Fetch the N most recent security audit events across all accounts."""
    if limit < 0:
        raise ValueError("Limit must be a non-negative integer.")

    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM security_audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Failed to query recent security audit events: {e}")
        return []


def _hash_password(password: str) -> str:
    """Return an Argon2 hash for the given password."""
    return _ph.hash(password)


def set_password_change_required(username: str, required: bool) -> None:
    """Set or clear the must_change_password flag for a user account.

    When *required* is True the user will be forced to change their password
    on their next successful login.
    """
    username = _validate_username(username)
    with _connect() as conn:
        result = conn.execute(
            "UPDATE users SET must_change_password = ? WHERE username = ?",
            (1 if required else 0, username),
        )
        conn.commit()
    if result.rowcount == 0:
        raise ValueError(f"User '{username}' not found.")


def _verify_password_hash(password: str, stored_hash: str) -> bool:
    """Return True if password matches stored Argon2 or bcrypt hash."""
    if not stored_hash:
        return False
    if stored_hash.startswith("$argon2"):
        try:
            _ph.verify(stored_hash, password)
            return True
        except (VerifyMismatchError, VerificationError):
            return False
    elif stored_hash.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        except Exception:
            return False
    return False


def _validate_username(username: str) -> str:
    username = str(username).strip().lower()
    if not username:
        raise ValueError("Username cannot be empty.")
    return username


def _validate_password(password: str) -> str:
    """Basic validation for authentication checks."""
    password = str(password)
    if not password:
        raise ValueError("Password cannot be empty.")
    return password


def _validate_password_complexity(password: str) -> str:
    """Enforce strong password policy for user creation and password updates."""
    password = str(password)
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain at least one uppercase letter.")
    if not re.search(r"\d", password):
        raise ValueError("Password must contain at least one number.")
    if not re.search(r"[@$!%*?&_\-#^()+=\[\]{}|:<>,./~\\]", password):
        raise ValueError(
            "Password must contain at least one special character (e.g. @$!%*?&)."
        )
    return password


def _validate_role(role: str) -> str:
    role = str(role).strip().lower()
    if role not in VALID_ROLES:
        raise ValueError(f"Role must be one of: {', '.join(sorted(VALID_ROLES))}")
    return role


@with_sqlite_retry
def _record_login_timestamp(username: str) -> None:
    """Update last_login_at timestamp for a given user."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE username = ?",
            (now_str, username),
        )
        conn.commit()


def init_db() -> None:
    """Create or upgrade users.db and seed default administrator accounts."""
    try:
        with _connect() as conn:
            migrate_auth_database(conn)

            row = conn.execute(
                "SELECT COUNT(1) FROM users WHERE username = ?",
                ("admin",),
            ).fetchone()
            exists = bool(row and row[0])

            if not exists:
                hashed = _hash_password("Admin123!")
                conn.execute(
                    """
                    INSERT INTO users (username, password, role)
                    VALUES (?, ?, ?)
                    """,
                    ("admin", hashed, "admin"),
                )
                conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to initialize authentication database: {e}") from e

    try:
        os.chmod(_DB_PATH, 0o600)
    except OSError:
        pass


def verify_user(
    username: str,
    password: str,
    return_details: bool = False,
) -> bool | dict:
    """Authenticate a user and return auth status.

    If return_details is True, returns a dict
    ``{"authenticated": bool, "must_change_password": bool}``.
    Otherwise returns a boolean (True on success, False on failure).
    """
    try:
        username = _validate_username(username)
        password = _validate_password(password)
    except ValueError:
        if return_details:
            return {"authenticated": False, "must_change_password": False}
        return False

    with _connect() as conn:
        row = conn.execute(
            "SELECT password, is_active, must_change_password FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if not row:
        if return_details:
            return {"authenticated": False, "must_change_password": False}
        return False

    stored_hash, is_active, must_change_password = row
    if not is_active:
        if return_details:
            return {"authenticated": False, "must_change_password": False}
        return False

    authenticated = False
    if stored_hash.startswith("$argon2"):
        try:
            _ph.verify(stored_hash, password)
            if _ph.check_needs_rehash(stored_hash):
                hashed = _hash_password(password)
                with _connect() as conn_rehash:
                    conn_rehash.execute(
                        "UPDATE users SET password = ? WHERE username = ?",
                        (hashed, username),
                    )
                    conn_rehash.commit()
            _record_login_timestamp(username)
            authenticated = True
        except (VerifyMismatchError, VerificationError):
            authenticated = False

    elif stored_hash.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
                hashed = _hash_password(password)
                with _connect() as conn_migrate:
                    conn_migrate.execute(
                        "UPDATE users SET password = ? WHERE username = ?",
                        (hashed, username),
                    )
                    conn_migrate.commit()
                _record_login_timestamp(username)
                authenticated = True
        except ValueError:
            authenticated = False

    if return_details:
        return {
            "authenticated": authenticated,
            "must_change_password": bool(must_change_password) if authenticated else False,
        }
    return authenticated


authenticate_user = verify_user


def get_user_role(username: str) -> str | None:
    """Return the role of a user, or None if not found."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            row = conn.execute(
                "SELECT role FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            return row[0] if row else None
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to retrieve user role: {e}") from e


def get_user_roles(user_ids: list[int]) -> dict[int, str]:
    """Return a mapping of user_id -> role for the given user IDs."""
    if not user_ids:
        return {}
    try:
        placeholders = ",".join("?" for _ in user_ids)
        with _connect() as conn:
            rows = conn.execute(
                f"SELECT id, role FROM users WHERE id IN ({placeholders})",
                user_ids,
            ).fetchall()
            return {row[0]: row[1] for row in rows}
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to batch query user roles: {e}") from e


@with_sqlite_retry
def add_user(username: str, password: str, role: str = "teacher") -> None:
    """Insert a user and preserve SQLite duplicate-user semantics."""
    try:
        username = _validate_username(username)
        password = _validate_password(password)
        role = _validate_role(role)
        hashed = _hash_password(password)
        now_str = dt.now(timezone.utc).isoformat()
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, hashed, role),
            )
            conn.execute(
                """
                INSERT INTO password_history (username, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (username, hashed, now_str),
            )
            conn.commit()
    except sqlite3.IntegrityError as e:
        raise ValueError(f"Username '{username}' already exists.") from e
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to add user: {e}") from e


def get_all_users(role: str | None = None) -> list:
    """Return all users as a list of dicts (excludes password hashes).

    Args:
        role: If provided, only return users with this role
            (e.g. "admin" or "teacher").

    Returns:
        List of user dicts, optionally filtered by role.
    """
    try:
        query = "SELECT id, username, role, is_active, version FROM users"
        params: list = []
        if role is not None:
            query += " WHERE role = ?"
            params.append(role)
        query += " ORDER BY id"
        with _connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                {
                    "id": r[0],
                    "username": r[1],
                    "role": r[2],
                    "is_active": bool(r[3]),
                    "version": r[4],
                }
                for r in rows
            ]
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to retrieve users: {e}") from e


@with_sqlite_retry
def delete_user(username: str) -> None:
    """Delete a user and their associated authorization records by username."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            conn.execute("DELETE FROM users WHERE username = ?", (username,))
            conn.execute(
                "DELETE FROM security_audit_log WHERE username = ?", (username,)
            )
            conn.execute("DELETE FROM password_history WHERE username = ?", (username,))

            for table_name in ("user_sessions", "authorization_tokens"):
                if table_exists(conn, table_name):
                    conn.execute(
                        f"DELETE FROM {table_name} WHERE username = ?",
                        (username,),
                    )

            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to delete user: {e}") from e


@with_sqlite_retry
def update_password(
    username: str, new_password: str, current_user: str | None = None
) -> None:
    """Update a user's password with a new Argon2 hash."""
    if current_user and current_user != username:
        if get_user_role(current_user) != "admin":
            raise PermissionError(
                "Unauthorized password modifications for foreign user_ids"
            )

    try:
        username = _validate_username(username)
        new_password = _validate_password(new_password)

        with _connect() as conn:
            cursor = conn.execute(
                "SELECT password FROM users WHERE username = ?",
                (username,),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError("User not found.")
            current_hash = row[0]

            history_rows = conn.execute(
                """
                SELECT password_hash FROM password_history
                WHERE username = ?
                ORDER BY id DESC LIMIT 3
                """,
                (username,),
            ).fetchall()
            recent_hashes = [r[0] for r in history_rows]

            if current_hash and current_hash not in recent_hashes:
                recent_hashes.append(current_hash)

            recent_hashes = recent_hashes[:3]

            for old_hash in recent_hashes:
                if _verify_password_hash(new_password, old_hash):
                    raise ValueError(
                        "New password cannot be one of your last 3 passwords"
                    )

            hashed = _hash_password(new_password)
            password_changed_at = dt.now(timezone.utc).isoformat()
            cursor = conn.execute(
                """
                UPDATE users
                SET password = ?,
                    password_changed_at = ?
                WHERE username = ?
                """,
                (
                    hashed,
                    password_changed_at,
                    username,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("User not found.")

            conn.execute(
                """
                INSERT INTO password_history (username, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (username, hashed, password_changed_at),
            )
            conn.commit()

        log_security_event(
            event_type="password_change",
            username=username,
            details="Password updated successfully.",
        )
    except (ValueError, PermissionError):
        raise
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to update password: {e}") from e


def get_tour_completed(username: str) -> bool:
    """Return whether a user has completed the onboarding tour."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            row = conn.execute(
                "SELECT tour_completed FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            return bool(row[0]) if row else False
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to retrieve tour status: {e}") from e


@with_sqlite_retry
def set_tour_completed(username: str, completed: bool = True) -> None:
    """Mark a user as having completed the onboarding tour."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            conn.execute(
                "UPDATE users SET tour_completed = ? WHERE username = ?",
                (1 if completed else 0, username),
            )
            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to update tour status: {e}") from e


def get_2fa_status(username: str) -> tuple[bool, str | None]:
    """Return (two_factor_enabled, otp_secret) for a user."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT two_factor_enabled, otp_secret FROM users WHERE username = ?",
            (username.lower(),),
        ).fetchone()
    if not row:
        return False, None
    return bool(row[0]), row[1]


@with_sqlite_retry
def enable_2fa(username: str, secret: str) -> None:
    """Enable 2FA for a user and store their OTP secret."""
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET two_factor_enabled = 1, otp_secret = ? WHERE username = ?",
            (secret, username.lower()),
        )
        conn.commit()


@with_sqlite_retry
def disable_2fa(username: str) -> None:
    """Disable 2FA for a user and clear their OTP secret."""
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET two_factor_enabled = 0, otp_secret = NULL WHERE username = ?",
            (username.lower(),),
        )
        conn.commit()


def check_login_rate_limit(username: str) -> tuple[bool, str | None]:
    """Check if username is rate limited. Returns (is_allowed, error_message)."""
    from src.utils.redis_cache import get_login_attempts, is_login_locked_out

    identifier = username.lower()
    if is_login_locked_out(identifier):
        attempts = get_login_attempts(identifier)
        return (
            False,
            f"Account locked due to too many failed attempts. Please try again in 15 minutes. ({attempts}/5 attempts)",
        )
    return True, None


@with_sqlite_retry
def record_failed_login(username: str) -> None:
    """Record a failed login attempt for rate limiting."""
    from src.utils.redis_cache import increment_login_attempts

    increment_login_attempts(username.lower())


@with_sqlite_retry
def clear_login_attempts(username: str) -> None:
    """Clear failed login attempts after successful login."""
    from src.utils.redis_cache import clear_login_attempts as redis_clear_login_attempts

    redis_clear_login_attempts(username.lower())


def get_user_preferences(username: str) -> dict:
    """Return user preferences as a dictionary, or empty dict if none exist."""
    username = username.lower()
    with _connect() as conn:
        row = conn.execute(
            "SELECT preferences FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return {}
    return {}


@with_sqlite_retry
def update_user_preferences(username: str, preferences: dict) -> None:
    """Serialize and update user preferences in the database."""
    username = username.lower()
    prefs_str = json.dumps(preferences)
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET preferences = ? WHERE username = ?",
            (prefs_str, username),
        )
        conn.commit()


def get_notification_preferences(username: str) -> dict:
    """Return user notification preferences dict with defaults."""
    username = _validate_username(username)
    prefs = get_user_preferences(username)
    email_val = prefs.get("email_notifications", True)
    webhook_val = prefs.get("webhook_notifications", True)

    if not isinstance(email_val, bool):
        email_val = True
    if not isinstance(webhook_val, bool):
        webhook_val = True

    return {
        "email_notifications": email_val,
        "webhook_notifications": webhook_val,
    }


@with_sqlite_retry
def update_notification_preferences(
    username: str,
    email_notifications: bool = True,
    webhook_notifications: bool = True,
) -> dict:
    """Update notification preferences for a user."""
    if not isinstance(email_notifications, bool):
        raise TypeError("email_notifications must be a boolean")
    if not isinstance(webhook_notifications, bool):
        raise TypeError("webhook_notifications must be a boolean")

    username = _validate_username(username)
    prefs = get_user_preferences(username)
    prefs["email_notifications"] = email_notifications
    prefs["webhook_notifications"] = webhook_notifications
    update_user_preferences(username, prefs)
    return prefs


def get_user_theme(username: str) -> str:
    """Return the user's theme preference (default 'light')."""
    username = username.lower()
    with _connect() as conn:
        row = conn.execute(
            "SELECT theme FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return row[0] if row else "light"


@with_sqlite_retry
def set_user_theme(username: str, theme: str) -> None:
    """Update the user's theme preference."""
    username = username.lower()
    if theme not in ("light", "dark"):
        theme = "light"
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET theme = ? WHERE username = ?",
            (theme, username),
        )
        conn.commit()


@with_sqlite_retry
def get_or_create_sso_user(email: str, default_role: str = "teacher") -> str:
    """Finds a user by email (as username) or creates a new one for SSO."""
    username = _validate_username(email)
    with _connect() as conn:
        row = conn.execute(
            "SELECT role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row:
            return row[0]
        hashed = _hash_password("!")
        role = _validate_role(default_role)
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hashed, role),
        )
        conn.commit()
        return role


def get_user_active_status(username: str) -> bool:
    """Return whether a user account is active."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            row = conn.execute(
                "SELECT is_active FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            return bool(row[0]) if row else False
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to retrieve user active status: {e}") from e


@with_sqlite_retry
def set_user_active_status(username: str, is_active: bool) -> None:
    """Set whether a user account is active (suspended or active)."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            if username == "admin" and not is_active:
                raise ValueError("The admin account cannot be suspended.")
            conn.execute(
                "UPDATE users SET is_active = ? WHERE username = ?",
                (1 if is_active else 0, username),
            )
            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to update user active status: {e}") from e


@with_sqlite_retry
def update_user_profile(
    username: str,
    role: str,
    is_active: bool,
    expected_version: int,
) -> None:
    """Update a user's role and active status with optimistic locking.

    Args:
        username: The user to update.
        role: The new role.
        is_active: Active status.
        expected_version: Expected database version.

    Raises:
        StaleDataException: If database version != expected_version.
        ValueError: If user not found or suspension check fails.
    """
    username = _validate_username(username)
    role = _validate_role(role)
    is_active_val = 1 if is_active else 0

    if username == "admin" and not is_active:
        raise ValueError("The admin account cannot be suspended.")

    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT version FROM users WHERE username = ?",
                (username,),
            ).fetchone()

            if not row:
                raise ValueError("User not found.")

            current_version = row[0]
            if current_version != expected_version:
                raise StaleDataException(
                    f"Conflict detected: User profile updated by another process. "
                    f"Expected version {expected_version}, but database has version {current_version}."
                )

            cursor = conn.execute(
                """
                UPDATE users
                SET role = ?,
                    is_active = ?,
                    version = version + 1
                WHERE username = ? AND version = ?
                """,
                (role, is_active_val, username, expected_version),
            )
            if cursor.rowcount == 0:
                raise StaleDataException(
                    "Conflict detected: User profile was updated concurrently."
                )
            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to update user profile: {e}") from e


def is_user_active(username: str) -> bool:
    """Return True if username exists and is_active is 1, or if username does not exist yet."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            row = conn.execute(
                "SELECT is_active FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            return bool(row[0]) if row else True
    except sqlite3.Error:
        return True


def get_user_count() -> int:
    """Returns the total number of registered users in the system."""
    with _connect() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM users")
        row = cursor.fetchone()
        return row[0] if row else 0


def get_active_users_count() -> int:
    """Return the total number of active users in the database."""
    with _connect() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
        row = cursor.fetchone()
        return row[0] if row else 0


def format_user_created_date(iso_str: str) -> str:
    """Format an ISO date string as a human-readable date (e.g. "Jul 28, 2026")."""
    if not iso_str or not isinstance(iso_str, str):
        return "Unknown"

    iso_str = iso_str.strip()
    if not iso_str:
        return "Unknown"

    try:
        from dateutil import parser as dateutil_parser

        dt_obj = dateutil_parser.parse(iso_str)
        return dt_obj.strftime("%b %d, %Y")
    except Exception:
        pass

    cleaned = iso_str.rstrip("Z")
    for parser_fn in (
        dt.fromisoformat,
        lambda s: dt.strptime(s, "%Y-%m-%d"),
        lambda s: dt.strptime(s, "%Y-%m-%d %H:%M:%S"),
        lambda s: dt.strptime(s, "%Y-%m-%dT%H:%M:%S"),
    ):
        try:
            dt_obj = parser_fn(cleaned)
            return dt_obj.strftime("%b %d, %Y")
        except Exception:
            continue

    return "Unknown"


def _get_token_signature(token: str) -> str:
    """Return a SHA-256 hex digest signature for a token."""
    import hashlib

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@with_sqlite_retry
def revoke_token(token: str, details: str | None = None) -> None:
    """Revoke an active Bearer token by storing its signature in revoked_tokens table."""
    if not token or not isinstance(token, str):
        raise ValueError("Token must be a non-empty string.")

    token = token.strip()
    if not token:
        raise ValueError("Token cannot be empty.")

    signature = _get_token_signature(token)
    revoked_at = datetime.datetime.now(timezone.utc).isoformat()

    try:
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS revoked_tokens (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_signature TEXT UNIQUE NOT NULL,
                    revoked_at TEXT NOT NULL,
                    details    TEXT DEFAULT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO revoked_tokens (token_signature, revoked_at, details)
                VALUES (?, ?, ?)
                """,
                (signature, revoked_at, details),
            )
            if signature != token:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO revoked_tokens (token_signature, revoked_at, details)
                    VALUES (?, ?, ?)
                    """,
                    (token, revoked_at, details),
                )
            conn.commit()
            log_security_event(
                event_type="token_revocation",
                username="system",
                details=details or f"Token signature {signature[:12]}... revoked",
            )
    except sqlite3.Error as e:
        logger.error(f"Failed to revoke token: {e}")
        raise sqlite3.Error(f"Failed to revoke token: {e}") from e


def is_token_revoked(token: str) -> bool:
    """Return True if the token or its SHA-256 signature exists in revoked_tokens."""
    if not token or not isinstance(token, str):
        return False

    token = token.strip()
    if not token:
        return False

    signature = _get_token_signature(token)

    try:
        with _connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='revoked_tokens'"
            )
            if not cursor.fetchone():
                return False

            row = conn.execute(
                "SELECT 1 FROM revoked_tokens WHERE token_signature = ? OR token_signature = ? LIMIT 1",
                (signature, token),
            ).fetchone()
            return bool(row)
    except sqlite3.Error as e:
        logger.error(f"Failed to check token revocation status: {e}")
        return False


def get_upload_count(username: str | None = None) -> int:
    """Return total number of uploads for a user or system-wide."""
    try:
        with _connect() as conn:
            if username:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM security_audit_log WHERE username = ? AND event_type = 'file_upload'",
                    (username.lower(),),
                )
            else:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM security_audit_log WHERE event_type = 'file_upload'"
                )
            row = cursor.fetchone()
            return row[0] if row else 0
    except sqlite3.Error:
        return 0
