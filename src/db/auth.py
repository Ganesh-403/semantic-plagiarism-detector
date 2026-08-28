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
src/db/auth.py
--------------
SQLite-backed authentication with Argon2 password hashing (via argon2-cffi),
automatic transparent migration from legacy bcrypt hashes, user login tracking,
and strong password complexity policies.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import string
import time
from contextlib import contextmanager
from datetime import datetime as dt
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import bcrypt
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError

try:
    import zxcvbn
except ImportError:
    zxcvbn = None

from src.core.app_config import AUTH_DB_PATH
from src.db.base import BaseRepository
from src.db.common import with_sqlite_retry
from src.db.connection import get_connection
from src.db.migrations import migrate_auth_database, table_exists
from src.db.security_audit import count_recent_failed_logins
from src.exceptions import StaleDataException

logger = logging.getLogger(__name__)

from src.core.app_config import get_valid_roles

_DB_PATH = os.path.abspath(str(AUTH_DB_PATH))


def get_auth_db_path() -> Path:
    return Path(_DB_PATH)


VALID_ROLES = get_valid_roles()

SQLITE_TIMEOUT: float = 5.0
"""float: Busy timeout in seconds (15.0s) for SQLite database connections in the authentication module.

Architecture & High-Concurrency System Rationale:
-------------------------------------------------
This high timeout (15.0 seconds) is intentionally configured to prevent lock contention failures in `users.db`
when background plagiarism detection tasks, vector database syncs, and multi-user authentication requests execute concurrently.

Although SQLite WAL (Write-Ahead Logging) mode allows concurrent readers alongside one writer, writing operations
(such as transparent password re-hashing, audit log insertion, failed attempt tracking, and user profile updates)
must acquire an exclusive write lock.

Specific Scenarios Requiring a 15.0-Second Busy Timeout in `auth.py`:
----------------------------------------------------------------------
1. **Concurrent User Logins & Transparent Bcrypt-to-Argon2 Re-hashing:**
   During peak user activity or automated batch tests, multiple authentication threads attempt to update user records
   simultaneously when migrating legacy passwords to Argon2id.

2. **Security Audit Log Persistence & Login Rate-Limiting:**
   Every authentication attempt, failed login, or password update writes security audit events to `users.db`.
   High-frequency parallel authentication requests contend for write locks on the audit log table.

3. **Background WAL Checkpointing Sweeps:**
   SQLite automatically flushes write-ahead log pages (`users.db-wal`) back to the main `users.db` database.
   Checkpointing holds temporary exclusive write locks.

⚠️ WARNING FOR DEVELOPERS:
------------------------
Do NOT reduce `SQLITE_TIMEOUT` below 15.0 seconds. Lowering this value risks raising spurious
`sqlite3.OperationalError: database is locked` exceptions under concurrent workloads.
"""

PASSWORD_COMPLEXITY_REGEX = re.compile(
    r"^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&_\-#^()+=\[\]{}|:<>,./~\\])[A-Za-z\d@$!%*?&_\-#^()+=\[\]{}|:<>,./~\\]{8,128}$"
)

_ph = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
)

# Configuration for account lockout (Issue #2704)
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_WINDOW_MINUTES = 15


def is_account_locked(
    username: str,
    max_attempts: int = MAX_FAILED_ATTEMPTS,
    window_minutes: int = LOCKOUT_WINDOW_MINUTES,
) -> bool:
    """Check if an account is temporarily locked due to too many failed login attempts.

    Args:
        username: The username to check.
        max_attempts: Maximum allowed failed attempts before lockout.
        window_minutes: Time window in minutes for counting attempts.

    Returns:
        True if the account is locked, False otherwise.
    """
    if not username:
        return False

    failed_count = count_recent_failed_logins(username, window_minutes=window_minutes)

    is_locked = failed_count >= max_attempts

    if is_locked:
        logger.warning(
            "Account lockout triggered for %s: %d failed attempts in last %d minutes.",
            username,
            failed_count,
            window_minutes,
        )

    return is_locked


class AuthRepository(BaseRepository):
    """Data access repository for authentication, user management, and security audit logs."""

    def __init__(self, db_path: str | os.PathLike = AUTH_DB_PATH) -> None:
        super().__init__(db_path)
        self._cached_event_types: list[str] | None = None
        self._cached_event_types_timestamp: float = 0.0
        self._event_types_cache_ttl: float = 60.0

    @property
    def db_path(self) -> Path:
        return Path(_DB_PATH)

    @contextmanager
    def connection(
        self, read_only: bool = False
    ) -> Generator[sqlite3.Connection, None, None]:
        with get_connection(Path(_DB_PATH), read_only=read_only) as conn:
            yield conn

    def clear_distinct_event_types_cache(self) -> None:
        """Clear the cached distinct audit event types."""
        self._cached_event_types = None
        self._cached_event_types_timestamp = 0.0

    def init_db(self) -> None:
        """Create or upgrade users.db and seed default administrator accounts."""
        init_db()

    def log_security_event(
        self,
        event_type: str,
        username: str,
        details: str | None = None,
    ) -> None:
        """Record a security-relevant event in the security_audit_log table."""
        timestamp = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            with self.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO security_audit_log (event_type, username, timestamp, details)
                    VALUES (?, ?, ?, ?)
                    """,
                    (event_type, username, timestamp, details),
                )
                conn.commit()
            if (
                self._cached_event_types is not None
                and event_type
                and event_type not in self._cached_event_types
            ):
                self._cached_event_types.append(event_type)
                self._cached_event_types.sort()
        except Exception as exc:
            logger.warning(
                "Failed to write security audit log entry [%s, %s]: %s",
                event_type,
                username,
                exc,
            )
            try:
                from src.db.security_audit import _emit_audit_log_failure_alert

                _emit_audit_log_failure_alert(
                    event_type=event_type, username=username, error=exc
                )
            except Exception as alert_exc:
                logger.warning(
                    "Failed to trigger audit log failure alert: %s", alert_exc
                )

    def _build_audit_log_query_conditions(
        self,
        username: str | None = None,
        event_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[str, tuple]:
        """Build WHERE clause snippet (if any) and parameters tuple for security audit log queries."""
        conditions: list[str] = []
        params: list = []

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

        where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        return where_clause, tuple(params)

    def get_security_audit_logs(
        self,
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

        where_clause, params = self._build_audit_log_query_conditions(
            username=username,
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
        )
        query = (
            f"SELECT id, event_type, username, timestamp, details FROM security_audit_log{where_clause}"  # nosec
            " ORDER BY id DESC LIMIT ? OFFSET ?"
        )
        query_params = params + (limit, offset)

        try:
            with self.connection(read_only=True) as conn:
                rows = conn.execute(query, query_params).fetchall()
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
        self,
        username: str | None = None,
        event_type: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        """Return total number of matching security audit log entries."""
        where_clause, params = self._build_audit_log_query_conditions(
            username=username,
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
        )
        query = f"SELECT COUNT(*) FROM security_audit_log{where_clause}"  # nosec

        try:
            with self.connection(read_only=True) as conn:
                row = conn.execute(query, params).fetchone()
                return row[0] if row else 0
        except sqlite3.Error as e:
            logger.error(f"Failed to count security audit logs: {e}")
            raise

    def get_distinct_audit_event_types(self, force_refresh: bool = False) -> list[str]:
        """Return a list of all distinct event_type values from security_audit_log with TTL caching (Issue #2687)."""
        now = time.monotonic()
        if (
            not force_refresh
            and self._cached_event_types is not None
            and (now - self._cached_event_types_timestamp) < self._event_types_cache_ttl
        ):
            return list(self._cached_event_types)

        try:
            with self.connection(read_only=True) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT event_type FROM security_audit_log ORDER BY event_type"
                ).fetchall()
                event_types = [r[0] for r in rows if r[0]]
                self._cached_event_types = event_types
                self._cached_event_types_timestamp = now
                return list(event_types)
        except sqlite3.Error:
            return []

    def get_recent_audit_events(self, limit: int = 20) -> list[dict]:
        """Fetch the N most recent security audit events across all accounts."""
        if limit < 0:
            raise ValueError("Limit must be a non-negative integer.")

        try:
            with self.connection(read_only=True) as conn:
                rows = conn.execute(
                    "SELECT * FROM security_audit_log ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to query recent security audit events: {e}")
            return []


auth_repo = AuthRepository(_DB_PATH)


def configure_db_path(db_path: str | os.PathLike) -> None:
    """Configure the SQLite database path used by the authentication module."""
    global _DB_PATH
    _DB_PATH = os.path.abspath(os.fspath(db_path))
    auth_repo.configure_db_path(_DB_PATH)


from contextlib import contextmanager


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    """Establish a connection to the SQLite database with configured timeout and close on exit."""
    conn = sqlite3.connect(_DB_PATH, timeout=SQLITE_TIMEOUT, check_same_thread=False)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def log_security_event(
    event_type: str,
    username: str,
    details: str | None = None,
) -> None:
    """Record a security-relevant event in the security_audit_log table."""
    auth_repo.log_security_event(event_type, username, details)


def get_security_audit_logs(
    username: str | None = None,
    event_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Retrieve security audit log entries with limit, offset, and optional filters (username, event_type, start_date, end_date)."""
    return auth_repo.get_security_audit_logs(
        username=username,
        event_type=event_type,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )


def get_security_audit_log_count(
    username: str | None = None,
    event_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> int:
    """Return total number of matching security audit log entries."""
    return auth_repo.get_security_audit_log_count(
        username=username,
        event_type=event_type,
        start_date=start_date,
        end_date=end_date,
    )


def get_distinct_audit_event_types(force_refresh: bool = False) -> list[str]:
    """Return a list of all distinct event_type values from security_audit_log with TTL caching (Issue #2687)."""
    return auth_repo.get_distinct_audit_event_types(force_refresh=force_refresh)


def clear_distinct_audit_event_types_cache() -> None:
    """Clear the cached distinct audit event types (Issue #2687)."""
    auth_repo.clear_distinct_event_types_cache()


def get_recent_audit_events(limit: int = 20) -> list[dict]:
    """Fetch the N most recent security audit events across all accounts."""
    return auth_repo.get_recent_audit_events(limit=limit)


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


def _validate_username(username: Any) -> str:
    """Validate and sanitize a username string.

    Raises:
        ValueError: If username is None, not a string, or empty/whitespace.
    """
    if username is None or not isinstance(username, str):
        raise ValueError("Username cannot be empty.")
    normalized = username.strip().lower()
    if not normalized:
        raise ValueError("Username cannot be empty.")
    return normalized


def _validate_password(password: str) -> str:
    """Basic validation for authentication checks."""
    password = str(password)
    if not password:
        raise ValueError("Password cannot be empty.")
    if len(password) > 128:
        raise ValueError("Password cannot exceed 128 characters.")
    return password


def _validate_password_complexity(password: str) -> str:
    """Enforce strong password policy for user creation and password updates."""
    password = str(password)
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if len(password) > 128:
        raise ValueError("Password cannot exceed 128 characters.")
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain at least one uppercase letter.")
    if not re.search(r"\d", password):
        raise ValueError("Password must contain at least one number.")
    if not re.search(r"[@$!%*?&_\-#^()+=\[\]{}|:<>,./~\\]", password):
        raise ValueError(
            "Password must contain at least one special character (e.g. @$!%*?&)."
        )
    if zxcvbn is not None:
        result = zxcvbn.zxcvbn(password)
        if result.get("score", 0) < 3:
            feedback = result.get("feedback", {})
            warning = feedback.get("warning")
            if warning:
                raise ValueError(f"Password is too weak or common: {warning}")
            raise ValueError(
                "Password is too weak or commonly used. Please choose a stronger password."
            )
    return password


validate_password_complexity = _validate_password_complexity


def _validate_role(role: str) -> str:
    role = str(role).strip().lower()
    valid_roles = get_valid_roles()
    if role not in valid_roles:
        raise ValueError(f"Role must be one of: {', '.join(sorted(valid_roles))}")
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
                hashed = str(_hash_password("Admin123!"))
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
    ``{"authenticated": bool, "must_change_password": bool, "password_expired": bool}``.
    Otherwise returns a boolean (True on success, False on failure).

    Implements account lockout protection (Issue #2704) by checking for
    recent failed login attempts before verifying the password hash.
    """
    try:
        username = _validate_username(username)
        password = _validate_password(password)
    except ValueError:
        if return_details:
            return {"authenticated": False, "must_change_password": False}
        return False

    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT password, status, must_change_password FROM users WHERE username = ?",
                (username,),
            ).fetchone()

        if not row:
            if return_details:
                return {"authenticated": False, "must_change_password": False}
            return False

        stored_hash, status, must_change_password = row
        if status == "suspended":
            if return_details:
                return {"authenticated": False, "must_change_password": False}
            return False

        # Issue #2704: Check for account lockout before doing expensive password hashing
        if is_account_locked(username):
            log_security_event(
                event_type="login_blocked_lockout",
                username=username,
                details=f"Login attempt blocked due to lockout ({MAX_FAILED_ATTEMPTS} failures in {LOCKOUT_WINDOW_MINUTES}m)",
            )
            if return_details:
                return {"authenticated": False, "must_change_password": False}
            return False

        authenticated = False
        if stored_hash and stored_hash.startswith("$argon2"):
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

        elif stored_hash and stored_hash.startswith(("$2a$", "$2b$", "$2y$")):
            try:
                if bcrypt.checkpw(
                    password.encode("utf-8"), stored_hash.encode("utf-8")
                ):
                    hashed = _hash_password(password)
                    with _connect() as conn_migrate:
                        conn_migrate.execute(
                            "UPDATE users SET password = ? WHERE username = ?",
                            (hashed, username),
                        )
                        conn_migrate.commit()
                    _record_login_timestamp(username)
                    authenticated = True
            except Exception:
                authenticated = False

        # Check password expiration after successful authentication (Issue #2716)
        password_expired = False
        if authenticated:
            password_expired = is_password_expired(username)
            if password_expired:
                log_security_event(
                    event_type="login_success_password_expired",
                    username=username,
                    details="Successful login but password requires rotation",
                )
            else:
                log_security_event(
                    event_type="login_success",
                    username=username,
                    details="Successful authentication",
                )

        if return_details:
            return {
                "authenticated": authenticated,
                "must_change_password": (
                    bool(must_change_password) if authenticated else False
                ),
                "password_expired": password_expired if authenticated else False,
            }
        return authenticated
    except sqlite3.Error as e:
        logger.error(f"Failed to verify user: {e}")
        if return_details:
            return {"authenticated": False, "must_change_password": False}
        return False


authenticate_user = verify_user


def get_user_role(username: str) -> str:
    """
    Return the role of a user, or 'user' as default if not found.

    Args:
        username: The username to look up

    Returns:
        str: The user's role (admin, teacher, or user)
    """
    try:
        username = _validate_username(username)
        with _connect() as conn:
            row = conn.execute(
                "SELECT role FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            # Return the role if found, otherwise default to "user"
            return row[0] if row else "user"
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve user role for {username}: {e}")
        return "user"  # Safe fallback


def get_user_role_safe(username: str, default: str = "user") -> str:
    """
    Safely get user role with a custom default.

    Args:
        username: The username to look up
        default: Default role if user not found (default: "user")

    Returns:
        str: The user's role or the default
    """
    try:
        return get_user_role(username)
    except Exception as e:
        logger.error(f"Error getting role for {username}: {e}")
        return default


def is_admin(username: str) -> bool:
    """
    Check if a user is an admin.

    Args:
        username: The username to check

    Returns:
        bool: True if user is admin, False otherwise
    """
    return get_user_role(username) == "admin"


def is_teacher(username: str) -> bool:
    """
    Check if a user is a teacher.

    Args:
        username: The username to check

    Returns:
        bool: True if user is teacher, False otherwise
    """
    role = get_user_role(username)
    return role == "teacher" or role == "admin"


def get_user_last_login(username: str) -> str | None:
    """Return the last_login_at timestamp for a user, or None if not found/never logged in."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            row = conn.execute(
                "SELECT last_login_at FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            return row[0] if row else None
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to retrieve user last login: {e}") from e


def get_user_roles(user_ids: list[int]) -> dict[int, str]:
    """Return a mapping of user_id -> role for the given user IDs."""
    if not user_ids:
        return {}
    try:
        placeholders = ",".join("?" for _ in user_ids)
        with _connect() as conn:
            rows = conn.execute(
                f"SELECT id, role FROM users WHERE id IN ({placeholders})",  # nosec
                tuple(user_ids),
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
        query = "SELECT id, username, role, status, version FROM users"
        params: tuple = ()
        if role is not None:
            query += " WHERE role = ?"
            params = (role,)
        query += " ORDER BY id"
        with _connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                {
                    "id": r[0],
                    "username": r[1],
                    "role": r[2],
                    "status": r[3],
                    "is_active": (r[3] == "active"),
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
                        f"DELETE FROM {table_name} WHERE username = ?",  # nosec
                        (username,),
                    )

            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to delete user: {e}") from e


@with_sqlite_retry
def update_password(
    username: str, new_password: str, current_user: str | None = None
) -> None:
    """Update a user's password with a new Argon2 hash and record password_changed_at timestamp."""
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
                SET password = ?, password_changed_at = ?
                WHERE username = ?
                """,
                (hashed, password_changed_at, username),
            )
            if cursor.rowcount != 1:
                raise ValueError("User not found.")

            conn.execute(
                """
                INSERT INTO password_history (username, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (username, current_hash, password_changed_at),
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
    except Exception as e:
        logger.error("Failed to update password for user %s: %s", username, e)
        raise


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


def _get_fernet_key() -> bytes:
    """Load or derive a valid 32-byte Fernet key from environment variables."""
    import base64
    import hashlib

    key_str = os.getenv("OTP_ENCRYPTION_KEY") or os.getenv("ENCRYPTION_KEY")
    if not key_str:
        key_str = "default-fallback-otp-encryption-key-do-not-use-in-production"

    hashed = hashlib.sha256(key_str.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(hashed)


def _encrypt_otp_secret(secret: str) -> str:
    """Encrypt the OTP secret using cryptography.fernet."""
    if not secret:
        return secret
    from cryptography.fernet import Fernet

    key = _get_fernet_key()
    f = Fernet(key)
    return f.encrypt(secret.encode("utf-8")).decode("utf-8")


def _decrypt_otp_secret(encrypted_secret: str) -> str:
    """Decrypt the OTP secret using cryptography.fernet, falling back to plaintext on error."""
    if not encrypted_secret:
        return encrypted_secret
    from cryptography.fernet import Fernet, InvalidToken

    key = _get_fernet_key()
    f = Fernet(key)
    try:
        return f.decrypt(encrypted_secret.encode("utf-8")).decode("utf-8")
    except (InvalidToken, Exception):
        return encrypted_secret


def get_2fa_status(username: str) -> tuple[bool, str | None]:
    """Return (two_factor_enabled, otp_secret) for a user."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT two_factor_enabled, otp_secret FROM users WHERE username = ?",
            (username.lower(),),
        ).fetchone()
    if not row:
        return False, None
    decrypted_secret = _decrypt_otp_secret(row[1]) if row[1] is not None else None
    return bool(row[0]), decrypted_secret


@with_sqlite_retry
def enable_2fa(username: str, secret: str) -> None:
    """Enable 2FA for a user and store their OTP secret."""
    encrypted_secret = _encrypt_otp_secret(secret)
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET two_factor_enabled = 1, otp_secret = ? WHERE username = ?",
            (encrypted_secret, username.lower()),
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


# ============================================================================
# PASSWORD EXPIRATION - Issue #2716
# ============================================================================

DEFAULT_PASSWORD_LIFETIME_DAYS = 90


def is_password_expired(username: str) -> bool:
    """Check if a user's password has expired based on password_expires_at.

    Args:
        username: The username to check.

    Returns:
        True if the password is expired, False if still valid or if
        expiration is not configured (NULL).
    """
    if not username:
        return False

    try:
        username = _validate_username(username)
        with _connect() as conn:
            row = conn.execute(
                "SELECT password_expires_at FROM users WHERE username = ?", (username,)
            ).fetchone()

            if not row or not row[0]:
                # No expiration set means password never expires
                return False

            expires_at_str = row[0]
            expires_at = dt.fromisoformat(expires_at_str.replace("Z", "+00:00"))

            # Compare with current UTC time
            now_utc = dt.now(timezone.utc)
            is_expired = now_utc >= expires_at

            if is_expired:
                logger.info(
                    "Password expired for user %s (expired at %s)",
                    username,
                    expires_at_str,
                )

            return is_expired

    except sqlite3.Error as e:
        logger.error("Failed to check password expiration for %s: %s", username, e)
        # Fail open: don't block login if we can't read the expiration date
        return False
    except ValueError as e:
        logger.error(
            "Invalid date format in password_expires_at for %s: %s", username, e
        )
        return False


@with_sqlite_retry
def set_password_expiration(
    username: str, days_until_expiration: int = DEFAULT_PASSWORD_LIFETIME_DAYS
) -> bool:
    """Set or update the password expiration date for a user.

    Args:
        username: The username to update.
        days_until_expiration: Number of days until the password expires.

    Returns:
        True if the update was successful, False otherwise.
    """
    if not username or days_until_expiration < 0:
        return False

    try:
        username = _validate_username(username)
        expiration_date = (
            dt.now(timezone.utc) + timedelta(days=days_until_expiration)
        ).isoformat()

        with _connect() as conn:
            cursor = conn.execute(
                """
                UPDATE users
                SET password_expires_at = ?
                WHERE username = ?
                """,
                (expiration_date, username),
            )
            conn.commit()

            if cursor.rowcount == 0:
                logger.warning("set_password_expiration: User %s not found", username)
                return False

            logger.info(
                "Set password expiration for %s to %s", username, expiration_date
            )
            return True

    except sqlite3.Error as e:
        logger.error("Failed to set password expiration for %s: %s", username, e)
        return False


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
def _generate_secure_password(length: int = 32) -> str:
    """
    Generate a cryptographically secure random password.

    Args:
        length: Length of the password (default: 32 characters)

    Returns:
        A secure random password containing uppercase, lowercase, digits, and symbols
    """
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return "".join(secrets.choice(alphabet) for _ in range(length))


def get_or_create_sso_user(email: str, default_role: str = "teacher") -> str:
    """
    Get or create SSO user with enhanced security.
    Wrapper around get_or_create_sso_user_enhanced for backward compatibility.
    """
    result = get_or_create_sso_user_enhanced(
        email=email,
        provider="unknown",
        provider_user_id=email,
        default_role=default_role,
    )
    return result["role"]


def is_sso_user(username: str) -> bool:
    """Check if a user was created via SSO."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            row = conn.execute(
                "SELECT sso_provider FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            return row is not None and row[0] is not None
    except sqlite3.Error:
        return False


def get_user_active_status(username: str) -> bool:
    """Return whether a user account is active."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            row = conn.execute(
                "SELECT status FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            return (row[0] == "active") if row else False
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to retrieve user active status: {e}") from e


@with_sqlite_retry
def set_user_status(username: str, status: str) -> None:
    """Set a user's account status."""
    try:
        username = _validate_username(username)

        with _connect() as conn:
            if username == "admin" and status != "active":
                raise ValueError("The admin account cannot be suspended.")

            conn.execute(
                """
                UPDATE users
                SET status = ?
                WHERE username = ?
                """,
                (status, username),
            )
            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to update user status: {e}") from e


@with_sqlite_retry
def set_user_active_status(username: str, is_active: bool) -> None:
    """Set whether a user account is active (suspended or active)."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            if username == "admin" and not is_active:
                raise ValueError("The admin account cannot be suspended.")
            conn.execute(
                """
                UPDATE users
                SET status = ?
                WHERE username = ?
                """,
                (
                    "active" if is_active else "suspended",
                    username,
                ),
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
                    status = ?,
                    version = version + 1
                WHERE username = ? AND version = ?
                """,
                (
                    role,
                    "active" if is_active else "suspended",
                    username,
                    expected_version,
                ),
            )
            if cursor.rowcount == 0:
                raise StaleDataException(
                    "Conflict detected: User profile was updated concurrently."
                )
            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to update user profile: {e}") from e


def is_user_active(username: str) -> bool:
    """Return True if username exists and status is 'active', or if username does not exist yet."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            row = conn.execute(
                "SELECT status FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            return (row[0] == "active") if row else True
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
        cursor = conn.execute("SELECT COUNT(1) FROM users WHERE status = 'active'")
        row = cursor.fetchone()
        return int(row[0]) if row else 0


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


# ── Revoked Tokens In-Memory Cache (Issue #3018) ──────────────────────────────
_REVOKED_TOKEN_CACHE_TTL: int = 60
_REVOKED_TOKEN_CACHE_MAXSIZE: int = 10000

try:
    import cachetools

    _revoked_token_cache: cachetools.TTLCache = cachetools.TTLCache(
        maxsize=_REVOKED_TOKEN_CACHE_MAXSIZE, ttl=_REVOKED_TOKEN_CACHE_TTL
    )
except ImportError:

    class _FallbackTTLCache(dict):
        def __init__(self, maxsize: int = 10000, ttl: int = 60):
            super().__init__()
            self._ttl = ttl
            self._times: dict[str, float] = {}

        def __getitem__(self, key: str) -> bool:
            if key in self._times and time.time() - self._times[key] > self._ttl:
                del self[key]
                del self._times[key]
                raise KeyError(key)
            return super().__getitem__(key)

        def __contains__(self, key: object) -> bool:
            k_str = str(key)
            if k_str in self._times and time.time() - self._times[k_str] > self._ttl:
                del self[k_str]
                del self._times[k_str]
                return False
            return super().__contains__(key)

        def __setitem__(self, key: str, value: bool) -> None:
            self._times[key] = time.time()
            super().__setitem__(key, value)

        def clear(self) -> None:
            self._times.clear()
            super().clear()

    _revoked_token_cache = _FallbackTTLCache(
        maxsize=_REVOKED_TOKEN_CACHE_MAXSIZE, ttl=_REVOKED_TOKEN_CACHE_TTL
    )


def clear_revocation_cache() -> None:
    """Clear the in-memory cache of revoked token check results (Issue #3018)."""
    global _revoked_token_cache
    _revoked_token_cache.clear()


_last_revoked_cleanup = 0.0


def _cleanup_revoked_tokens() -> int:
    """Delete expired JWT tokens and their corresponding SHA-256 signatures from revoked_tokens.

    Returns:
        The number of rows deleted.
    """
    import base64
    import hashlib
    import json
    import time

    deleted_count = 0
    try:
        with _connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='revoked_tokens'"
            )
            if not cursor.fetchone():
                return 0

            cursor = conn.execute("SELECT token_signature FROM revoked_tokens")
            rows = cursor.fetchall()

            now_ts = int(time.time())
            expired_signatures = []

            for row in rows:
                token_sig = row[0]
                if not token_sig:
                    continue
                parts = token_sig.split(".")
                if len(parts) == 3:
                    try:
                        payload_b64 = parts[1]
                        rem = len(payload_b64) % 4
                        if rem > 0:
                            payload_b64 += "=" * (4 - rem)
                        payload_bytes = base64.urlsafe_b64decode(payload_b64)
                        payload = json.loads(payload_bytes.decode("utf-8"))
                        exp = payload.get("exp")
                        if exp is not None:
                            exp_int = int(exp)
                            if now_ts >= exp_int:
                                expired_signatures.append(token_sig)
                                token_hash = hashlib.sha256(
                                    token_sig.encode("utf-8")
                                ).hexdigest()
                                expired_signatures.append(token_hash)
                    except Exception:
                        pass

            if expired_signatures:
                placeholders = ",".join("?" for _ in expired_signatures)
                cur = conn.execute(
                    f"DELETE FROM revoked_tokens WHERE token_signature IN ({placeholders})",  # nosec
                    expired_signatures,
                )
                deleted_count = cur.rowcount
                conn.commit()
                if deleted_count > 0:
                    clear_revocation_cache()
                    logger.info(
                        f"Cleaned up {deleted_count} expired entries from revoked_tokens table."
                    )
    except Exception as e:
        logger.error(f"Failed to cleanup revoked tokens: {e}")
    return deleted_count


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
            # Update in-memory TTLCache immediately on revocation
            _revoked_token_cache[token] = True
            _revoked_token_cache[signature] = True
            log_security_event(
                event_type="token_revocation",
                username="system",
                details=details or f"Token signature {signature[:12]}... revoked",
            )
        global _last_revoked_cleanup
        now = time.time()
        if now - _last_revoked_cleanup > 3600:
            _last_revoked_cleanup = now
            _cleanup_revoked_tokens()
    except sqlite3.Error as e:
        logger.error(f"Failed to revoke token: {e}")
        raise sqlite3.Error(f"Failed to revoke token: {e}") from e


def is_token_revoked(token: str) -> bool:
    """Return True if the token or its SHA-256 signature exists in revoked_tokens.

    Caches query results in-memory using cachetools.TTLCache (60s TTL) to drastically
    reduce database disk reads and latency on authenticated requests (Issue #3018).
    """
    if not token or not isinstance(token, str):
        return False

    token = token.strip()
    if not token:
        return False

    # 1. Check in-memory TTLCache first
    if token in _revoked_token_cache:
        return _revoked_token_cache[token]

    signature = _get_token_signature(token)
    if signature in _revoked_token_cache:
        return _revoked_token_cache[signature]

    global _last_revoked_cleanup
    now = time.time()
    if now - _last_revoked_cleanup > 3600:
        _last_revoked_cleanup = now
        _cleanup_revoked_tokens()

    try:
        with _connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='revoked_tokens'"
            )
            if not cursor.fetchone():
                _revoked_token_cache[token] = False
                _revoked_token_cache[signature] = False
                return False

            row = conn.execute(
                "SELECT 1 FROM revoked_tokens WHERE token_signature = ? OR token_signature = ? LIMIT 1",
                (signature, token),
            ).fetchone()
            revoked = bool(row)
            _revoked_token_cache[token] = revoked
            _revoked_token_cache[signature] = revoked
            return revoked
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


def format_user_creation_date(iso_str: str) -> str:
    """Format an ISO creation date as 'MMM DD, YYYY'."""
    date = dt.fromisoformat(iso_str.replace("Z", "+00:00"))
    return date.strftime("%b %d, %Y")


# ============================================================================
# ROLE-BASED ACCESS CONTROL (RBAC) ENHANCEMENTS - Issue #2171
# ============================================================================

from enum import Enum
from functools import wraps
from typing import Set

import streamlit as st

# ============================================================================
# ROLE DEFINITIONS
# ============================================================================


class UserRole(Enum):
    """User roles with hierarchical permissions."""

    USER = "user"
    TEACHER = "teacher"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

    @classmethod
    def from_string(cls, role: str) -> UserRole:
        """Convert string to UserRole enum."""
        try:
            return cls(role.lower())
        except ValueError:
            return cls.USER

    def level(self) -> int:
        """Get role hierarchy level (higher = more permissions)."""
        levels = {
            UserRole.USER: 0,
            UserRole.TEACHER: 1,
            UserRole.ADMIN: 2,
            UserRole.SUPER_ADMIN: 3,
        }
        return levels.get(self, 0)

    def has_permission(self, required_role: UserRole) -> bool:
        """Check if this role has permission for a required role."""
        return self.level() >= required_role.level()


# ============================================================================
# PERMISSION DEFINITIONS
# ============================================================================


class Permission(Enum):
    """Available permissions in the system."""

    # User permissions
    VIEW_DASHBOARD = "view_dashboard"
    VIEW_PROFILE = "view_profile"
    EDIT_PROFILE = "edit_profile"

    # Document permissions
    UPLOAD_DOCUMENTS = "upload_documents"
    VIEW_DOCUMENTS = "view_documents"
    DELETE_DOCUMENTS = "delete_documents"
    EXPORT_DOCUMENTS = "export_documents"

    # Analysis permissions
    RUN_ANALYSIS = "run_analysis"
    VIEW_ANALYSIS = "view_analysis"
    DELETE_ANALYSIS = "delete_analysis"
    EXPORT_ANALYSIS = "export_analysis"

    # User management permissions
    VIEW_USERS = "view_users"
    CREATE_USERS = "create_users"
    EDIT_USERS = "edit_users"
    DELETE_USERS = "delete_users"
    MANAGE_ROLES = "manage_roles"

    # System permissions
    VIEW_LOGS = "view_logs"
    VIEW_SETTINGS = "view_settings"
    EDIT_SETTINGS = "edit_settings"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_BACKUPS = "manage_backups"
    VIEW_SYSTEM_HEALTH = "view_system_health"


# ============================================================================
# ROLE-PERMISSION MAPPING
# ============================================================================

_ROLE_PERMISSIONS: dict[UserRole, set[Permission]] = {
    UserRole.USER: {
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_PROFILE,
        Permission.EDIT_PROFILE,
        Permission.VIEW_DOCUMENTS,
        Permission.VIEW_ANALYSIS,
    },
    UserRole.TEACHER: {
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_PROFILE,
        Permission.EDIT_PROFILE,
        Permission.UPLOAD_DOCUMENTS,
        Permission.VIEW_DOCUMENTS,
        Permission.EXPORT_DOCUMENTS,
        Permission.RUN_ANALYSIS,
        Permission.VIEW_ANALYSIS,
        Permission.EXPORT_ANALYSIS,
        Permission.VIEW_USERS,
    },
    UserRole.ADMIN: {
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_PROFILE,
        Permission.EDIT_PROFILE,
        Permission.UPLOAD_DOCUMENTS,
        Permission.VIEW_DOCUMENTS,
        Permission.DELETE_DOCUMENTS,
        Permission.EXPORT_DOCUMENTS,
        Permission.RUN_ANALYSIS,
        Permission.VIEW_ANALYSIS,
        Permission.DELETE_ANALYSIS,
        Permission.EXPORT_ANALYSIS,
        Permission.VIEW_USERS,
        Permission.CREATE_USERS,
        Permission.EDIT_USERS,
        Permission.DELETE_USERS,
        Permission.MANAGE_ROLES,
        Permission.VIEW_LOGS,
        Permission.VIEW_SETTINGS,
        Permission.EDIT_SETTINGS,
        Permission.VIEW_AUDIT_LOGS,
        Permission.MANAGE_BACKUPS,
        Permission.VIEW_SYSTEM_HEALTH,
    },
    UserRole.SUPER_ADMIN: {
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_PROFILE,
        Permission.EDIT_PROFILE,
        Permission.UPLOAD_DOCUMENTS,
        Permission.VIEW_DOCUMENTS,
        Permission.DELETE_DOCUMENTS,
        Permission.EXPORT_DOCUMENTS,
        Permission.RUN_ANALYSIS,
        Permission.VIEW_ANALYSIS,
        Permission.DELETE_ANALYSIS,
        Permission.EXPORT_ANALYSIS,
        Permission.VIEW_USERS,
        Permission.CREATE_USERS,
        Permission.EDIT_USERS,
        Permission.DELETE_USERS,
        Permission.MANAGE_ROLES,
        Permission.VIEW_LOGS,
        Permission.VIEW_SETTINGS,
        Permission.EDIT_SETTINGS,
        Permission.VIEW_AUDIT_LOGS,
        Permission.MANAGE_BACKUPS,
        Permission.VIEW_SYSTEM_HEALTH,
    },
}


# ============================================================================
# PERMISSION CHECK FUNCTIONS
# ============================================================================


def get_role_permissions(role: UserRole) -> set[Permission]:
    """Get all permissions for a role."""
    return _ROLE_PERMISSIONS.get(role, set())


def has_permission(username: str, permission: Permission) -> bool:
    """
    Check if a user has a specific permission.

    Args:
        username: The username to check
        permission: The permission to check for

    Returns:
        bool: True if user has the permission
    """
    try:
        role_str = get_user_role(username)
        role = UserRole.from_string(role_str)
        permissions = get_role_permissions(role)
        return permission in permissions
    except Exception as e:
        logger.error(f"Failed to check permission for {username}: {e}")
        return False


def has_any_permission(username: str, *permissions: Permission) -> bool:
    """Check if a user has any of the given permissions."""
    for permission in permissions:
        if has_permission(username, permission):
            return True
    return False


def has_all_permissions(username: str, *permissions: Permission) -> bool:
    """Check if a user has all of the given permissions."""
    for permission in permissions:
        if not has_permission(username, permission):
            return False
    return True


def require_permission(permission: Permission):
    """
    Decorator to require a specific permission for a function.

    Usage:
        @require_permission(Permission.VIEW_AUDIT_LOGS)
        def admin_function():
            pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get username from session state
            username = st.session_state.get(SessionKeys.USERNAME)  # noqa: F821
            if not username:
                st.error("🔒 Authentication required.")
                return None

            if not has_permission(username, permission):
                st.error(f"🔒 Permission denied. Requires: {permission.value}")
                return None

            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_role(required_role: UserRole):
    """
    Decorator to require a specific role for a function.

    Usage:
        @require_role(UserRole.ADMIN)
        def admin_function():
            pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            username = st.session_state.get(SessionKeys.USERNAME)  # noqa: F821
            if not username:
                st.error("🔒 Authentication required.")
                return None

            role_str = get_user_role(username)
            role = UserRole.from_string(role_str)

            if not role.has_permission(required_role):
                st.error(f"🔒 Role required: {required_role.value}")
                return None

            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# RBAC HELPERS
# ============================================================================


def get_user_role_enhanced(username: str) -> dict[str, Any]:
    """
    Get enhanced user role information.

    Returns:
        Dict with role, level, permissions, and hierarchy info
    """
    role_str = get_user_role(username)
    role = UserRole.from_string(role_str)
    permissions = get_role_permissions(role)

    return {
        "username": username,
        "role": role_str,
        "role_enum": role,
        "level": role.level(),
        "permissions": [p.value for p in permissions],
        "permission_count": len(permissions),
        "is_admin": role in [UserRole.ADMIN, UserRole.SUPER_ADMIN],
        "is_teacher": role in [UserRole.TEACHER, UserRole.ADMIN, UserRole.SUPER_ADMIN],
        "is_super_admin": role == UserRole.SUPER_ADMIN,
    }


def get_roles_hierarchy() -> dict[str, int]:
    """Get the hierarchy levels for all roles."""
    return {role.value: role.level() for role in UserRole}


def get_available_permissions() -> list[str]:
    """Get list of all available permissions."""
    return [p.value for p in Permission]


def get_roles_summary() -> dict[str, dict[str, Any]]:
    """Get summary of all roles and their permissions."""
    summary = {}
    for role in UserRole:
        permissions = get_role_permissions(role)
        summary[role.value] = {
            "level": role.level(),
            "permission_count": len(permissions),
            "permissions": [p.value for p in permissions],
        }
    return summary


def get_users_by_role(role: UserRole) -> list[str]:
    """
    Get all users with a specific role.

    Args:
        role: The role to filter by

    Returns:
        List of usernames with the specified role
    """
    try:
        from src.db.auth import get_all_users

        users = get_all_users()
        return [
            user["username"]
            for user in users
            if UserRole.from_string(user["role"]) == role
        ]
    except Exception as e:
        logger.error(f"Failed to get users by role: {e}")
        return []


def get_users_by_permission(permission: Permission) -> list[str]:
    """
    Get all users who have a specific permission.

    Args:
        permission: The permission to check

    Returns:
        List of usernames with the permission
    """
    try:
        from src.db.auth import get_all_users

        users = get_all_users()
        return [
            user["username"]
            for user in users
            if has_permission(user["username"], permission)
        ]
    except Exception as e:
        logger.error(f"Failed to get users by permission: {e}")
        return []


def promote_user(username: str, new_role: UserRole, admin_username: str) -> bool:
    """
    Promote a user to a new role.

    Args:
        username: The user to promote
        new_role: The new role
        admin_username: The admin performing the promotion

    Returns:
        bool: True if promotion was successful
    """
    try:
        username = _validate_username(username)
        admin_role = UserRole.from_string(get_user_role(admin_username))

        # Only admins can promote users
        if not admin_role.has_permission(UserRole.ADMIN):
            raise PermissionError("Only admins can promote users")

        # Cannot promote to higher than admin
        if new_role.level() > UserRole.ADMIN.level():
            raise ValueError("Cannot promote users to Super Admin")

        with _connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET role = ? WHERE username = ?",
                (new_role.value, username),
            )
            affected = cursor.rowcount
            conn.commit()

            if affected > 0:
                log_security_event(
                    event_type="user_role_changed",
                    username=username,
                    details=f"Role changed to {new_role.value} by {admin_username}",
                )
                return True
            return False

    except Exception as e:
        logger.error(f"Failed to promote user {username}: {e}")
        return False


# ============================================================================
# SSO SECURITY ENHANCEMENTS - Issue #2172
# ============================================================================

# ============================================================================
# SECURE PASSWORD GENERATION
# ============================================================================


def generate_secure_password(length: int = 32) -> str:
    """
    Generate a cryptographically secure random password.

    Args:
        length: Length of the password (default: 32 characters)

    Returns:
        A secure random password containing uppercase, lowercase, digits, and symbols

    Examples:
        >>> generate_secure_password(16)
        'K#9mP$2vL&8qR!x4'
    """
    if length < 12:
        raise ValueError("Password length must be at least 12 characters for security.")

    alphabet = string.ascii_letters + string.digits + string.punctuation
    password = "".join(secrets.choice(alphabet) for _ in range(length))

    # Ensure password meets complexity requirements
    while True:
        try:
            _validate_password_complexity(password)
            break
        except ValueError:
            password = "".join(secrets.choice(alphabet) for _ in range(length))

    return password


def generate_sso_token() -> str:
    """
    Generate a secure token for SSO session management.

    Returns:
        A 64-character hex token
    """
    return secrets.token_hex(64)


def store_sso_state(state: str, expires_in_seconds: int = 600) -> bool:
    """
    Store an OAuth SSO state parameter in the database with an expiration time.

    Args:
        state: The state token string.
        expires_in_seconds: Lifetime of state in seconds (default 600s / 10m).

    Returns:
        bool: True if state was stored successfully.
    """
    if not state:
        return False
    try:
        expires_at = (dt.now() + timedelta(seconds=expires_in_seconds)).isoformat()
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sso_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state TEXT UNIQUE NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    used_at TEXT DEFAULT NULL,
                    expires_at TEXT NOT NULL
                )
            """
            )
            conn.execute(
                """INSERT OR REPLACE INTO sso_states (state, expires_at, used_at)
                   VALUES (?, ?, NULL)""",
                (state, expires_at),
            )
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to store SSO state: {e}")
        return False


def validate_sso_state(state: str) -> bool:
    """
    Validate an OAuth SSO state parameter and invalidate it after validation to prevent replay attacks.

    Args:
        state: The state token to validate.

    Returns:
        bool: True if valid, unexpired, and not previously used; False otherwise.
    """
    if not state:
        return False

    try:
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sso_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state TEXT UNIQUE NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    used_at TEXT DEFAULT NULL,
                    expires_at TEXT NOT NULL
                )
            """
            )
            row = conn.execute(
                "SELECT expires_at, used_at FROM sso_states WHERE state = ?", (state,)
            ).fetchone()

            if not row:
                return False

            expires_at, used_at = row
            if used_at is not None:
                logger.warning(f"OAuth state replay attack detected for state: {state}")
                return False

            if expires_at < dt.now().isoformat():
                logger.warning(f"OAuth state expired for state: {state}")
                return False

            # Invalidate state immediately after validation to prevent replay attacks
            conn.execute(
                "UPDATE sso_states SET used_at = CURRENT_TIMESTAMP WHERE state = ?",
                (state,),
            )
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Failed to validate SSO state: {e}")
        return False


def verify_sso_state(state: str) -> bool:
    """Alias for validate_sso_state."""
    return validate_sso_state(state)


def generate_sso_state() -> str:
    """
    Generate a secure state parameter for OAuth2 flow and store it.

    Returns:
        A 32-character hex state token
    """
    state = secrets.token_hex(32)
    store_sso_state(state)
    return state


# ============================================================================
# SSO USER MANAGEMENT
# ============================================================================


def get_or_create_sso_user_enhanced(
    email: str, provider: str, provider_user_id: str, default_role: str = "teacher"
) -> dict[str, Any]:
    """
    Enhanced SSO user creation with security features.

    This function:
    1. Checks if user exists
    2. Updates SSO provider info if needed
    3. Creates user with secure random password
    4. Logs security events
    5. Returns user info with security status

    Args:
        email: User's email address
        provider: SSO provider (github, google, etc.)
        provider_user_id: User ID from the provider
        default_role: Default role for new users

    Returns:
        Dict containing user info and security status
    """
    username = _validate_username(email)
    provider = provider.lower()

    if provider not in ["github", "google", "microsoft", "gitlab"]:
        raise ValueError(f"Unsupported SSO provider: {provider}")

    with _connect() as conn:
        # Check if user exists
        row = conn.execute(
            "SELECT id, role, sso_provider, sso_provider_user_id FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if row:
            user_id, role, existing_provider, existing_provider_id = row

            # Update provider info if changed
            if (
                existing_provider != provider
                or existing_provider_id != provider_user_id
            ):
                conn.execute(
                    """UPDATE users
                       SET sso_provider = ?,
                           sso_provider_user_id = ?,
                           updated_at = CURRENT_TIMESTAMP
                       WHERE username = ?""",
                    (provider, provider_user_id, username),
                )
                conn.commit()

                log_security_event(
                    event_type="sso_provider_updated",
                    username=username,
                    details=f"SSO provider updated from {existing_provider} to {provider}",
                )

            # Log successful SSO login
            log_security_event(
                event_type="sso_login_success",
                username=username,
                details=f"SSO login via {provider} (user_id: {user_id})",
            )

            return {
                "username": username,
                "role": role,
                "user_id": user_id,
                "is_new_user": False,
                "provider": provider,
                "sso_enabled": True,
            }

        # New user - generate secure random password
        secure_password = generate_secure_password(32)
        hashed = _hash_password(secure_password)
        role = _validate_role(default_role)

        # Insert new user with SSO info
        cursor = conn.execute(
            """INSERT INTO users
               (username, password, role, sso_provider, sso_provider_user_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
            (username, hashed, role, provider, provider_user_id),
        )
        user_id = cursor.lastrowid
        conn.commit()

        # Store secure password in a separate table for recovery (optional)
        _store_sso_recovery_token(username, secure_password)

        # Log security events
        log_security_event(
            event_type="sso_user_created",
            username=username,
            details=f"SSO user created via {provider} with role: {role}",
        )

        log_security_event(
            event_type="user_created",
            username=username,
            details=f"User created via SSO ({provider}) with secure random password",
        )

        return {
            "username": username,
            "role": role,
            "user_id": user_id,
            "is_new_user": True,
            "provider": provider,
            "sso_enabled": True,
            "secure_password_set": True,
        }


def _store_sso_recovery_token(username: str, password: str) -> None:
    """
    Store a recovery token for SSO users in case they need to reset password.
    """
    try:
        token = generate_sso_token()
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = (dt.now() + timedelta(days=7)).isoformat()

        with _connect() as conn:
            # Create recovery_tokens table if not exists
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sso_recovery_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    token_hash TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    used_at TEXT DEFAULT NULL,
                    FOREIGN KEY (username) REFERENCES users(username)
                )
            """
            )

            # Store recovery token
            conn.execute(
                """INSERT INTO sso_recovery_tokens (username, token_hash, expires_at)
                   VALUES (?, ?, ?)""",
                (username, token_hash, expires_at),
            )
            conn.commit()

            # Log for security
            log_security_event(
                event_type="sso_recovery_token_created",
                username=username,
                details="SSO recovery token created",
            )
    except Exception as e:
        logger.error(f"Failed to store SSO recovery token: {e}")


def verify_sso_recovery_token(username: str, token: str) -> bool:
    """
    Verify an SSO recovery token.

    Args:
        username: The username
        token: The recovery token to verify

    Returns:
        True if token is valid and not expired
    """
    try:
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        with _connect() as conn:
            row = conn.execute(
                """SELECT expires_at, used_at FROM sso_recovery_tokens
                   WHERE username = ? AND token_hash = ?""",
                (username, token_hash),
            ).fetchone()

            if not row:
                return False

            expires_at, used_at = row
            if used_at is not None:
                return False  # Token already used

            if expires_at < dt.now().isoformat():
                return False  # Token expired

            # Mark token as used
            conn.execute(
                "UPDATE sso_recovery_tokens SET used_at = CURRENT_TIMESTAMP WHERE username = ? AND token_hash = ?",
                (username, token_hash),
            )
            conn.commit()
            return True

    except Exception as e:
        logger.error(f"Failed to verify SSO recovery token: {e}")
        return False


def get_sso_user_info(username: str) -> dict[str, Any] | None:
    """
    Get SSO user information.

    Args:
        username: The username to lookup

    Returns:
        Dict with SSO info or None if not an SSO user
    """
    try:
        username = _validate_username(username)
        with _connect() as conn:
            row = conn.execute(
                """SELECT id, username, role, sso_provider, sso_provider_user_id,
                          created_at, updated_at, status
                   FROM users WHERE username = ? AND sso_provider IS NOT NULL""",
                (username,),
            ).fetchone()

            if not row:
                return None

            return {
                "user_id": row[0],
                "username": row[1],
                "role": row[2],
                "sso_provider": row[3],
                "sso_provider_user_id": row[4],
                "created_at": row[5],
                "updated_at": row[6],
                "status": row[7],
                "is_active": (row[7] == "active"),
                "is_sso_user": True,
            }
    except Exception as e:
        logger.error(f"Failed to get SSO user info: {e}")
        return None


def list_sso_users() -> list[dict[str, Any]]:
    """
    List all SSO users in the system.

    Returns:
        List of SSO user info dicts
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                """SELECT id, username, role, sso_provider, sso_provider_user_id,
                          created_at, updated_at, status
                   FROM users WHERE sso_provider IS NOT NULL
                   ORDER BY created_at DESC"""
            ).fetchall()

            return [
                {
                    "user_id": row[0],
                    "username": row[1],
                    "role": row[2],
                    "sso_provider": row[3],
                    "sso_provider_user_id": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "status": row[7],
                    "is_active": (row[7] == "active"),
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"Failed to list SSO users: {e}")
        return []


def revoke_sso_access(username: str) -> bool:
    """
    Unlink a user's SSO provider.

    Clears ``sso_provider`` and ``sso_provider_user_id`` so the account can no
    longer authenticate through the identity provider. The local account, its
    role and its password hash are left untouched.

    The provider is read before the update so the audit entry can name it;
    afterwards the column is NULL and the information is gone.

    Args:
        username: The username to revoke access for

    Returns:
        True when a linked provider was cleared. False when the user does not
        exist, had no provider linked, or the update failed.
    """
    try:
        username = _validate_username(username)
        with _connect() as conn:
            row = conn.execute(
                "SELECT sso_provider FROM users WHERE username = ?",
                (username,),
            ).fetchone()

            if row is None or row[0] is None:
                # Nothing to revoke: an unknown user, or a local-only account.
                # Not an error, but not a revocation either, so no audit entry.
                return False

            provider = row[0]

            cursor = conn.execute(
                """UPDATE users
                   SET sso_provider = NULL,
                       sso_provider_user_id = NULL,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE username = ?""",
                (username,),
            )
            affected = cursor.rowcount
            conn.commit()

            if affected > 0:
                log_security_event(
                    event_type="sso_access_revoked",
                    username=username,
                    details=f"SSO access revoked for provider {provider}",
                )
                return True
            return False

    except Exception as e:
        logger.error(f"Failed to revoke SSO access for {username}: {e}")
        return False


def demote_user(username: str, admin_username: str) -> bool:
    """
    Demote a user to the default USER role.

    Args:
        username: The user to demote
        admin_username: The admin performing the demotion

    Returns:
        bool: True if demotion was successful
    """
    return promote_user(username, UserRole.USER, admin_username)


# ============================================================================
# STREAMLIT UI HELPERS
# ============================================================================


def render_role_badge(role: str) -> str:
    """
    Render a role badge HTML.

    Args:
        role: The role string

    Returns:
        HTML string for the role badge
    """
    badges = {
        "super_admin": '<span style="background: #8B0000; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem;">🛡️ Super Admin</span>',
        "admin": '<span style="background: #1e3a8a; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem;">🔑 Admin</span>',
        "teacher": '<span style="background: #0d9488; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem;">👨‍🏫 Teacher</span>',
        "user": '<span style="background: #6b7280; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem;">👤 User</span>',
    }
    return badges.get(role.lower(), badges.get("user"))


def render_role_selector(username: str, current_role: str) -> None:
    """
    Render a role selector dropdown for admin users.

    Args:
        username: The user to change role for
        current_role: The current role
    """
    roles = [r.value for r in UserRole if r != UserRole.SUPER_ADMIN]
    selected = st.selectbox(
        f"Role for {username}",
        options=roles,
        index=roles.index(current_role) if current_role in roles else 0,
        key=f"role_select_{username}",
    )

    if selected != current_role:
        if st.button(f"Update Role for {username}", key=f"role_update_{username}"):
            admin = st.session_state.get(SessionKeys.USERNAME)  # noqa: F821
            new_role = UserRole.from_string(selected)
            if promote_user(username, new_role, admin):
                st.success(f"✅ Role updated to {selected} for {username}")
                st.rerun()
            else:
                st.error("❌ Failed to update role")


def render_permission_checklist(username: str) -> None:
    """
    Render a checklist of permissions for a user.

    Args:
        username: The user to display permissions for
    """
    role_str = get_user_role(username)
    role = UserRole.from_string(role_str)
    permissions = get_role_permissions(role)

    st.markdown(f"### Permissions for {username}")
    st.caption(f"Role: {role_str} (Level {role.level()})")

    cols = st.columns(3)
    for idx, permission in enumerate(sorted(permissions, key=lambda x: x.value)):
        col_idx = idx % 3
        with cols[col_idx]:
            st.markdown(f"✅ {permission.value}")


def get_sso_users_count() -> int:
    """
    Get the total number of SSO users.

    Returns:
        Count of SSO users
    """
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM users WHERE sso_provider IS NOT NULL"
            ).fetchone()
            return row[0] if row else 0
    except Exception as e:
        logger.error(f"Failed to count SSO users: {e}")
        return 0


def migrate_existing_sso_users() -> dict[str, Any]:
    """
    Migrate existing SSO users to secure passwords.

    This function finds all SSO users with weak passwords and
    upgrades them to secure random passwords.

    Returns:
        Dict with migration statistics
    """
    try:
        from src.db.auth import get_all_users, update_password  # noqa: F401

        sso_users = list_sso_users()
        migrated = 0
        failed = 0

        for user in sso_users:
            try:
                username = user["username"]
                # Generate new secure password
                new_password = generate_secure_password(32)
                # Update password
                update_password(username, new_password)
                migrated += 1

                log_security_event(
                    event_type="sso_password_upgraded",
                    username=username,
                    details="SSO user password upgraded from weak to secure",
                )
            except Exception as e:
                logger.error(f"Failed to upgrade password for {username}: {e}")
                failed += 1

        return {
            "total_sso_users": len(sso_users),
            "migrated": migrated,
            "failed": failed,
            "success": failed == 0,
        }

    except Exception as e:
        logger.error(f"Failed to migrate SSO users: {e}")
        return {
            "total_sso_users": 0,
            "migrated": 0,
            "failed": 1,
            "success": False,
            "error": str(e),
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "UserRole",
    "Permission",
    "get_role_permissions",
    "has_permission",
    "has_any_permission",
    "has_all_permissions",
    "require_permission",
    "require_role",
    "get_user_role_enhanced",
    "get_roles_hierarchy",
    "get_available_permissions",
    "get_roles_summary",
    "get_users_by_role",
    "get_users_by_permission",
    "promote_user",
    "demote_user",
    "render_role_badge",
    "render_role_selector",
    "render_permission_checklist",
    "generate_secure_password",
    "generate_sso_token",
    "generate_sso_state",
    "store_sso_state",
    "validate_sso_state",
    "verify_sso_state",
    "get_or_create_sso_user_enhanced",
    "get_sso_user_info",
    "list_sso_users",
    "revoke_sso_access",
    "is_sso_user",
    "get_sso_users_count",
    "migrate_existing_sso_users",
    "verify_sso_recovery_token",
    "validate_password_complexity",
    "_validate_password_complexity",
]
