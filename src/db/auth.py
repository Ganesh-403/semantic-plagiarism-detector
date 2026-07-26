from __future__ import annotations
import json

"""

src/db/auth.py
--------------
User authentication, registration, and credential management routines.
auth.py
-------
SQLite-backed authentication with Argon2 password hashing (via argon2-cffi)
and automatic transparent migration from legacy bcrypt hashes.

Public API
----------
init_db()                         → create tables + seed default admin
verify_user(username, password)    → bool
get_user_role(username)            → str | None
add_user(username, password, role) → None
get_all_users()                    → list[dict]
delete_user(username)              → None
update_password(username, password)→ None
get_tour_completed(username)       → bool
set_tour_completed(username, completed) → None
"""

import os
import sqlite3

import bcrypt
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError

# Database setup
from src.db.migrations import migrate_auth_database

_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "users.db")
)


VALID_ROLES = {"admin", "teacher"}

# Initialize Argon2 password hasher
_ph = PasswordHasher()


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH, check_same_thread=False)


VALID_ROLES = {"admin", "teacher"}


def _hash_password(password: str) -> str:
    """Return an Argon2 hash for the given password."""
    return _ph.hash(password)


def _validate_username(username: str) -> str:
    username = str(username).strip().lower()
    if not username:
        raise ValueError("Username cannot be empty.")
    return username


def _validate_password(password: str) -> str:
    try:
        password = str(password)
        if len(password.strip()) < 5:
            raise ValueError("Password must be at least 5 characters long.")
        return password
    finally:
        password = "REDACTED"


def _validate_role(role: str) -> str:
    role = str(role).strip().lower()
    if role not in VALID_ROLES:
        raise ValueError(f"Role must be one of: {', '.join(sorted(VALID_ROLES))}")
    return role


def init_db() -> None:
    """Create or upgrade users.db and seed the default administrator."""
    try:
        with _connect() as conn:
            migrate_auth_database(conn)

            row = conn.execute(
                "SELECT COUNT(1) FROM users WHERE username = ?",
                ("admin",),
            ).fetchone()
            exists = bool(row and row[0])

            if not exists:
                hashed = _hash_password("admin123")
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

    # Restrict database file permissions to owner read/write only
    # Prevents other local users on the server from reading user credentials
    try:
        os.chmod(_DB_PATH, 0o600)
    except OSError:
        pass  # Best-effort; some platforms (e.g., Windows) may not support chmod


def verify_user(username: str, password: str) -> bool:
    """Return True if username exists, password matches the stored hash, and account is active."""
    init_db()  # Ensure DB is initialized
    try:
        username = _validate_username(username)
        password = _validate_password(password)
    except ValueError:
        return False

    with _connect() as conn:
        row = conn.execute(
            "SELECT password, is_active FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if not row:
        return False

    stored_hash, is_active = row
    if not is_active:
        return False

    try:
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    except ValueError:
        return False

        try:
            return bcrypt.checkpw(password.encode(), stored_hash.encode())
        except ValueError:
            return False

# Alias for compatibility
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


def add_user(username: str, password: str, role: str = "teacher") -> None:
    """Insert a user and preserve SQLite duplicate-user semantics."""
    try:
        username = _validate_username(username)
        password = _validate_password(password)
        role = _validate_role(role)

        hashed = _hash_password(password)

        with _connect() as conn:
            # The UNIQUE constraint is the source of truth. Existing callers and
            # tests rely on sqlite3.IntegrityError for duplicate usernames.
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, hashed, role),
            )
            conn.commit()
    except sqlite3.IntegrityError as e:
        raise ValueError(f"Username '{username}' already exists.") from e
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to add user: {e}") from e
    finally:
        password = "REDACTED"



        if not exists:
            hashed = _hash_password("admin123")

def get_all_users() -> list:
    """Return all users as a list of dicts (excludes password hashes)."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT id, username, role, is_active FROM users ORDER BY id"
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "username": r[1],
                    "role": r[2],
                    "is_active": bool(r[3]),
                }
                for r in rows
            ]
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to retrieve users: {e}") from e


def delete_user(username: str) -> None:
    """Delete a user by username."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            conn.execute(
                "DELETE FROM users WHERE username = ?",
                (username,),
            )
            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to delete user: {e}") from e



def update_password(username: str, new_password: str) -> None:
    """Update a user's password with a new bcrypt hash."""
    try:
        username = _validate_username(username)
        new_password = _validate_password(new_password)

        with _connect() as conn:
            # Optimized check using COUNT(1) for #185
            cursor = conn.execute(
                "SELECT COUNT(1) FROM users WHERE username = ?",
                (username,),
            )
            if cursor.fetchone()[0] == 0:
                raise ValueError("User not found.")

            hashed = _hash_password(new_password)
            conn.execute(
                "UPDATE users SET password = ? WHERE username = ?",
                (hashed, username),
            )
            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to update password: {e}") from e
    finally:
        new_password = "REDACTED"


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


def verify_user(username: str, password: str) -> bool:
    """
    Return True if username exists and password matches stored hash.
    Supports both Argon2 and legacy bcrypt hashes, automatically migrating 
    bcrypt hashes to Argon2 upon successful login.
    """
    username = _validate_username(username)
    password = _validate_password(password)


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


    stored_hash = row[0]

    # Case 1: Stored hash is Argon2
    if stored_hash.startswith("$argon2"):
        try:
            _ph.verify(stored_hash, password)
            if _ph.check_needs_rehash(stored_hash):
                update_password(username, password)
            return True
        except (VerifyMismatchError, VerificationError):
            return False

    # Case 2: Legacy Bcrypt hash -> Verify & migrate to Argon2
    if stored_hash.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
                update_password(username, password)
                return True
        except ValueError:
            return False

    return False

def enable_2fa(username: str, secret: str) -> None:
    """Enable 2FA for a user and store their OTP secret."""
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET two_factor_enabled = 1, otp_secret = ? WHERE username = ?",
            (secret, username.lower()),
        )
        conn.commit()


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


def add_user(username: str, password: str, role: str = "teacher") -> None:
    """Insert a new user with an Argon2-hashed password."""
    username = _validate_username(username)
    password = _validate_password(password)
    role = _validate_role(role)

    hashed = _hash_password(password)

    identifier = username.lower()
    if is_login_locked_out(identifier):
        attempts = get_login_attempts(identifier)
        return (
            False,
            f"Account locked due to too many failed attempts. Please try again in 15 minutes. ({attempts}/5 attempts)",
        )
    return True, None


def record_failed_login(username: str) -> None:
    """Record a failed login attempt for rate limiting."""
    from src.utils.redis_cache import increment_login_attempts

    identifier = username.lower()
    increment_login_attempts(identifier)



def clear_login_attempts(username: str) -> None:
    """Clear failed login attempts after successful login."""
    from src.utils.redis_cache import \
        clear_login_attempts as redis_clear_login_attempts

    identifier = username.lower()
    redis_clear_login_attempts(identifier)


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



def update_password(username: str, new_password: str) -> None:
    """Update a user's password with a new Argon2 hash."""
    username = _validate_username(username)
    new_password = _validate_password(new_password)

    hashed = _hash_password(new_password)

def get_user_theme(username: str) -> str:
    """Return the user's theme preference (default 'light')."""
    username = username.lower()
    with _connect() as conn:
        row = conn.execute(
            "SELECT theme FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return row[0] if row else "light"



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

        # Create user with dummy password
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


def set_user_active_status(username: str, is_active: bool) -> None:
    """Set whether a user account is active (suspended or active)."""
    try:
        username = _validate_username(username)
        with _connect() as conn:
            # We don't allow suspending the 'admin' account to prevent lockouts
            if username == "admin" and not is_active:
                raise ValueError("The admin account cannot be suspended.")

            conn.execute(
                "UPDATE users SET is_active = ? WHERE username = ?",
                (1 if is_active else 0, username),
            )
            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to update user active status: {e}") from e


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
    """
    Returns the total number of registered users in the system.
    This is highly optimized for fast telemetry lookups.
    """
    with _connect() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM users")
        row = cursor.fetchone()
        return row[0] if row else 0
