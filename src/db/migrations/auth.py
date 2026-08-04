"""Versioned migrations for users.db."""

from __future__ import annotations

import sqlite3

from .common import column_exists, run_migrations
AUTH_SCHEMA_VERSION = 11

def migration_001_create_users(
    connection: sqlite3.Connection,
) -> None:
    """Create the original authentication table."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role     TEXT NOT NULL DEFAULT 'teacher'
        )
        """
    )


def migration_002_add_onboarding_state(
    connection: sqlite3.Connection,
) -> None:
    """Add onboarding completion state."""
    if not column_exists(connection, "users", "tour_completed"):
        connection.execute(
            """
            ALTER TABLE users
            ADD COLUMN tour_completed INTEGER NOT NULL DEFAULT 0
            """
        )


def migration_003_add_two_factor_fields(
    connection: sqlite3.Connection,
) -> None:
    """Add optional TOTP secret and enablement fields."""
    if not column_exists(connection, "users", "otp_secret"):
        connection.execute("ALTER TABLE users ADD COLUMN otp_secret TEXT DEFAULT NULL")

    if not column_exists(connection, "users", "two_factor_enabled"):
        connection.execute(
            """
            ALTER TABLE users
            ADD COLUMN two_factor_enabled INTEGER NOT NULL DEFAULT 0
            """
        )


def migration_005_add_preferences(db_cursor):
    db_cursor.execute("ALTER TABLE users ADD COLUMN preferences TEXT DEFAULT '{}'")


def migration_004_add_role_index(
    connection: sqlite3.Connection,
) -> None:
    """Add an index used by role-based administration queries."""
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_users_role
        ON users(role)
        """
    )


def migration_006_add_active_flag(
    connection: sqlite3.Connection,
) -> None:
    """Add is_active field to temporarily suspend user accounts."""
    if not column_exists(connection, "users", "is_active"):
        connection.execute(
            """
            ALTER TABLE users
            ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1
            """
        )

def migration_007_add_theme_preference(
    connection: sqlite3.Connection,
) -> None:
    """Add theme field for persistent UI preference."""
    if not column_exists(connection, "users", "theme"):
        connection.execute(
            """
            ALTER TABLE users
            ADD COLUMN theme TEXT NOT NULL DEFAULT 'light'
            """
        )


def migration_008_create_security_audit_log(
    connection: sqlite3.Connection,
) -> None:
    """Create the security_audit_log table for recording security events."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS security_audit_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT    NOT NULL,
            username   TEXT    NOT NULL,
            timestamp  TEXT    NOT NULL,
            details    TEXT    DEFAULT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_audit_log_username
        ON security_audit_log(username)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_audit_log_event_type
        ON security_audit_log(event_type)
        """
    )


def migration_009_add_last_login_at(
    connection: sqlite3.Connection,
) -> None:
    """Add last_login_at field for tracking user activity."""
    if not column_exists(connection, "users", "last_login_at"):
        connection.execute(
            """
            ALTER TABLE users
            ADD COLUMN last_login_at TEXT DEFAULT NULL
            """
        )



def migration_010_add_password_changed_at(
    connection: sqlite3.Connection,
) -> None:
    """Add password age tracking to authentication records."""
    if not column_exists(
        connection,
        "users",
        "password_changed_at",
    ):
        connection.execute(
            """
            ALTER TABLE users
            ADD COLUMN password_changed_at TEXT DEFAULT NULL
            """
        )


def migration_011_add_version_column(
    connection: sqlite3.Connection,
) -> None:
    """Add version column for optimistic locking."""
    if not column_exists(connection, "users", "version"):
        connection.execute(
            """
            ALTER TABLE users
            ADD COLUMN version INTEGER DEFAULT 1
            """
        )


AUTH_MIGRATIONS = {
    1: migration_001_create_users,
    2: migration_002_add_onboarding_state,
    3: migration_003_add_two_factor_fields,
    4: migration_004_add_role_index,
    5: migration_005_add_preferences,
    6: migration_006_add_active_flag,
    7: migration_007_add_theme_preference,
    8: migration_008_create_security_audit_log,
    9: migration_009_add_last_login_at,
    10: migration_010_add_password_changed_at,
    11: migration_011_add_version_column,
}


def migrate_auth_database(
    connection: sqlite3.Connection,
) -> int:
    """Upgrade users.db to the latest supported schema version."""
    return run_migrations(
        connection,
        migrations=AUTH_MIGRATIONS,
        target_version=AUTH_SCHEMA_VERSION,
    )
