"""Versioned migrations for users.db."""

import inspect
import sqlite3
import sys

from .common import column_exists, run_migrations

AUTH_SCHEMA_VERSION = 17


def migration_001_create_users(
    connection: sqlite3.Connection,
) -> None:
    """Create the original authentication table."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role     TEXT NOT NULL DEFAULT 'teacher'
        )
        """)


def migration_002_add_onboarding_state(
    connection: sqlite3.Connection,
) -> None:
    """Add onboarding completion state."""
    if not column_exists(connection, "users", "tour_completed"):
        connection.execute("""
            ALTER TABLE users
            ADD COLUMN tour_completed INTEGER NOT NULL DEFAULT 0
            """)


def migration_003_add_two_factor_fields(
    connection: sqlite3.Connection,
) -> None:
    """Add optional TOTP secret and enablement fields."""
    if not column_exists(connection, "users", "otp_secret"):
        connection.execute("ALTER TABLE users ADD COLUMN otp_secret TEXT DEFAULT NULL")

    if not column_exists(connection, "users", "two_factor_enabled"):
        connection.execute("""
            ALTER TABLE users
            ADD COLUMN two_factor_enabled INTEGER NOT NULL DEFAULT 0
            """)


def migration_005_add_preferences(db_cursor):
    db_cursor.execute("ALTER TABLE users ADD COLUMN preferences TEXT DEFAULT '{}'")


def migration_004_add_role_index(
    connection: sqlite3.Connection,
) -> None:
    """Add an index used by role-based administration queries."""
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_role
        ON users(role)
        """)


def migration_006_add_active_flag(
    connection: sqlite3.Connection,
) -> None:
    """Add is_active field to temporarily suspend user accounts."""
    if not column_exists(connection, "users", "is_active"):
        connection.execute("""
            ALTER TABLE users
            ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1
            """)


def migration_007_add_theme_preference(
    connection: sqlite3.Connection,
) -> None:
    """Add theme field for persistent UI preference."""
    if not column_exists(connection, "users", "theme"):
        connection.execute("""
            ALTER TABLE users
            ADD COLUMN theme TEXT NOT NULL DEFAULT 'light'
            """)


def migration_008_create_security_audit_log(
    connection: sqlite3.Connection,
) -> None:
    """Create the security_audit_log table for recording security events."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS security_audit_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT    NOT NULL,
            username   TEXT    NOT NULL,
            timestamp  TEXT    NOT NULL,
            details    TEXT    DEFAULT NULL
        )
        """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_log_username
        ON security_audit_log(username)
        """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_log_event_type
        ON security_audit_log(event_type)
        """)


def migration_009_add_last_login_at(
    connection: sqlite3.Connection,
) -> None:
    """Add last_login_at field for tracking user activity."""
    if not column_exists(connection, "users", "last_login_at"):
        connection.execute("""
            ALTER TABLE users
            ADD COLUMN last_login_at TEXT DEFAULT NULL
            """)


def migration_010_add_password_changed_at(
    connection: sqlite3.Connection,
) -> None:
    """Add password age tracking to authentication records."""
    if not column_exists(
        connection,
        "users",
        "password_changed_at",
    ):
        connection.execute("""
            ALTER TABLE users
            ADD COLUMN password_changed_at TEXT DEFAULT NULL
            """)


def migration_011_add_version_column(
    connection: sqlite3.Connection,
) -> None:
    """Add version column for optimistic locking."""
    if not column_exists(connection, "users", "version"):
        connection.execute("""
            ALTER TABLE users
            ADD COLUMN version INTEGER DEFAULT 1
            """)


def migration_012_create_revoked_tokens_table(
    connection: sqlite3.Connection,
) -> None:
    """Create revoked_tokens table for tracking invalidated tokens."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS revoked_tokens (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            token_signature TEXT UNIQUE NOT NULL,
            revoked_at TEXT NOT NULL,
            details    TEXT DEFAULT NULL
        )
        """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_revoked_tokens_signature
        ON revoked_tokens(token_signature)
        """)


def migration_013_create_password_history_table(
    connection: sqlite3.Connection,
) -> None:
    """Create password_history table for tracking recent password hashes per user."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS password_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at    TEXT NOT NULL
        )
        """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_password_history_username
        ON password_history(username)
        """)


def migration_014_add_user_status(
    connection: sqlite3.Connection,
) -> None:
    """Add account status field and migrate the legacy is_active flag."""
    if not column_exists(connection, "users", "status"):
        connection.execute("""
            ALTER TABLE users
            ADD COLUMN status TEXT NOT NULL DEFAULT 'active'
            """)

    connection.execute("""
        UPDATE users
        SET status = CASE
            WHEN is_active = 0 THEN 'suspended'
            ELSE 'active'
        END
        """)


def migration_015_add_must_change_password(
    connection: sqlite3.Connection,
) -> None:
    """Add must_change_password flag to force password reset on next login."""
    if not column_exists(connection, "users", "must_change_password"):
        connection.execute("""
            ALTER TABLE users
            ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0
            """)


def migration_016_add_audit_log_indexes(
    connection: sqlite3.Connection,
) -> None:
    """Create indexes on security_audit_log(username) and security_audit_log(event_type)."""
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_log_username
        ON security_audit_log(username)
        """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_log_event_type
        ON security_audit_log(event_type)
        """)


def migration_017_drop_is_active(
    connection: sqlite3.Connection,
) -> None:
    """Drop deprecated is_active column using the table rebuild pattern."""
    if not column_exists(connection, "users", "is_active"):
        return

    connection.execute("""
        CREATE TABLE users_temp (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            username              TEXT UNIQUE NOT NULL,
            password              TEXT NOT NULL,
            role                  TEXT NOT NULL DEFAULT 'teacher',
            tour_completed        INTEGER NOT NULL DEFAULT 0,
            otp_secret            TEXT DEFAULT NULL,
            two_factor_enabled    INTEGER NOT NULL DEFAULT 0,
            preferences           TEXT DEFAULT '{}',
            theme                 TEXT NOT NULL DEFAULT 'light',
            last_login_at         TEXT DEFAULT NULL,
            password_changed_at   TEXT DEFAULT NULL,
            version               INTEGER DEFAULT 1,
            status                TEXT NOT NULL DEFAULT 'active',
            must_change_password  INTEGER NOT NULL DEFAULT 0
        )
    """)

    connection.execute("""
        INSERT INTO users_temp (
            id, username, password, role, tour_completed, otp_secret,
            two_factor_enabled, preferences, theme, last_login_at,
            password_changed_at, version, status, must_change_password
        )
        SELECT
            id, username, password, role,
            COALESCE(tour_completed, 0),
            otp_secret,
            COALESCE(two_factor_enabled, 0),
            COALESCE(preferences, '{}'),
            COALESCE(theme, 'light'),
            last_login_at,
            password_changed_at,
            COALESCE(version, 1),
            COALESCE(status, 'active'),
            COALESCE(must_change_password, 0)
        FROM users
    """)

    connection.execute("DROP TABLE users")
    connection.execute("ALTER TABLE users_temp RENAME TO users")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")


def _discover_migrations() -> dict[int, callable]:
    """Dynamically discover migration functions starting with 'migration_' sorted numerically."""
    current_module = sys.modules[__name__]
    migration_funcs = [
        func
        for name, func in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("migration_")
    ]
    # Sort by function name to ensure ascending numeric order (e.g. migration_001, migration_002, ...)
    migration_funcs.sort(key=lambda f: f.__name__)
    return {idx + 1: func for idx, func in enumerate(migration_funcs)}


AUTH_MIGRATIONS = _discover_migrations()


def _drop_column_if_exists(
    connection: sqlite3.Connection, table_name: str, column_name: str
) -> None:
    if column_exists(connection, table_name, column_name):
        try:
            connection.execute(
                f'ALTER TABLE "{table_name}" DROP COLUMN "{column_name}"'
            )
        except sqlite3.OperationalError:
            pass


def down_001_create_users(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS users")


def down_002_add_onboarding_state(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(connection, "users", "tour_completed")


def down_003_add_two_factor_fields(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(connection, "users", "otp_secret")
    _drop_column_if_exists(connection, "users", "two_factor_enabled")


def down_004_add_role_index(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_users_role")


def down_005_add_preferences(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(connection, "users", "preferences")


def down_006_add_active_flag(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(connection, "users", "is_active")


def down_007_add_theme_preference(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(connection, "users", "theme")


def down_008_create_security_audit_log(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_audit_log_username")
    connection.execute("DROP INDEX IF EXISTS idx_audit_log_event_type")
    connection.execute("DROP TABLE IF EXISTS security_audit_log")


def down_009_add_last_login_at(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(connection, "users", "last_login_at")


def down_010_add_password_changed_at(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(connection, "users", "password_changed_at")


def down_011_add_version_column(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(connection, "users", "version")


def down_012_create_revoked_tokens_table(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_revoked_tokens_signature")
    connection.execute("DROP TABLE IF EXISTS revoked_tokens")


def down_013_create_password_history_table(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_password_history_username")
    connection.execute("DROP TABLE IF EXISTS password_history")


def down_014_add_user_status(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(connection, "users", "status")


def down_015_add_must_change_password(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(connection, "users", "must_change_password")


def down_016_add_audit_log_indexes(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_audit_log_username")
    connection.execute("DROP INDEX IF EXISTS idx_audit_log_event_type")


def down_017_drop_is_active(connection: sqlite3.Connection) -> None:
    if not column_exists(connection, "users", "is_active"):
        connection.execute("""
            ALTER TABLE users
            ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1
        """)


AUTH_DOWN_MIGRATIONS = {
    1: down_001_create_users,
    2: down_002_add_onboarding_state,
    3: down_003_add_two_factor_fields,
    4: down_004_add_role_index,
    5: down_005_add_preferences,
    6: down_006_add_active_flag,
    7: down_007_add_theme_preference,
    8: down_008_create_security_audit_log,
    9: down_009_add_last_login_at,
    10: down_010_add_password_changed_at,
    11: down_011_add_version_column,
    12: down_012_create_revoked_tokens_table,
    13: down_013_create_password_history_table,
    14: down_014_add_user_status,
    15: down_015_add_must_change_password,
    16: down_016_add_audit_log_indexes,
    17: down_017_drop_is_active,
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

