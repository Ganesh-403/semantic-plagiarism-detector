"""
Shared SQLite schema migration helpers.

This module provides robust, atomic, and rollback-safe utilities for managing
SQLite database schema migrations, including journal mode optimization.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import List

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

Migration: TypeAlias = Callable[[sqlite3.Connection], None]


def quote_identifier(identifier: str) -> str:
    """Return a safely quoted SQLite identifier to prevent SQL injection."""
    value = str(identifier)
    if not value or "\x00" in value:
        raise ValueError("SQLite identifier must be non-empty and contain no NUL.")
    return '"' + value.replace('"', '""') + '"'


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """Return whether a table exists in the current database."""
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (str(table_name),),
    ).fetchone()
    return row is not None


def check_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Verify that a target table exists by querying sqlite_master."""
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (str(table_name),),
    ).fetchone()
    return row is not None


def column_exists(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
) -> bool:
    """Return whether a column exists on a specific table."""
    if not table_exists(connection, table_name):
        return False

    table = quote_identifier(table_name)
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row[1]) == str(column_name) for row in rows)


def index_exists(connection: sqlite3.Connection, index_name: str) -> bool:
    """Return whether an index exists in the current database."""
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'index' AND name = ?
        LIMIT 1
        """,
        (str(index_name),),
    ).fetchone()
    return row is not None


def get_user_version(connection: sqlite3.Connection) -> int:
    """Return the current SQLite PRAGMA user_version."""
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def set_user_version(connection: sqlite3.Connection, version: int) -> None:
    """Set the SQLite PRAGMA user_version using a trusted integer."""
    value = int(version)
    if value < 0:
        raise ValueError("Schema version cannot be negative.")
    connection.execute(f"PRAGMA user_version = {value}")


def get_migration_status(
    db_path: str | Path,
    migrations_dict: Mapping[int, Migration],
) -> dict[str, int | list[int]]:
    """Inspect migration status without modifying the database."""
    path = Path(db_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Database file does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Database path is not a file: {path}")

    versions = sorted(int(version) for version in migrations_dict)
    if any(version <= 0 for version in versions):
        raise ValueError("Migration versions must be positive integers.")

    target_version = versions[-1] if versions else 0
    expected = list(range(1, target_version + 1))
    if versions != expected:
        missing = sorted(set(expected).difference(versions))
        raise ValueError(
            "Migration definitions are missing for versions: "
            + ", ".join(map(str, missing))
        )

    database_uri = f"{path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        current_version = get_user_version(connection)

    if current_version > target_version:
        raise RuntimeError(
            f"Database schema version {current_version} is newer than "
            f"supported version {target_version}."
        )

    pending = [version for version in versions if version > current_version]
    return {
        "current_version": current_version,
        "target_version": target_version,
        "pending_migrations": pending,
    }


@contextmanager
def migration_transaction(connection: sqlite3.Connection):
    """Execute migrations inside a rollback-safe SQLite savepoint."""
    savepoint = "schema_migration"
    connection.execute(f"SAVEPOINT {savepoint}")
    try:
        yield
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
    except Exception:
        connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise


from datetime import datetime, timezone


def ensure_migration_history_table(connection: sqlite3.Connection) -> None:
    """Ensure that the migration_history table exists."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS migration_history (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT
        )
    """)


def record_migration_applied(
    connection: sqlite3.Connection, version: int, description: str = ""
) -> None:
    """Record an applied migration in migration_history table."""
    ensure_migration_history_table(connection)
    now_iso = datetime.now(timezone.utc).isoformat()
    connection.execute(
        """
        INSERT OR REPLACE INTO migration_history (version, applied_at, description)
        VALUES (?, ?, ?)
        """,
        (version, now_iso, description),
    )


def record_migration_rolled_back(connection: sqlite3.Connection, version: int) -> None:
    """Remove a rolled-back migration from migration_history table."""
    ensure_migration_history_table(connection)
    connection.execute(
        "DELETE FROM migration_history WHERE version = ?",
        (version,),
    )


def get_latest_applied_migration(connection: sqlite3.Connection) -> int:
    """Get the latest applied migration version from migration_history table (or PRAGMA user_version)."""
    ensure_migration_history_table(connection)
    cursor = connection.execute("SELECT MAX(version) FROM migration_history")
    row = cursor.fetchone()
    if row and row[0] is not None:
        return int(row[0])

    user_ver = get_user_version(connection)
    if user_ver > 0:
        now_iso = datetime.now(timezone.utc).isoformat()
        for v in range(1, user_ver + 1):
            connection.execute(
                """
                INSERT OR IGNORE INTO migration_history (version, applied_at, description)
                VALUES (?, ?, ?)
                """,
                (v, now_iso, f"v{v}"),
            )
        return user_ver
    return 0


def run_migrations(
    connection: sqlite3.Connection,
    *,
    migrations: Mapping[int, Migration],
    target_version: int,
) -> int:
    """Apply every missing migration sequentially and atomically."""
    new_ver = int(target_version)
    old_ver = get_user_version(connection)

    if old_ver > new_ver:
        raise RuntimeError(
            f"Database schema version {old_ver} is newer than supported version {new_ver}."
        )

    expected_versions = set(range(1, new_ver + 1))
    missing_definitions = sorted(expected_versions.difference(migrations))
    if missing_definitions:
        raise RuntimeError(
            "Migration definitions are missing for versions: "
            + ", ".join(map(str, missing_definitions))
        )

    if old_ver == new_ver:
        ensure_migration_history_table(connection)
        return old_ver

    with migration_transaction(connection):
        ensure_migration_history_table(connection)
        for version in range(old_ver + 1, new_ver + 1):
            migration_fn = migrations[version]
            migration_name = getattr(migration_fn, "__name__", f"v{version}")
            start_time = time.perf_counter()
            migration_fn(connection)
            elapsed_sec = time.perf_counter() - start_time
            record_migration_applied(connection, version, migration_name)
            logger.info(
                "Migration [%s] executed in %.3f seconds.",
                migration_name,
                elapsed_sec,
            )
        set_user_version(connection, new_ver)

    logger.info(
        "Database migration from version %d to %d completed successfully.",
        old_ver,
        new_ver,
    )

    return new_ver


def rollback_migration(
    conn: sqlite3.Connection,
    target_version: int,
    *,
    down_migrations: Mapping[int, Migration],
) -> int:
    """Roll back the schema to ``target_version``.

    Executes the registered down-migration DDL script for each schema
    version being undone — from the current version down to
    ``target_version`` — in reverse order, atomically, then restores the
    schema version (SQLite's ``PRAGMA user_version``) to ``target_version``.

    Mirrors :func:`run_migrations`'s design: callers supply a mapping of
    version -> down-migration callable, where ``down_migrations[v]`` must
    undo whatever forward migration ``v`` did (bringing the schema from
    version ``v`` back to ``v - 1``).

    Args:
        conn: Open SQLite connection.
        target_version: Schema version to roll back to. Must be less than
            or equal to the database's current version.
        down_migrations: Mapping of version -> down-migration callable.

    Returns:
        The schema version after rollback (equal to ``target_version``).

    Raises:
        ValueError: If ``target_version`` is negative.
        RuntimeError: If ``target_version`` is greater than the database's
            current version, or a required down-migration definition is
            missing for one of the versions being undone.
    """
    new_ver = int(target_version)
    if new_ver < 0:
        raise ValueError("Schema version cannot be negative.")

    old_ver = get_user_version(conn)

    if new_ver > old_ver:
        raise RuntimeError(
            f"Cannot roll back to version {new_ver}: current schema version "
            f"{old_ver} is already older than the requested target."
        )

    if old_ver == new_ver:
        ensure_migration_history_table(conn)
        return old_ver

    versions_to_undo = range(old_ver, new_ver, -1)
    missing_definitions = sorted(
        v for v in versions_to_undo if v not in down_migrations
    )
    if missing_definitions:
        raise RuntimeError(
            "Down-migration definitions are missing for versions: "
            + ", ".join(map(str, missing_definitions))
        )

    with migration_transaction(conn):
        ensure_migration_history_table(conn)
        for version in versions_to_undo:
            down_fn = down_migrations[version]
            migration_name = getattr(down_fn, "__name__", f"v{version}_down")
            start_time = time.perf_counter()
            down_fn(conn)
            elapsed_sec = time.perf_counter() - start_time
            record_migration_rolled_back(conn, version)
            logger.info(
                "Rollback migration [%s] executed in %.3f seconds.",
                migration_name,
                elapsed_sec,
            )
        set_user_version(conn, new_ver)

    logger.info(
        "Database schema rolled back from version %d to %d successfully.",
        old_ver,
        new_ver,
    )

    return new_ver


def delete_all_if_table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    """Delete every row when the optional table exists."""
    if not table_exists(connection, table_name):
        return False

    table = quote_identifier(table_name)
    connection.execute(f"DELETE FROM {table}")
    return True


def enable_wal_mode(conn: sqlite3.Connection) -> str:
    """Enable Write-Ahead Logging (WAL) mode and NORMAL synchronous mode."""
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL;")
        journal_mode_result = cursor.fetchone()
        journal_mode = str(journal_mode_result[0]) if journal_mode_result else "unknown"

        cursor.execute("PRAGMA synchronous=NORMAL;")

        logger.info(
            f"SQLite WAL mode enabled. Journal mode: {journal_mode}, Synchronous: NORMAL"
        )
        return journal_mode
    except sqlite3.Error as e:
        logger.error(f"Failed to enable WAL mode: {e}")
        raise


def get_journal_mode(conn: sqlite3.Connection) -> str:
    """Retrieve the current SQLite journal mode."""
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA journal_mode;")
        result = cursor.fetchone()
        return str(result[0]) if result else "unknown"
    except sqlite3.Error as e:
        logger.error(f"Failed to get journal mode: {e}")
        return "unknown"


def perform_wal_checkpoint(conn: sqlite3.Connection, mode: str = "PASSIVE") -> dict:
    """Perform a Write-Ahead Log (WAL) checkpoint."""
    valid_modes = {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}
    if mode.upper() not in valid_modes:
        raise ValueError(f"Invalid checkpoint mode. Must be one of {valid_modes}")

    cursor = conn.cursor()
    try:
        cursor.execute(f"PRAGMA wal_checkpoint({mode.upper()});")
        result = cursor.fetchone()
        logger.info(f"WAL checkpoint ({mode}) performed. Result: {result}")
        return {"mode": mode.upper(), "result": result}
    except sqlite3.Error as e:
        logger.error(f"Failed to perform WAL checkpoint: {e}")
        raise


def verify_schema_integrity(db_path: Path, expected_tables: List[str]) -> bool:
    """Verify that the active database schema matches the expected table definitions.

    This function inspects the `sqlite_master` table of the specified SQLite database
    and compares the user-defined table names against a list of expected tables.
    It is designed to be used by administrators deploying updates to ensure that
    migrations have been applied correctly and no tables are missing.

    Args:
        db_path: Path to the SQLite database file to inspect.
        expected_tables: List of table names that must exist in the database.
                         Internal SQLite tables (prefixed with 'sqlite_') are
                         automatically excluded from the comparison.

    Returns:
        True if all expected tables exist in the database and no unexpected
        user-defined tables are present. False otherwise.

    Raises:
        FileNotFoundError: If the specified database file does not exist.
        sqlite3.DatabaseError: If the file is not a valid SQLite database.

    Examples:
        >>> from pathlib import Path
        >>> expected = ["documents", "chunks", "plagiarism_incidents"]
        >>> verify_schema_integrity(Path("data/corpus.db"), expected)
        True
    """
    import sqlite3

    # Validate input path
    resolved_path = Path(db_path).expanduser().resolve()

    if not resolved_path.exists():
        logger.error(
            "verify_schema_integrity: database file does not exist: %s",
            resolved_path,
        )
        raise FileNotFoundError(f"Database file not found: {resolved_path}")

    if not resolved_path.is_file():
        logger.error(
            "verify_schema_integrity: path is not a file: %s",
            resolved_path,
        )
        raise IsADirectoryError(f"Database path is not a file: {resolved_path}")

    # Normalize expected tables to lowercase for case-insensitive comparison
    expected_set = {t.lower().strip() for t in expected_tables if t.strip()}

    actual_tables = set()

    try:
        # Connect in read-only mode to prevent accidental modifications
        uri = f"file:{resolved_path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True, check_same_thread=False) as conn:
            # Query sqlite_master for all user-defined tables
            # Exclude internal SQLite tables (sqlite_sequence, etc.) and views
            cursor = conn.execute("""
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """)

            for row in cursor.fetchall():
                table_name = row[0].lower().strip()
                actual_tables.add(table_name)

    except sqlite3.DatabaseError as exc:
        logger.error(
            "verify_schema_integrity: failed to read database schema: %s",
            exc,
        )
        raise

    # Compare actual tables against expected tables
    missing_tables = expected_set - actual_tables
    unexpected_tables = actual_tables - expected_set

    is_valid = True

    if missing_tables:
        logger.error(
            "verify_schema_integrity: MISSING tables in %s: %s",
            resolved_path,
            ", ".join(sorted(missing_tables)),
        )
        is_valid = False

    if unexpected_tables:
        logger.warning(
            "verify_schema_integrity: UNEXPECTED tables in %s: %s",
            resolved_path,
            ", ".join(sorted(unexpected_tables)),
        )
        # Unexpected tables might be acceptable in some scenarios (e.g., legacy tables),
        # but for strict integrity verification, we flag them as invalid.
        is_valid = False

    if is_valid:
        logger.info(
            "verify_schema_integrity: schema verification PASSED for %s (%d tables verified).",
            resolved_path,
            len(actual_tables),
        )
    else:
        logger.error(
            "verify_schema_integrity: schema verification FAILED for %s.",
            resolved_path,
        )

    return is_valid
