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

    pending = [
        version
        for version in versions
        if version > current_version
    ]
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


def run_migrations(
    connection: sqlite3.Connection,
    *,
    migrations: Mapping[int, Migration],
    target_version: int,
) -> int:
    """Apply every missing migration sequentially and atomically."""
    target = int(target_version)
    current = get_user_version(connection)

    if current > target:
        raise RuntimeError(
            f"Database schema version {current} is newer than supported version {target}."
        )

    expected_versions = set(range(1, target + 1))
    missing_definitions = sorted(expected_versions.difference(migrations))
    if missing_definitions:
        raise RuntimeError(
            "Migration definitions are missing for versions: "
            + ", ".join(map(str, missing_definitions))
        )

    if current == target:
        return current

    with migration_transaction(connection):
        for version in range(current + 1, target + 1):
            migration_fn = migrations[version]
            migration_name = getattr(migration_fn, "__name__", f"v{version}")
            start_time = time.perf_counter()
            migration_fn(connection)
            elapsed_sec = time.perf_counter() - start_time
            logger.info(
                "Migration [%s] executed in %.3f seconds.",
                migration_name,
                elapsed_sec,
            )
        set_user_version(connection, target)

    logger.info(
        "Database migration from version %d to %d completed successfully.",
        current,
        target,
    )

    return target


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

        logger.info(f"SQLite WAL mode enabled. Journal mode: {journal_mode}, Synchronous: NORMAL")
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
    