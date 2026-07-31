"""
Shared SQLite schema migration helpers.

This module provides robust, atomic, and rollback-safe utilities for managing
SQLite database schema migrations, including journal mode optimization.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import contextmanager

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

Migration: TypeAlias = Callable[[sqlite3.Connection], None]


def quote_identifier(identifier: str) -> str:
    """
    Return a safely quoted SQLite identifier to prevent SQL injection.
    
    Args:
        identifier: The database object name (table, column, index).
        
    Returns:
        str: The safely quoted identifier.
        
    Raises:
        ValueError: If the identifier is empty or contains NUL bytes.
    """
    value = str(identifier)
    if not value or "\x00" in value:
        raise ValueError("SQLite identifier must be non-empty and contain no NUL.")
    return '"' + value.replace('"', '""') + '"'


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """
    Return whether a table exists in the current database.
    
    Args:
        connection: An active sqlite3.Connection object.
        table_name: The name of the table to check.
        
    Returns:
        bool: True if the table exists, False otherwise.
    """
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
    """
    Return whether a column exists on a specific table.
    
    Args:
        connection: An active sqlite3.Connection object.
        table_name: The name of the table.
        column_name: The name of the column to check.
        
    Returns:
        bool: True if the column exists, False otherwise.
    """
    if not table_exists(connection, table_name):
        return False

    table = quote_identifier(table_name)
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row[1]) == str(column_name) for row in rows)


def index_exists(connection: sqlite3.Connection, index_name: str) -> bool:
    """
    Return whether an index exists in the current database.
    
    Args:
        connection: An active sqlite3.Connection object.
        index_name: The name of the index to check.
        
    Returns:
        bool: True if the index exists, False otherwise.
    """
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
    """
    Return the current SQLite PRAGMA user_version.
    
    Args:
        connection: An active sqlite3.Connection object.
        
    Returns:
        int: The current schema version.
    """
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def set_user_version(connection: sqlite3.Connection, version: int) -> None:
    """
    Set the SQLite PRAGMA user_version using a trusted integer.
    
    Args:
        connection: An active sqlite3.Connection object.
        version: The new schema version to set.
        
    Raises:
        ValueError: If the version is negative.
    """
    value = int(version)
    if value < 0:
        raise ValueError("Schema version cannot be negative.")
    connection.execute(f"PRAGMA user_version = {value}")


@contextmanager
def migration_transaction(connection: sqlite3.Connection):
    """
    Execute migrations inside a rollback-safe SQLite savepoint.
    
    This ensures that if any migration step fails, all schema and data 
    changes, as well as the PRAGMA user_version update, are rolled back.
    
    Args:
        connection: An active sqlite3.Connection object.
    """
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
    """
    Apply every missing migration sequentially and atomically.
    
    Args:
        connection: An active sqlite3.Connection object.
        migrations: A mapping of version numbers to migration callables.
        target_version: The desired final schema version.
        
    Returns:
        int: The final schema version after successful migration.
        
    Raises:
        RuntimeError: If the database is newer than the target, or if 
                      migration definitions are missing.
    """
    target = int(target_version)
    current = get_user_version(connection)

    if current > target:
        raise RuntimeError(
            f"Database schema version {current} is newer than supported "
            f"version {target}."
        )

    expected_versions = set(range(1, target + 1))
    missing_definitions = sorted(expected_versions.difference(migrations))
    if missing_definitions:
        raise RuntimeError(
            "Migration definitions are missing for versions: "
            + ", ".join(map(str, missing_definitions))
        )

    with migration_transaction(connection):
        for version in range(current + 1, target + 1):
            migrations[version](connection)
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
    """
    Delete every row when the optional table exists.
    
    Args:
        connection: An active sqlite3.Connection object.
        table_name: The name of the table to clear.
        
    Returns:
        bool: True if rows were deleted, False if the table did not exist.
    """
    if not table_exists(connection, table_name):
        return False

    table = quote_identifier(table_name)
    connection.execute(f"DELETE FROM {table}")
    return True


def enable_wal_mode(conn: sqlite3.Connection) -> str:
    """
    Enable Write-Ahead Logging (WAL) mode and NORMAL synchronous mode.
    
    This improves concurrent read/write performance by allowing readers 
    to proceed without blocking writers, and vice versa.
    
    Args:
        conn: An active sqlite3.Connection object.
        
    Returns:
        str: The resulting journal mode (should be 'wal').
        
    Raises:
        sqlite3.Error: If the PRAGMA commands fail to execute.
    """
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
    """
    Retrieve the current SQLite journal mode.
    
    Args:
        conn: An active sqlite3.Connection object.
        
    Returns:
        str: The current journal mode (e.g., 'delete', 'wal', 'memory').
    """
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA journal_mode;")
        result = cursor.fetchone()
        return str(result[0]) if result else "unknown"
    except sqlite3.Error as e:
        logger.error(f"Failed to get journal mode: {e}")
        return "unknown"


def perform_wal_checkpoint(conn: sqlite3.Connection, mode: str = "PASSIVE") -> dict:
    """
    Perform a Write-Ahead Log (WAL) checkpoint.
    
    Args:
        conn: An active sqlite3.Connection object.
        mode: Checkpoint mode ('PASSIVE', 'FULL', 'RESTART', 'TRUNCATE').
        
    Returns:
        dict: A dictionary containing the mode and the result tuple from SQLite.
        
    Raises:
        ValueError: If an invalid checkpoint mode is provided.
        sqlite3.Error: If the checkpoint fails to execute.
    """
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
