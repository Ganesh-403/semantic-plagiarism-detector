"""Shared SQLite schema migration helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from contextlib import contextmanager
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

Migration: TypeAlias = Callable[[sqlite3.Connection], None]


def quote_identifier(identifier: str) -> str:
    """Return a safely quoted SQLite identifier."""
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
    """Return whether a column exists on a table."""
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
    """Return SQLite PRAGMA user_version."""
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def set_user_version(connection: sqlite3.Connection, version: int) -> None:
    """Set SQLite PRAGMA user_version using a trusted integer."""
    value = int(version)
    if value < 0:
        raise ValueError("Schema version cannot be negative.")
    connection.execute(f"PRAGMA user_version = {value}")


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
    """Apply every missing migration sequentially and atomically.

    The complete upgrade is wrapped in one savepoint. If any migration fails,
    all schema/data changes and the PRAGMA user_version update are rolled back.
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

    if current == target:
        return current

    with migration_transaction(connection):
        for version in range(current + 1, target + 1):
            migrations[version](connection)
        set_user_version(connection, target)

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
