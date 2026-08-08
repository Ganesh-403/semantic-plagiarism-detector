"""Shared SQLite connection helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def get_read_connection(
    db_path: Path,
) -> sqlite3.Connection:
    """Open an existing SQLite database in read-only mode.

    The database path is converted to a platform-safe ``file:`` URI
    and opened with ``mode=ro``. SQLite therefore refuses
    ``INSERT``, ``UPDATE``, ``DELETE``, schema changes, and attempts
    to create a database that does not already exist.

    Args:
        db_path: Path to an existing SQLite database file.

    Returns:
        A read-only SQLite connection configured with
        :class:`sqlite3.Row` as its row factory.

    Raises:
        TypeError: If ``db_path`` is not a :class:`pathlib.Path`.
        FileNotFoundError: If the database file does not exist.
        IsADirectoryError: If ``db_path`` points to a directory.
        sqlite3.Error: If SQLite cannot open the file.
    """
    if not isinstance(db_path, Path):
        raise TypeError("db_path must be a pathlib.Path.")

    resolved_path = db_path.expanduser().resolve(strict=False)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"SQLite database does not exist: {resolved_path}"
        )
    if not resolved_path.is_file():
        raise IsADirectoryError(
            f"SQLite database path is not a file: {resolved_path}"
        )

    # Path.as_uri() handles Windows drive letters, spaces, Unicode,
    # and other characters that require URI escaping.
    database_uri = f"{resolved_path.as_uri()}?mode=ro"
    connection = sqlite3.connect(
        database_uri,
        uri=True,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection
