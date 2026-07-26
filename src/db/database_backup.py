"""Consistent SQLite database download helpers."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from src.db.corpus_db import get_corpus_db_path

SQLITE_HEADER = b"SQLite format 3\x00"


def create_sqlite_snapshot(database_path: str | Path) -> bytes:
    """Return a transactionally consistent SQLite snapshot.

    SQLite's online backup API is used instead of reading a live database
    file directly. This includes committed pages correctly even when the
    source database uses WAL journaling.
    """
    source_path = Path(database_path).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(
            f"SQLite database does not exist: {source_path}"
        )
    if not source_path.is_file():
        raise IsADirectoryError(
            f"SQLite database path is not a file: {source_path}"
        )

    with tempfile.TemporaryDirectory(
        prefix="semantic-plagiarism-backup-"
    ) as temporary_directory:
        snapshot_path = Path(temporary_directory) / source_path.name
        source_uri = f"{source_path.as_uri()}?mode=ro"

        with sqlite3.connect(
            source_uri,
            uri=True,
            check_same_thread=False,
        ) as source_connection:
            with sqlite3.connect(snapshot_path) as destination:
                source_connection.backup(destination)

        snapshot = snapshot_path.read_bytes()

    if not snapshot.startswith(SQLITE_HEADER):
        raise sqlite3.DatabaseError(
            "Generated backup is not a valid SQLite database."
        )

    return snapshot


def create_corpus_database_snapshot() -> bytes:
    """Return a downloadable snapshot of the configured corpus DB."""
    return create_sqlite_snapshot(get_corpus_db_path())
