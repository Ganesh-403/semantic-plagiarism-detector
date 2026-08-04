"""Regression tests for Issue #831 database optimization."""

import sqlite3
from pathlib import Path

from src.db.database_backup import optimize_database


def _create_fragmented_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA auto_vacuum = NONE")
        connection.execute(
            "CREATE TABLE payloads (id INTEGER PRIMARY KEY, payload BLOB)"
        )
        blob = b"x" * 8192
        connection.executemany(
            "INSERT INTO payloads (payload) VALUES (?)",
            [(blob,) for _ in range(600)],
        )
        connection.commit()
        connection.execute("DELETE FROM payloads WHERE id > 5")
        connection.commit()
    finally:
        connection.close()


def test_optimize_database_reclaims_deleted_pages(tmp_path):
    database_path = tmp_path / "fragmented.db"
    _create_fragmented_database(database_path)
    size_before = database_path.stat().st_size

    assert optimize_database(database_path) is True

    size_after = database_path.stat().st_size
    assert size_after < size_before

    with sqlite3.connect(database_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM payloads"
        ).fetchone()[0]
        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

    assert count == 5
    assert integrity == "ok"


def test_optimize_database_rejects_non_sqlite_file(tmp_path):
    invalid_path = tmp_path / "not-sqlite.db"
    invalid_path.write_text("not a sqlite database", encoding="utf-8")

    assert optimize_database(invalid_path) is False
    assert invalid_path.read_text(encoding="utf-8") == "not a sqlite database"


def test_optimize_database_rejects_directory(tmp_path):
    assert optimize_database(tmp_path) is False


def test_maintenance_connection_is_closed_after_success(tmp_path):
    database_path = tmp_path / "closed.db"
    _create_fragmented_database(database_path)

    assert optimize_database(database_path) is True

    # An exclusive transaction can be acquired immediately only when the
    # optimizer's connection has been closed.
    with sqlite3.connect(database_path, timeout=0.1) as connection:
        connection.execute("BEGIN EXCLUSIVE")
        connection.rollback()
