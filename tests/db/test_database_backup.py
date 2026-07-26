import sqlite3

import pytest

from src.db.database_backup import SQLITE_HEADER, create_sqlite_snapshot


def create_test_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO documents (filename) VALUES (?)",
            [("alpha.pdf",), ("beta.pdf",)],
        )
        connection.commit()


def test_snapshot_is_valid_and_preserves_data(tmp_path):
    source = tmp_path / "corpus.db"
    create_test_database(source)

    snapshot = create_sqlite_snapshot(source)

    assert snapshot.startswith(SQLITE_HEADER)

    restored = tmp_path / "restored.db"
    restored.write_bytes(snapshot)

    with sqlite3.connect(restored) as connection:
        rows = connection.execute(
            "SELECT filename FROM documents ORDER BY id"
        ).fetchall()

    assert rows == [("alpha.pdf",), ("beta.pdf",)]


def test_snapshot_does_not_modify_source_database(tmp_path):
    source = tmp_path / "corpus.db"
    create_test_database(source)
    before = source.read_bytes()

    create_sqlite_snapshot(source)

    assert source.read_bytes() == before


def test_missing_database_raises_file_not_found(tmp_path):
    missing = tmp_path / "missing.db"

    with pytest.raises(FileNotFoundError):
        create_sqlite_snapshot(missing)


def test_directory_path_is_rejected(tmp_path):
    with pytest.raises(IsADirectoryError):
        create_sqlite_snapshot(tmp_path)


def test_non_sqlite_file_is_rejected(tmp_path):
    invalid = tmp_path / "invalid.db"
    invalid.write_text("not sqlite", encoding="utf-8")

    with pytest.raises(sqlite3.DatabaseError):
        create_sqlite_snapshot(invalid)
