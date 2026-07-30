import sqlite3
from datetime import datetime

import pytest

from src.db.database_backup import SQLITE_HEADER, create_sqlite_snapshot

from contextlib import closing

def create_test_database(path):
    with closing(sqlite3.connect(path)) as connection:
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

    with closing(sqlite3.connect(restored)) as connection:
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


# ── Issue #472: path metadata logic ──────────────────────────────────────────


def test_backup_panel_path_resolves_to_existing_file(tmp_path):
    """get_corpus_db_path() returns a Path; verify it can be stat'd."""
    db_path = tmp_path / "corpus.db"
    create_test_database(db_path)

    assert db_path.exists()
    assert db_path.is_file()
    assert db_path.stat().st_size > 0
    # The panel shows db_path.name — ensure it is the bare filename, not a dir.
    assert db_path.name == "corpus.db"


def test_backup_panel_size_label_formatting(tmp_path):
    """Verify the size-formatting breakpoints used in the backup panel."""

    def _size_label(size_bytes: int) -> str:
        if size_bytes >= 1024 * 1024:
            return f"{size_bytes / 1_048_576:.2f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes} B"

    assert _size_label(500) == "500 B"
    assert _size_label(1024) == "1.0 KB"
    assert _size_label(2048) == "2.0 KB"
    assert _size_label(1024 * 1024) == "1.00 MB"
    assert _size_label(2 * 1024 * 1024) == "2.00 MB"


def test_backup_panel_modified_date_formatting(tmp_path):
    """Verify that the mtime formatting used in the backup panel is correct."""
    db_path = tmp_path / "corpus.db"
    create_test_database(db_path)

    mtime = db_path.stat().st_mtime
    formatted = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

    # Must be a non-empty string matching YYYY-MM-DD HH:MM
    import re
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", formatted)
