import sqlite3

import pytest

from src.db.migrations.common import get_migration_status


def migration(connection):
    connection.execute("SELECT 1")


MIGRATIONS = {1: migration, 2: migration, 3: migration}


def create_database(path, version):
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {version}")


@pytest.mark.parametrize(
    ("current", "pending"),
    [
        (0, [1, 2, 3]),
        (1, [2, 3]),
        (2, [3]),
        (3, []),
    ],
)
def test_status_reports_versions_and_pending(tmp_path, current, pending):
    database = tmp_path / "test.db"
    create_database(database, current)

    assert get_migration_status(database, MIGRATIONS) == {
        "current_version": current,
        "target_version": 3,
        "pending_migrations": pending,
    }


def test_empty_migration_mapping_has_zero_target(tmp_path):
    database = tmp_path / "test.db"
    create_database(database, 0)

    assert get_migration_status(database, {}) == {
        "current_version": 0,
        "target_version": 0,
        "pending_migrations": [],
    }


def test_missing_database_is_not_created(tmp_path):
    database = tmp_path / "missing.db"

    with pytest.raises(FileNotFoundError):
        get_migration_status(database, MIGRATIONS)

    assert not database.exists()


def test_directory_path_is_rejected(tmp_path):
    with pytest.raises(IsADirectoryError):
        get_migration_status(tmp_path, MIGRATIONS)


def test_incomplete_migration_map_is_rejected(tmp_path):
    database = tmp_path / "test.db"
    create_database(database, 0)

    with pytest.raises(ValueError, match="missing.*2"):
        get_migration_status(
            database,
            {1: migration, 3: migration},
        )


def test_newer_database_is_rejected(tmp_path):
    database = tmp_path / "test.db"
    create_database(database, 4)

    with pytest.raises(RuntimeError, match="newer than supported"):
        get_migration_status(database, MIGRATIONS)


def test_invalid_sqlite_file_is_rejected(tmp_path):
    database = tmp_path / "invalid.db"
    database.write_text("not sqlite", encoding="utf-8")

    with pytest.raises(sqlite3.DatabaseError):
        get_migration_status(database, MIGRATIONS)
