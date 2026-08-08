import sqlite3
from pathlib import Path

import pytest

from src.db.common import get_read_connection


def create_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO documents (title)
            VALUES (?)
            """,
            ("Reference document",),
        )
        connection.commit()


def test_get_read_connection_allows_select_queries(tmp_path):
    database = tmp_path / "corpus.db"
    create_database(database)

    with get_read_connection(database) as connection:
        row = connection.execute(
            """
            SELECT id, title
            FROM documents
            """
        ).fetchone()

    assert isinstance(row, sqlite3.Row)
    assert row["id"] == 1
    assert row["title"] == "Reference document"


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO documents (title) VALUES ('new')",
        "UPDATE documents SET title = 'changed' WHERE id = 1",
        "DELETE FROM documents WHERE id = 1",
        "CREATE TABLE forbidden (id INTEGER)",
        "DROP TABLE documents",
    ],
)
def test_get_read_connection_rejects_write_attempts(
    tmp_path,
    statement,
):
    database = tmp_path / "corpus.db"
    create_database(database)

    with get_read_connection(database) as connection:
        with pytest.raises(
            sqlite3.OperationalError,
            match="readonly|read-only",
        ):
            connection.execute(statement)


def test_get_read_connection_does_not_create_missing_file(
    tmp_path,
):
    missing_database = tmp_path / "missing.db"

    with pytest.raises(FileNotFoundError):
        get_read_connection(missing_database)

    assert missing_database.exists() is False


def test_get_read_connection_rejects_directory(tmp_path):
    with pytest.raises(IsADirectoryError):
        get_read_connection(tmp_path)


def test_get_read_connection_requires_path_object(tmp_path):
    database = tmp_path / "corpus.db"
    create_database(database)

    with pytest.raises(
        TypeError,
        match="pathlib.Path",
    ):
        get_read_connection(str(database))


def test_get_read_connection_handles_spaces_in_path(tmp_path):
    directory = tmp_path / "database directory"
    directory.mkdir()
    database = directory / "read replica.db"
    create_database(database)

    with get_read_connection(database) as connection:
        title = connection.execute(
            "SELECT title FROM documents"
        ).fetchone()["title"]

    assert title == "Reference document"


def test_get_read_connection_enables_foreign_keys(tmp_path):
    database = tmp_path / "corpus.db"
    create_database(database)

    with get_read_connection(database) as connection:
        enabled = connection.execute(
            "PRAGMA foreign_keys"
        ).fetchone()[0]

    assert enabled == 1
