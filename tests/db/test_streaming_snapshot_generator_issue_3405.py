"""Unit tests for streaming SQLite snapshot generator and streaming backup endpoint (Issue #3405)."""

import os
import sqlite3
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.middleware import verify_bearer_token
from src.api.dependencies import get_current_user
from src.db.database_backup import (
    SQLITE_HEADER,
    create_sqlite_snapshot,
    iter_corpus_database_snapshot_chunks,
    iter_sqlite_snapshot_chunks,
)

client = TestClient(app)


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary populated SQLite database."""
    db_file = tmp_path / "test_snapshot.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.executemany(
            "INSERT INTO users (name) VALUES (?)",
            [(f"user_{i}",) for i in range(100)],
        )
        conn.commit()
    return db_file


def test_iter_sqlite_snapshot_chunks_yields_valid_chunks(temp_db):
    """Verify iter_sqlite_snapshot_chunks yields chunks and preserves SQLite validity."""
    chunks = list(iter_sqlite_snapshot_chunks(temp_db, chunk_size=512))
    assert len(chunks) > 0
    snapshot_bytes = b"".join(chunks)

    assert snapshot_bytes.startswith(SQLITE_HEADER)

    # Verify snapshot database can be queried
    with sqlite3.connect(":memory:") as memory_conn:
        temp_file = temp_db.parent / "restored.db"
        temp_file.write_bytes(snapshot_bytes)
        with sqlite3.connect(temp_file) as restored_conn:
            cursor = restored_conn.cursor()
            count = cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            assert count == 100


def test_create_sqlite_snapshot_matches_iter_chunks(temp_db):
    """Verify create_sqlite_snapshot matches b''.join(iter_sqlite_snapshot_chunks)."""
    snapshot = create_sqlite_snapshot(temp_db)
    iter_snapshot = b"".join(iter_sqlite_snapshot_chunks(temp_db))
    assert snapshot == iter_snapshot


def test_iter_sqlite_snapshot_chunks_nonexistent_raises_file_not_found(tmp_path):
    """Verify missing database path raises FileNotFoundError."""
    missing = tmp_path / "nonexistent.db"
    with pytest.raises(FileNotFoundError):
        list(iter_sqlite_snapshot_chunks(missing))


def test_iter_sqlite_snapshot_chunks_directory_raises_is_a_directory(tmp_path):
    """Verify directory path raises IsADirectoryError."""
    with pytest.raises(IsADirectoryError):
        list(iter_sqlite_snapshot_chunks(tmp_path))


def test_api_download_backup_endpoint_streams_snapshot(tmp_path):
    """Verify /api/v1/backup/download streams SQLite database snapshot response."""
    app.dependency_overrides[verify_bearer_token] = lambda: "test-token"
    app.dependency_overrides[get_current_user] = lambda: {"username": "admin", "scopes": ["admin"]}
    try:
        res = client.get("/api/v1/backup/download")
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/x-sqlite3"
        assert "attachment" in res.headers["content-disposition"]
        assert res.content.startswith(SQLITE_HEADER)
    finally:
        app.dependency_overrides.clear()
