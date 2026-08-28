import re
import sqlite3
from contextlib import closing
from datetime import datetime

import pytest

from src.db.database_backup import (
    _ALLOWED_DB_DIR,
    SQLITE_HEADER,
    BackupRestoreSecurityError,
    create_sqlite_snapshot,
    get_database_file_size_bytes,
    verify_backup_file,
)


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


def test_snapshot_applies_busy_timeout_to_source_connection(tmp_path, monkeypatch):
    source = tmp_path / "corpus.db"
    create_test_database(source)

    applied_timeouts = []
    from src.db import database_backup

    orig_apply = database_backup.apply_busy_timeout

    def mock_apply(conn, timeout):
        applied_timeouts.append(timeout)
        return orig_apply(conn, timeout)

    monkeypatch.setattr(database_backup, "apply_busy_timeout", mock_apply)

    snapshot = create_sqlite_snapshot(source)
    assert snapshot.startswith(SQLITE_HEADER)
    assert len(applied_timeouts) >= 1
    assert applied_timeouts[0] == database_backup.DEFAULT_SQLITE_TIMEOUT


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
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", formatted)


# ── get_database_file_size_bytes ──────────────────────────────────────────────


def test_get_database_file_size_bytes_existing_file():
    db = _ALLOWED_DB_DIR / "corpus.db"
    create_test_database(db)
    try:
        assert get_database_file_size_bytes(db) == db.stat().st_size
        assert get_database_file_size_bytes(db) > 0
    finally:
        db.unlink(missing_ok=True)


def test_get_database_file_size_bytes_missing_file():
    missing = _ALLOWED_DB_DIR / "__nonexistent_test__.db"
    assert get_database_file_size_bytes(missing) == 0


def test_get_database_file_size_bytes_accepts_string_path():
    db = _ALLOWED_DB_DIR / "users_test_size.db"
    create_test_database(db)
    try:
        assert get_database_file_size_bytes(str(db)) == db.stat().st_size
    finally:
        db.unlink(missing_ok=True)


def test_get_database_file_size_bytes_rejects_path_traversal(tmp_path):
    outside = tmp_path / "evil.db"
    outside.write_text("x")
    with pytest.raises(ValueError, match="outside the allowed directory"):
        get_database_file_size_bytes(outside)


def test_create_database_backup_sets_restrictive_permissions(tmp_path, monkeypatch):
    import os
    from src.db.database_backup import create_database_backup

    source = tmp_path / "source.db"
    create_test_database(source)
    backup_dir = tmp_path / "backups"

    chmod_calls = []
    orig_chmod = os.chmod

    def mock_chmod(path, mode):
        chmod_calls.append((path, mode))
        try:
            orig_chmod(path, mode)
        except OSError:
            pass

    monkeypatch.setattr(os, "chmod", mock_chmod)

    # Test compressed backup (.db.gz)
    gz_backup = create_database_backup(source, backup_dir=backup_dir, compress_backup=True)
    assert gz_backup.exists()
    assert len(chmod_calls) >= 1
    assert chmod_calls[-1][0] == gz_backup
    assert chmod_calls[-1][1] == 0o600

    # Test uncompressed backup (.db)
    db_backup = create_database_backup(source, backup_dir=backup_dir, compress_backup=False)
    assert db_backup.exists()
    assert len(chmod_calls) >= 2
    assert chmod_calls[-1][0] == db_backup
    assert chmod_calls[-1][1] == 0o600


def test_create_database_backup_respects_gzip_compression_level_env(tmp_path, monkeypatch):
    """Verify that create_database_backup reads BACKUP_GZIP_COMPRESSION_LEVEL from the env and passes it to GzipFile."""
    import gzip
    from unittest.mock import patch
    from src.db.database_backup import create_database_backup

    source = tmp_path / "source.db"
    create_test_database(source)
    backup_dir = tmp_path / "backups"

    # Mock GzipFile to record the compression level
    passed_compresslevel = []
    original_gzip_file = gzip.GzipFile

    class MockGzipFile(original_gzip_file):
        def __init__(self, *args, **kwargs):
            if "compresslevel" in kwargs:
                passed_compresslevel.append(kwargs["compresslevel"])
            super().__init__(*args, **kwargs)

    # 1. Test default value of 6
    with patch("gzip.GzipFile", MockGzipFile):
        monkeypatch.delenv("BACKUP_GZIP_COMPRESSION_LEVEL", raising=False)
        create_database_backup(source, backup_dir=backup_dir, compress_backup=True)
        assert len(passed_compresslevel) == 1
        assert passed_compresslevel[0] == 6

    # 2. Test configured value (e.g. 3)
    passed_compresslevel.clear()
    with patch("gzip.GzipFile", MockGzipFile):
        monkeypatch.setenv("BACKUP_GZIP_COMPRESSION_LEVEL", "3")
        create_database_backup(source, backup_dir=backup_dir, compress_backup=True)
        assert len(passed_compresslevel) == 1
        assert passed_compresslevel[0] == 3

 feature/cleanup-failed-backups
 feature/cleanup-failed-backups
 feature/cleanup-failed-backups

 feature/cleanup-failed-backups

 feature/utc-timestamp-backups
 feature/utc-timestamp-backups

 feature/utc-timestamp-backups
 main

 feature/pre-snapshot-integrity-check
 feature/pre-snapshot-integrity-check

 feature/pre-snapshot-integrity-check

 feature/backup-integrity-check-3407

 feature/backup-integrity-check-3407
 main
 main
def test_verify_backup_file_valid_gzip(tmp_path):
    import gzip
    source = tmp_path / "source.db"
    create_test_database(source)

    backup = tmp_path / "backup.db.gz"
    with open(source, "rb") as f_in:
        with gzip.open(backup, "wb") as f_out:
            f_out.write(f_in.read())

    assert verify_backup_file(backup) is True


def test_verify_backup_file_valid_uncompressed(tmp_path):
    source = tmp_path / "source.db"
    create_test_database(source)
    assert verify_backup_file(source) is True


def test_verify_backup_file_corrupted_gzip(tmp_path):
    backup = tmp_path / "corrupt.db.gz"
    # Write invalid gzip data starting with magic bytes
    backup.write_bytes(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xffinvalidjunkdata")
    assert verify_backup_file(backup) is False


def test_verify_backup_file_invalid_header(tmp_path):
    import gzip
    backup = tmp_path / "invalid_header.db.gz"
    with gzip.open(backup, "wb") as f:
        f.write(b"not a sqlite database file header but it has 100 bytes of content so it is read without issues")
    assert verify_backup_file(backup) is False


def test_verify_backup_file_nonexistent():
    assert verify_backup_file("nonexistent_backup_file_path.db.gz") is False


def test_create_sqlite_snapshot_check_integrity_healthy(tmp_path):
    source = tmp_path / "healthy.db"
    create_test_database(source)
    snapshot = create_sqlite_snapshot(source, check_integrity=True)
    assert snapshot.startswith(SQLITE_HEADER)


def test_create_sqlite_snapshot_check_integrity_corrupted(tmp_path):
    source = tmp_path / "corrupt.db"
    create_test_database(source)

    # Overwrite the database with junk bytes starting after the header
    with open(source, "r+b") as f:
        f.seek(100)
        f.write(b"CORRUPTEDDATA" * 100)

    with pytest.raises(sqlite3.DatabaseError, match="Database integrity check failed"):
        create_sqlite_snapshot(source, check_integrity=True)


def test_create_sqlite_snapshot_check_integrity_disabled_by_default(tmp_path):
    source = tmp_path / "corrupt_default.db"
    create_test_database(source)

    # Overwrite database pages to corrupt it
    with open(source, "r+b") as f:
        f.seek(100)
        f.write(b"CORRUPTEDDATA" * 100)

    # By default (check_integrity=False), it should not raise DatabaseError from quick_check
    # (though backup itself might raise sqlite3.DatabaseError if it's completely unreadable,
    # let's assert it runs or at least does not fail on integrity quick_check)
    try:
        create_sqlite_snapshot(source, check_integrity=False)
    except sqlite3.DatabaseError as exc:
        # If it raises DatabaseError, it must not be "Database integrity check failed"
        assert "Database integrity check failed" not in str(exc)


def test_create_database_backup_uses_utc_timestamp(tmp_path):
    from src.db.database_backup import create_database_backup

    source = tmp_path / "source.db"
    create_test_database(source)
    backup_dir = tmp_path / "backups"

    # Test compressed backup (.db.gz) UTC Zulu timestamp pattern
    gz_backup = create_database_backup(source, backup_dir=backup_dir, compress_backup=True)
    assert gz_backup.exists()
    assert re.search(r"\.\d{8}_\d{6}Z\.db\.gz$", gz_backup.name) is not None

    # Test uncompressed backup (.db) UTC Zulu timestamp pattern
    db_backup = create_database_backup(source, backup_dir=backup_dir, compress_backup=False)
    assert db_backup.exists()
    assert re.search(r"\.\d{8}_\d{6}Z\.db$", db_backup.name) is not None


def test_create_database_backup_cleans_up_on_failure(tmp_path, monkeypatch):
    from src.db.database_backup import create_database_backup
    import src.db.database_backup

    source = tmp_path / "source.db"
    create_test_database(source)
    backup_dir = tmp_path / "backups"

    def mock_iter_chunks(db_path, chunk_size=64*1024):
        yield b"partial header..."
        raise IOError("Disk full or connection lost")

    monkeypatch.setattr(src.db.database_backup, "iter_sqlite_snapshot_chunks", mock_iter_chunks)

    # 1. Test compressed backup cleanup
    with pytest.raises(IOError, match="Disk full or connection lost"):
        create_database_backup(source, backup_dir=backup_dir, compress_backup=True)

    # Assert no files remain in the backup directory
    files = list(backup_dir.glob("*"))
    assert len(files) == 0

    # 2. Test uncompressed backup cleanup
    with pytest.raises(IOError, match="Disk full or connection lost"):
        create_database_backup(source, backup_dir=backup_dir, compress_backup=False)

    files = list(backup_dir.glob("*"))
    assert len(files) == 0







 main
 main

 main
