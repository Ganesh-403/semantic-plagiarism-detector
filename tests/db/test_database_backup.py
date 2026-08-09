"""
test_database_backup.py
-----------------------
Unit tests for the database backup, restoration, and retention management module.

This module validates:
- SQLite snapshot creation (including corpus DB snapshots).
- Password-protected ZIP backups.
- Secure database restore workflow.
- Database optimization via VACUUM and PRAGMA optimize.
- Automated cleanup of old backups based on retention policies.
- Database file size inspection (issue #1047).
"""

import gzip
import logging
import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.db.database_backup import (
    SQLITE_HEADER,
    BackupRestoreSecurityError,
    cleanup_old_backups,
    create_database_backup,
    create_password_protected_backup,
    create_sqlite_snapshot,
    get_database_size_bytes,
    get_database_table_stats,
    get_table_schema_info,
    optimize_database,
    restore,
    run_incremental_vacuum,
    checkpoint_wal_log,
)

try:
    import pyzipper  # noqa: F401
    HAS_PYZIPPER = True
except ImportError:
    HAS_PYZIPPER = False


class TestCreateSqliteSnapshot:
    """Tests for the create_sqlite_snapshot function."""

    def test_create_snapshot_valid_db(self, tmp_path):
        """Verify that a valid SQLite database produces a correct snapshot."""
        db_path = tmp_path / "test.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO test (name) VALUES ('sample')")
        conn.commit()
        conn.close()

        snapshot = create_sqlite_snapshot(db_path)

        assert snapshot.startswith(
            SQLITE_HEADER
        ), "Snapshot must start with valid SQLite header"
        assert len(snapshot) > 1000, "Snapshot should have reasonable size"

    def test_create_snapshot_file_not_found(self):
        """Verify that a FileNotFoundError is raised for non-existent paths."""
        with pytest.raises(FileNotFoundError, match="SQLite database does not exist"):
            create_sqlite_snapshot("/nonexistent/path/to/db.db")

    def test_create_snapshot_is_a_directory(self, tmp_path):
        """Verify that an IsADirectoryError is raised if path is a directory."""
        with pytest.raises(
            IsADirectoryError, match="SQLite database path is not a file"
        ):
            create_sqlite_snapshot(tmp_path)


class TestCorpusSnapshotAndBackup:
    """Tests for corpus snapshots and password-protected backups."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary SQLite database with some test data."""
        db_file = tmp_path / "test_corpus.db"

        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.execute(
            "INSERT INTO test_table (name) VALUES ('test1'), ('test2'), ('test3')"
        )

        # Insert and delete data to create fragmentation for VACUUM to clean
        for i in range(100):
            cursor.execute(
                "INSERT INTO test_table (name) VALUES (?)", (f"temp_data_{i}",)
            )

        cursor.execute("DELETE FROM test_table WHERE name LIKE 'temp_data_%'")
        conn.commit()
        conn.close()

        return str(db_file)

    def test_create_corpus_database_snapshot(self, temp_db_path):
        """Test that a valid binary snapshot is created given a specific DB file."""
        snapshot = create_sqlite_snapshot(temp_db_path)

        assert isinstance(snapshot, bytes)
        assert len(snapshot) > 0
        assert snapshot[:16] == SQLITE_HEADER

    def test_create_corpus_database_snapshot_missing_file(self, tmp_path):
        """Test snapshot creation raises FileNotFoundError when database file does not exist."""
        missing_db = tmp_path / "nonexistent.db"
        with pytest.raises(FileNotFoundError, match="SQLite database does not exist"):
            create_sqlite_snapshot(missing_db)

    @pytest.mark.skipif(not HAS_PYZIPPER, reason="pyzipper is not installed")
    def test_create_password_protected_backup(self, temp_db_path):
        """Test creation of a password-protected ZIP backup."""
        snapshot = create_sqlite_snapshot(temp_db_path)
        password = "secure_password_123"

        zip_data = create_password_protected_backup(snapshot, password, archive_name="corpus.db")


        assert isinstance(zip_data, bytes)
        assert len(zip_data) > 0
        assert zip_data[:4] == b"PK\x03\x04"

    def test_create_and_verify_backup(self, temp_db_path, tmp_path):
        """Verify database backup creation, size, and integrity."""
        # 1. Create database backup
        snapshot = create_sqlite_snapshot(temp_db_path)
        backup_file = tmp_path / "test_backup.db"
        backup_file.write_bytes(snapshot)

        # 2. Verify backup file exists
        assert backup_file.exists()

        # 3. Verify size matches source
        source_size = os.path.getsize(temp_db_path)
        backup_size = backup_file.stat().st_size
        assert backup_size == source_size

        # 4. Verify integrity check on backup file returns True
        conn = sqlite3.connect(str(backup_file))
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        res = cursor.fetchone()
        conn.close()
        assert res[0] == "ok"



class TestOptimizeDatabase:
    """Tests for the optimize_database function."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary SQLite database with fragmented data."""
        db_file = tmp_path / "test_opt.db"
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.execute(
            "INSERT INTO test_table (name) VALUES ('test1'), ('test2'), ('test3')"
        )

        for i in range(100):
            cursor.execute(
                "INSERT INTO test_table (name) VALUES (?)", (f"temp_data_{i}",)
            )

        cursor.execute("DELETE FROM test_table WHERE name LIKE 'temp_data_%'")
        conn.commit()
        conn.close()

        return str(db_file)

    def test_optimize_database_success(self, temp_db_path):
        """Test that optimize_database successfully runs VACUUM and ANALYZE."""
        result = optimize_database(temp_db_path)

        assert result is True
        assert os.path.exists(temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM test_table LIMIT 1")
        assert cursor.fetchone()[0] == "test1"
        conn.close()

    def test_optimize_database_missing_file(self, tmp_path):
        """Test optimization when the database file does not exist."""
        missing_db = tmp_path / "nonexistent.db"
        result = optimize_database(missing_db)

        assert result is False

    def test_optimize_database_logs_size_reduction(self, temp_db_path, caplog):
        """Test that optimize_database logs the size reduction statistics."""
        caplog.set_level(logging.INFO)

        result = optimize_database(temp_db_path)

        assert result is True
        assert "Starting database optimization" in caplog.text
        assert "Executing PRAGMA optimize" in caplog.text
        assert "Executing VACUUM" in caplog.text
        assert "Executing ANALYZE" in caplog.text
        assert "Database optimization completed successfully" in caplog.text
        assert "Space reclaimed:" in caplog.text

    def test_run_incremental_vacuum(self, temp_db_path):
        """Verify incremental vacuum executes successfully."""
        conn = sqlite3.connect(temp_db_path)
        try:
            assert run_incremental_vacuum(conn) is True
        finally:
            conn.close()


class TestRestoreDatabase:
    """Tests for secure database restoration."""

    def test_restore_database_success(self, tmp_path):
        """Test restoring a valid SQLite database backup into a target destination."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        source_db = backup_dir / "valid_backup.db"
        conn = sqlite3.connect(str(source_db))
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO test_table (name) VALUES ('item1')")
        conn.commit()
        conn.close()

        dest_db = tmp_path / "restored_corpus.db"

        restored_path = restore(
            source="valid_backup.db",
            backup_dir=backup_dir,
            destination=dest_db,
        )

        assert restored_path == dest_db.resolve()
        assert dest_db.exists()

        conn = sqlite3.connect(str(dest_db))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM test_table")
        assert cursor.fetchone()[0] == "item1"
        conn.close()

    def test_restore_database_security_error_outside_dir(self, tmp_path):
        """Verify restore raises BackupRestoreSecurityError when source lies outside backup_dir."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        outside_file = tmp_path / "outside.db"
        outside_file.write_bytes(SQLITE_HEADER + b"data")

        with pytest.raises(
            BackupRestoreSecurityError,
            match="must be inside the designated backup directory",
        ):
            restore(
                source="../outside.db",
                backup_dir=backup_dir,
                destination=tmp_path / "target.db",
            )


class TestCleanupOldBackups:
    """Tests for the cleanup_old_backups function."""

    def test_cleanup_nonexistent_directory(self):
        """Verify graceful handling of non-existent backup directories."""
        result = cleanup_old_backups(backup_dir="/nonexistent/backup/dir")
        assert result["files_deleted"] == 0
        assert result["bytes_freed"] == 0

    def test_cleanup_empty_directory(self, tmp_path):
        """Verify that an empty directory results in no deletions."""
        result = cleanup_old_backups(backup_dir=tmp_path)
        assert result["files_deleted"] == 0
        assert result["bytes_freed"] == 0

    def test_cleanup_respects_max_backups(self, tmp_path):
        """Verify that only the newest `max_backups` files are retained."""
        for i in range(15):
            file_path = tmp_path / f"backup_{i}.db"
            file_path.write_bytes(SQLITE_HEADER + b"dummy data")
            old_time = time.time() - (15 - i)
            os.utime(file_path, (old_time, old_time))

        result = cleanup_old_backups(
            backup_dir=tmp_path, max_backups=10, max_age_days=365
        )

        assert (
            result["files_deleted"] == 5
        ), "Should delete 5 files to respect max_backups=10"
        assert result["bytes_freed"] > 0, "Should report freed bytes"

        remaining_files = list(tmp_path.glob("*.db"))
        assert len(remaining_files) == 10
        for f in remaining_files:
            assert (
                int(f.stem.split("_")[1]) >= 5
            ), "Oldest 5 files should have been deleted"

    def test_cleanup_respects_max_age_days(self, tmp_path):
        """Verify that files older than `max_age_days` are deleted regardless of count."""
        old_file = tmp_path / "old_backup.db"
        old_file.write_bytes(SQLITE_HEADER + b"old data")
        old_time = time.time() - (31 * 24 * 60 * 60)  # 31 days ago
        os.utime(old_file, (old_time, old_time))

        new_file = tmp_path / "new_backup.db"
        new_file.write_bytes(SQLITE_HEADER + b"new data")

        result = cleanup_old_backups(
            backup_dir=tmp_path, max_backups=10, max_age_days=30
        )

        assert result["files_deleted"] == 1, "Should delete the file older than 30 days"
        assert (tmp_path / "new_backup.db").exists(), "New file should be retained"
        assert not (tmp_path / "old_backup.db").exists(), "Old file should be deleted"

    def test_cleanup_handles_os_error_gracefully(self, tmp_path):
        """Verify that OSError during deletion is logged and does not crash the function."""
        file_path = tmp_path / "locked_backup.db"
        file_path.write_bytes(SQLITE_HEADER + b"locked data")

        with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
            result = cleanup_old_backups(
                backup_dir=tmp_path, max_backups=1, max_age_days=30
            )

            assert result["files_deleted"] == 0, "Should not count failed deletions"
            assert (
                result["bytes_freed"] == 0
            ), "Should not count freed bytes for failed deletions"


class TestGetDatabaseSizeBytes:
    """Tests for the get_database_size_bytes helper (issue #1047)."""

    def test_returns_zero_for_nonexistent_file(self, tmp_path):
        """A missing database file must report size 0, not raise."""
        missing_db = tmp_path / "does_not_exist.db"

        size = get_database_size_bytes(missing_db)

        assert size == 0

    def test_returns_zero_for_nonexistent_file_string_path(self, tmp_path):
        """The function must accept str paths as well as Path objects."""
        missing_db = str(tmp_path / "does_not_exist.db")

        size = get_database_size_bytes(missing_db)

        assert size == 0

    def test_returns_size_for_temp_db(self, tmp_path):
        """Verify size calculation for a temp DB created with known content.

        Acceptance criterion from issue #1047:
            "Add unit test verifying size calculation for temp DB."
        """
        db_path = tmp_path / "test_size.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, payload TEXT)")
        conn.executemany(
            "INSERT INTO sample (payload) VALUES (?)",
            [(f"row-{i}",) for i in range(50)],
        )
        conn.commit()
        conn.close()

        size = get_database_size_bytes(db_path)

        # The file must exist and report a positive size.
        assert size > 0
        # The reported size must match the actual on-disk size.
        assert size == db_path.stat().st_size

    def test_size_grows_after_insert(self, tmp_path):
        """Inserting more data must increase the reported size."""
        db_path = tmp_path / "test_grow.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT)")
        conn.execute("INSERT INTO docs (content) VALUES ('seed')")
        conn.commit()
        conn.close()

        size_before = get_database_size_bytes(db_path)

        conn = sqlite3.connect(str(db_path))
        conn.executemany(
            "INSERT INTO docs (content) VALUES (?)",
            [(f"padding-{'x' * 200}-{i}",) for i in range(100)],
        )
        conn.commit()
        conn.close()

        size_after = get_database_size_bytes(db_path)

        assert size_after > size_before

    def test_accepts_path_and_str_equivalently(self, tmp_path):
        """Both Path and str inputs must resolve to the same size."""
        db_path = tmp_path / "test_type.db"
        db_path.write_bytes(SQLITE_HEADER + b"payload")

        size_from_path = get_database_size_bytes(db_path)
        size_from_str = get_database_size_bytes(str(db_path))

        assert size_from_path == size_from_str == db_path.stat().st_size

    def test_expands_user_home(self, monkeypatch, tmp_path):
        """A leading ``~`` must be expanded against the home directory."""
        db_path = tmp_path / "home_db.db"
        db_path.write_bytes(b"not-a-real-sqlite-db-but-size-still-works")
        # Path.expanduser() reads $HOME or $USERPROFILE, so patch both env vars.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))

        size = get_database_size_bytes("~/home_db.db")

        assert size == db_path.stat().st_size

class TestGetDatabaseTableStats:
    """Tests for the get_database_table_stats helper (issue #1156)."""

    def test_returns_empty_stats_for_nonexistent_file(self, tmp_path):
        """A missing database file must return {'_table_count': 0}."""
        missing_db = tmp_path / "does_not_exist.db"

        stats = get_database_table_stats(missing_db)

        assert isinstance(stats, dict)
        assert stats["_table_count"] == 0
        # Should only contain the special key, no table entries
        assert len(stats) == 1

    def test_returns_empty_stats_for_nonexistent_file_string_path(self, tmp_path):
        """The function must accept str paths as well as Path objects."""
        missing_db = str(tmp_path / "does_not_exist.db")

        stats = get_database_table_stats(missing_db)

        assert stats["_table_count"] == 0

    def test_returns_stats_for_database_with_tables(self, tmp_path):
        """Verify stats dictionary contains correct table names and row counts."""
        db_path = tmp_path / "test_stats.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"
        )
        conn.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT)"
        )
        conn.execute(
            "CREATE TABLE incidents (id INTEGER PRIMARY KEY, severity TEXT)"
        )
        conn.executemany(
            "INSERT INTO users (name) VALUES (?)",
            [("Alice",), ("Bob",), ("Charlie",)],
        )
        conn.executemany(
            "INSERT INTO documents (title) VALUES (?)",
            [("Doc1",), ("Doc2",)],
        )
        conn.executemany(
            "INSERT INTO incidents (severity) VALUES (?)",
            [("High",), ("Medium",), ("Low",), ("Critical",)],
        )
        conn.commit()
        conn.close()

        stats = get_database_table_stats(db_path)

        # Verify _table_count special key
        assert stats["_table_count"] == 3

        # Verify individual table row counts
        assert stats["users"] == 3
        assert stats["documents"] == 2
        assert stats["incidents"] == 4

    def test_returns_zero_for_empty_table(self, tmp_path):
        """A table with no rows should report 0 as its row count."""
        db_path = tmp_path / "empty_table.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE empty_table (id INTEGER PRIMARY KEY, data TEXT)")
        conn.commit()
        conn.close()

        stats = get_database_table_stats(db_path)

        assert stats["_table_count"] == 1
        assert stats["empty_table"] == 0

    def test_excludes_sqlite_internal_tables(self, tmp_path):
        """Internal SQLite tables (sqlite_*) must be excluded from stats."""
        db_path = tmp_path / "internal_tables.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE user_data (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO user_data (id) VALUES (1)")
        # SQLite auto-creates sqlite_sequence for AUTOINCREMENT tables
        conn.commit()
        conn.close()

        stats = get_database_table_stats(db_path)

        # Only user_data should appear, not sqlite_sequence
        table_names = [k for k in stats.keys() if k != "_table_count"]
        assert "user_data" in table_names
        assert not any(name.startswith("sqlite_") for name in table_names)

    def test_excludes_views_from_stats(self, tmp_path):
        """Views should be excluded from table stats."""
        db_path = tmp_path / "views.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE base_table (id INTEGER PRIMARY KEY, val TEXT)")
        conn.execute("INSERT INTO base_table (val) VALUES ('test')")
        conn.execute(
            "CREATE VIEW test_view AS SELECT * FROM base_table WHERE val = 'test'"
        )
        conn.commit()
        conn.close()

        stats = get_database_table_stats(db_path)

        # Only base_table should appear, not test_view
        assert stats["_table_count"] == 1
        assert "base_table" in stats
        assert "test_view" not in stats

    def test_handles_directory_path(self, tmp_path):
        """A directory path must return empty stats, not raise."""
        stats = get_database_table_stats(tmp_path)

        assert stats["_table_count"] == 0

    def test_accepts_path_and_str_equivalently(self, tmp_path):
        """Both Path and str inputs must resolve to the same stats."""
        db_path = tmp_path / "type_test.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO items (id) VALUES (1)")
        conn.commit()
        conn.close()

        stats_from_path = get_database_table_stats(db_path)
        stats_from_str = get_database_table_stats(str(db_path))

        assert stats_from_path == stats_from_str

    def test_stats_with_multiple_tables(self, tmp_path):
        """Verify correct stats for a database with many tables."""
        db_path = tmp_path / "multi_table.db"

        conn = sqlite3.connect(str(db_path))
        # Create 5 tables with varying row counts
        for i in range(5):
            conn.execute(
                f"CREATE TABLE table_{i} (id INTEGER PRIMARY KEY, data TEXT)"
            )
            for j in range(i + 1):
                conn.execute(
                    f"INSERT INTO table_{i} (data) VALUES (?)",
                    (f"row_{j}",),
                )
        conn.commit()
        conn.close()

        stats = get_database_table_stats(db_path)

        assert stats["_table_count"] == 5
        for i in range(5):
            assert stats[f"table_{i}"] == i + 1

    def test_expands_user_home(self, monkeypatch, tmp_path):
        """A leading ``~`` must be expanded against the home directory."""
        db_path = tmp_path / "home_stats.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO test (id) VALUES (1)")
        conn.commit()
        conn.close()

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))

        stats = get_database_table_stats("~/home_stats.db")

        assert stats["_table_count"] == 1
        assert stats["test"] == 1



class TestGetDatabaseTableStatsIssue1773:
    """Regression tests for issue #1773 acceptance criteria.

    Issue #1773 asks for ``get_database_table_stats(db_path: str | Path)
    -> dict[str, int]`` in ``src/db/database_backup.py`` that queries
    ``sqlite_master`` for table names and returns
    ``{table_name: row_count}``.

    The function was originally added under issue #1156 — these tests
    lock in the issue #1773 acceptance criteria so a future refactor
    cannot silently break the contract.
    """

    def test_issue_1773_function_exists_with_correct_signature(self):
        """The function must exist and accept str | Path, return dict[str, int]."""
        import inspect
        from src.db.database_backup import get_database_table_stats

        sig = inspect.signature(get_database_table_stats)
        params = list(sig.parameters.values())
        assert len(params) == 1, (
            f"Expected 1 parameter (db_path), got {len(params)}"
        )
        # Annotation should be str | Path (or the typing-quoted form).
        annotation = params[0].annotation
        annotation_str = str(annotation)
        assert "str" in annotation_str and "Path" in annotation_str, (
            f"db_path annotation must be 'str | Path', got {annotation_str}"
        )
        # Return annotation must be dict[str, int].
        return_str = str(sig.return_annotation)
        assert "dict" in return_str and "int" in return_str, (
            f"return annotation must be dict[str, int], got {return_str}"
        )

    def test_issue_1773_queries_sqlite_master(self):
        """The function's source must query ``sqlite_master`` for table names."""
        import inspect
        from src.db.database_backup import get_database_table_stats

        source = inspect.getsource(get_database_table_stats)
        assert "sqlite_master" in source, (
            "get_database_table_stats must query sqlite_master for table names"
        )
        assert "type='table'" in source, (
            "get_database_table_stats must filter sqlite_master by type='table'"
        )

    def test_issue_1773_returns_dict_of_table_name_to_row_count(self, tmp_path):
        """Return value must be ``{table_name: row_count}`` with int values."""
        import sqlite3 as _sqlite3
        from src.db.database_backup import get_database_table_stats

        db_path = tmp_path / "issue1773.db"
        conn = _sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO t1 (id) VALUES (?)", [(1,), (2,), (3,)])
        conn.execute("INSERT INTO t2 (id) VALUES (1)")
        conn.commit()
        conn.close()

        stats = get_database_table_stats(db_path)

        assert isinstance(stats, dict)
        # Every value must be an int (row count).
        for key, value in stats.items():
            assert isinstance(value, int), (
                f"stats[{key!r}] = {value!r} is {type(value).__name__}, "
                "expected int"
            )
        # The special _table_count key is also an int.
        assert isinstance(stats.get("_table_count"), int)
        # Spot-check the actual row counts.
        assert stats["t1"] == 3
        assert stats["t2"] == 1

    def test_issue_1773_accepts_str_and_path_equivalently(self, tmp_path):
        """The function must accept both ``str`` and ``Path`` inputs."""
        import sqlite3 as _sqlite3
        from pathlib import Path as _Path
        from src.db.database_backup import get_database_table_stats

        db_path = tmp_path / "issue1773_types.db"
        conn = _sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE x (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO x (id) VALUES (1)")
        conn.commit()
        conn.close()

        stats_from_str = get_database_table_stats(str(db_path))
        stats_from_path = get_database_table_stats(_Path(db_path))

        assert stats_from_str == stats_from_path
        assert stats_from_str["x"] == 1

    def test_issue_1773_returns_empty_for_nonexistent_db(self, tmp_path):
        """A nonexistent db must return ``{'_table_count': 0}``, not raise."""
        from src.db.database_backup import get_database_table_stats

        stats = get_database_table_stats(tmp_path / "does_not_exist.db")

        assert isinstance(stats, dict)
        assert stats == {"_table_count": 0}

class TestGetTableSchemaInfo:
    """Tests for the get_table_schema_info helper (issue #1586)."""

    def _make_db(self, tmp_path) -> Path:
        """Create a temporary SQLite database with a documents table."""
        db_path = tmp_path / "schema_test.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                tags TEXT DEFAULT 'untagged',
                size REAL
            )
            """
        )
        conn.commit()
        conn.close()

        return db_path

    def test_returns_column_metadata_for_table(self, tmp_path):
        """Verify PRAGMA table_info output is returned as column dictionaries."""
        db_path = self._make_db(tmp_path)

        schema = get_table_schema_info(db_path, "documents")

        assert len(schema) == 4
        assert schema[0] == {
            "name": "id",
            "type": "INTEGER",
            "notnull": 0,
            "dflt_value": None,
            "pk": 1,
        }
        assert schema[1] == {
            "name": "filename",
            "type": "TEXT",
            "notnull": 1,
            "dflt_value": None,
            "pk": 0,
        }
        # TEXT defaults are reported with surrounding quotes by SQLite
        assert schema[2]["name"] == "tags"
        assert schema[2]["type"] == "TEXT"
        assert schema[2]["dflt_value"] == "'untagged'"
        assert schema[3] == {
            "name": "size",
            "type": "REAL",
            "notnull": 0,
            "dflt_value": None,
            "pk": 0,
        }

    def test_returns_empty_list_for_nonexistent_table(self, tmp_path):
        """An unknown table must return an empty list, not raise."""
        db_path = self._make_db(tmp_path)

        schema = get_table_schema_info(db_path, "missing_table")

        assert schema == []

    def test_returns_empty_list_for_nonexistent_db(self, tmp_path):
        """A missing database file must return an empty list."""
        missing_db = tmp_path / "does_not_exist.db"

        schema = get_table_schema_info(missing_db, "documents")

        assert schema == []

    def test_returns_empty_list_for_directory_path(self, tmp_path):
        """A directory path must return an empty list, not raise."""
        schema = get_table_schema_info(tmp_path, "documents")

        assert schema == []

    def test_rejects_unsafe_table_name(self, tmp_path):
        """Table names that are not plain identifiers must be refused."""
        db_path = self._make_db(tmp_path)

        schema = get_table_schema_info(
            db_path, "documents; DROP TABLE documents; --"
        )

        assert schema == []

    def test_accepts_path_and_str_equivalently(self, tmp_path):
        """Both Path and str inputs must return identical metadata."""
        db_path = self._make_db(tmp_path)

        schema_from_path = get_table_schema_info(db_path, "documents")
        schema_from_str = get_table_schema_info(str(db_path), "documents")

        assert schema_from_path == schema_from_str


class TestCheckpointWalLog:
    """Tests for the checkpoint_wal_log function."""

    def test_checkpoint_wal_log_success(self, tmp_path):
        """Verify that checkpoint_wal_log successfully checkpoints a database in WAL mode."""
        db_path = tmp_path / "wal_test.db"

        # 1. Create a database and enable WAL mode
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
        conn.execute("INSERT INTO test (val) VALUES ('test')")
        conn.commit()

        # Check WAL file exists (it is preserved on close when other connections are active, or since we keep conn open)
        wal_path = Path(f"{db_path}-wal")
        assert wal_path.exists()

        # 2. Run checkpoint and mock the logger to verify output
        with patch("src.db.database_backup.logger") as mock_logger:
            res = checkpoint_wal_log(db_path)
            assert res is True

            # Verify that logger.info was called with size updates
            log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("WAL file size before checkpoint" in log for log in log_calls)
            assert any("WAL file size after checkpoint" in log for log in log_calls)

        conn.close()


    def test_checkpoint_wal_log_file_not_found(self):
        """Verify checkpoint fails for non-existent database paths."""
        res = checkpoint_wal_log("/nonexistent/path/to/db.db")
        assert res is False

    def test_checkpoint_wal_log_is_a_directory(self, tmp_path):
        """Verify checkpoint fails if the database path is a directory."""
        res = checkpoint_wal_log(tmp_path)
        assert res is False

class TestCreateDatabaseBackup:
    """Tests for the create_database_backup function (issue #1488)."""

    def _make_sample_db(self, db_path):
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO test (name) VALUES ('sample')")
        conn.commit()
        conn.close()

    def test_creates_compressed_backup_by_default(self, tmp_path):
        db_path = tmp_path / "source.db"
        self._make_sample_db(db_path)

        backup_path = create_database_backup(db_path, backup_dir=tmp_path / "backups")

        assert backup_path.exists()
        assert backup_path.name.endswith(".db.gz")

        with gzip.open(backup_path, "rb") as gz_file:
            decompressed = gz_file.read()
        assert decompressed.startswith(SQLITE_HEADER)

    def test_creates_uncompressed_backup_when_disabled(self, tmp_path):
        db_path = tmp_path / "source.db"
        self._make_sample_db(db_path)

        backup_path = create_database_backup(
            db_path, backup_dir=tmp_path / "backups", compress_backup=False
        )

        assert backup_path.exists()
        assert backup_path.name.endswith(".db")
        assert not backup_path.name.endswith(".db.gz")
        assert backup_path.read_bytes().startswith(SQLITE_HEADER)

    def test_create_database_backup_raises_file_not_found_when_missing(self):
        """Verify FileNotFoundError is raised when source DB is missing (Issue #1885)."""
        missing_db_path = "non_existent_database.sqlite3"
        
        expected_message = f"Source database file does not exist: {missing_db_path}"
        with pytest.raises(FileNotFoundError, match=expected_message):
            create_database_backup(missing_db_path)
            