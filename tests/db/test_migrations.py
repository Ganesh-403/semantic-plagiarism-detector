from __future__ import annotations

import sqlite3

import pytest

from src.db.migrations import (AUTH_SCHEMA_VERSION, CORPUS_SCHEMA_VERSION,
                               column_exists, get_user_version, index_exists,
                               migrate_auth_database, migrate_corpus_database,
                               run_migrations, table_exists, check_table_exists)


def connect(path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def test_fresh_corpus_database_reaches_latest_version(tmp_path):
    with connect(tmp_path / "fresh-corpus.db") as connection:
        version = migrate_corpus_database(connection)

        assert version == CORPUS_SCHEMA_VERSION
        assert get_user_version(connection) == CORPUS_SCHEMA_VERSION
        assert table_exists(connection, "documents")
        assert table_exists(connection, "chunks")
        assert table_exists(connection, "plagiarism_incidents")
        assert column_exists(connection, "documents", "detected_language")
        assert column_exists(connection, "documents", "created_at")
        assert index_exists(connection, "idx_documents_upload_date")
        assert index_exists(connection, "idx_documents_class_section")
        assert index_exists(connection, "idx_chunks_filename")
        assert index_exists(connection, "idx_incidents_status")
    assert index_exists(connection, "idx_documents_created_at")
        assert index_exists(connection, "idx_incidents_severity_time")

def test_fresh_auth_database_reaches_latest_version(tmp_path):
    with connect(tmp_path / "fresh-users.db") as connection:
        version = migrate_auth_database(connection)

        assert version == AUTH_SCHEMA_VERSION
        assert get_user_version(connection) == AUTH_SCHEMA_VERSION
        assert table_exists(connection, "users")
        assert column_exists(connection, "users", "tour_completed")
        assert column_exists(connection, "users", "otp_secret")
        assert column_exists(connection, "users", "two_factor_enabled")
        assert column_exists(connection, "users", "is_active")
        assert index_exists(connection, "idx_users_role")


def test_empty_existing_databases_upgrade_safely(tmp_path):
    corpus_path = tmp_path / "empty-corpus.db"
    auth_path = tmp_path / "empty-users.db"
    sqlite3.connect(corpus_path).close()
    sqlite3.connect(auth_path).close()

    with connect(corpus_path) as connection:
        migrate_corpus_database(connection)
        assert get_user_version(connection) == CORPUS_SCHEMA_VERSION

    with connect(auth_path) as connection:
        migrate_auth_database(connection)
        assert get_user_version(connection) == AUTH_SCHEMA_VERSION


def test_old_corpus_database_migrates_without_data_loss(tmp_path):
    path = tmp_path / "old-corpus.db"

    with connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                upload_date TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE chunks (
                vector_id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (filename)
                    REFERENCES documents(filename)
                    ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            INSERT INTO documents (
                filename, file_hash, upload_date
            ) VALUES (?, ?, ?)
            """,
            ("legacy.pdf", "legacy-hash", "2026-01-01T00:00:00"),
        )
        connection.execute(
            """
            INSERT INTO chunks (
                vector_id, filename, chunk_index, chunk_text, embedding
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (0, "legacy.pdf", 0, "legacy text", b"\x00\x00\x00\x00"),
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()

        migrate_corpus_database(connection)

        row = connection.execute(
            """
            SELECT filename, file_hash, class_section,
                   student_name, assignment_title
            FROM documents
            """
        ).fetchone()
        assert row == (
            "legacy.pdf",
            "legacy-hash",
            None,
            None,
            None,
        )

        chunk = connection.execute("SELECT filename, chunk_text FROM chunks").fetchone()
        assert chunk == ("legacy.pdf", "legacy text")
        assert get_user_version(connection) == CORPUS_SCHEMA_VERSION


def test_old_auth_database_migrates_without_data_loss(tmp_path):
    path = tmp_path / "old-users.db"

    with connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'teacher'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO users (username, password, role)
            VALUES (?, ?, ?)
            """,
            ("legacy", "stored-hash", "teacher"),
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()

        migrate_auth_database(connection)

        row = connection.execute(
            """
            SELECT username, password, role, tour_completed,
                   otp_secret, two_factor_enabled
            FROM users
            WHERE username = ?
            """,
            ("legacy",),
        ).fetchone()
        assert row == (
            "legacy",
            "stored-hash",
            "teacher",
            0,
            None,
            0,
        )
        assert get_user_version(connection) == AUTH_SCHEMA_VERSION


def test_migrations_are_idempotent(tmp_path):
    with connect(tmp_path / "idempotent.db") as connection:
        first = migrate_corpus_database(connection)
        first_schema = connection.execute(
            """
            SELECT type, name, sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()

        second = migrate_corpus_database(connection)
        second_schema = connection.execute(
            """
            SELECT type, name, sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()

        assert first == second == CORPUS_SCHEMA_VERSION
        assert first_schema == second_schema


def test_migrations_execute_in_sequential_order():
    connection = sqlite3.connect(":memory:")
    calls: list[int] = []

    migrations = {
        1: lambda conn: calls.append(1),
        2: lambda conn: calls.append(2),
        3: lambda conn: calls.append(3),
    }

    try:
        version = run_migrations(
            connection,
            migrations=migrations,
            target_version=3,
        )
        assert version == 3
        assert calls == [1, 2, 3]
    finally:
        connection.close()


def test_failed_migration_rolls_back_schema_data_and_version():
    connection = sqlite3.connect(":memory:")

    def migration_one(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE preserved_test (value TEXT)")
        conn.execute("INSERT INTO preserved_test (value) VALUES ('temporary')")

    def migration_two(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE should_rollback (id INTEGER)")
        raise RuntimeError("intentional migration failure")

    try:
        with pytest.raises(
            RuntimeError,
            match="intentional migration failure",
        ):
            run_migrations(
                connection,
                migrations={1: migration_one, 2: migration_two},
                target_version=2,
            )

        assert get_user_version(connection) == 0
        assert not table_exists(connection, "preserved_test")
        assert not table_exists(connection, "should_rollback")
    finally:
        connection.close()


def test_newer_database_version_is_rejected():
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("PRAGMA user_version = 99")
        with pytest.raises(RuntimeError, match="newer than supported"):
            migrate_corpus_database(connection)
    finally:
        connection.close()


def test_schema_inspection_helpers_handle_missing_objects():
    connection = sqlite3.connect(":memory:")
    try:
        assert not table_exists(connection, "missing")
        assert not column_exists(connection, "missing", "column")
        assert not index_exists(connection, "missing_index")
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Issue #1051 — Log informational message on successful migration execution
# ---------------------------------------------------------------------------


def test_successful_migration_logs_info_message(tmp_path, caplog):
    """run_migrations must log an INFO message with the old and new versions
    after a successful migration (issue #1051)."""
    import logging

    connection = sqlite3.connect(str(tmp_path / "log-test.db"))
    try:
        with caplog.at_level(logging.INFO, logger="src.db.migrations.common"):
            run_migrations(
                connection,
                migrations={1: lambda conn: None},
                target_version=1,
            )

        assert any(
            "Database migration from version 0 to 1 completed successfully."
            in record.message
            for record in caplog.records
        ), f"Expected migration log message not found in: {[r.message for r in caplog.records]}"
    finally:
        connection.close()


def test_corpus_migration_logs_info_message(tmp_path, caplog):
    """migrate_corpus_database must log an INFO message on success (issue #1051)."""
    import logging

    with connect(tmp_path / "corpus-log.db") as connection:
        with caplog.at_level(logging.INFO, logger="src.db.migrations.common"):
            migrate_corpus_database(connection)

        assert any(
            "completed successfully" in record.message
            for record in caplog.records
        ), f"Expected migration log message not found in: {[r.message for r in caplog.records]}"


def test_auth_migration_logs_info_message(tmp_path, caplog):
    """migrate_auth_database must log an INFO message on success (issue #1051)."""
    import logging

    with connect(tmp_path / "auth-log.db") as connection:
        with caplog.at_level(logging.INFO, logger="src.db.migrations.common"):
            migrate_auth_database(connection)

        assert any(
            "completed successfully" in record.message
            for record in caplog.records
        ), f"Expected migration log message not found in: {[r.message for r in caplog.records]}"


def test_no_log_when_already_at_target_version(tmp_path, caplog):
    """run_migrations must NOT log when the database is already at the target
    version (no migration was performed)."""
    import logging

    connection = sqlite3.connect(str(tmp_path / "noop-test.db"))
    try:
        # First run: brings it to version 1
        run_migrations(
            connection,
            migrations={1: lambda conn: None},
            target_version=1,
        )

        caplog.clear()

        # Second run: already at version 1, should be a no-op
        with caplog.at_level(logging.INFO, logger="src.db.migrations.common"):
            result = run_migrations(
                connection,
                migrations={1: lambda conn: None},
                target_version=1,
            )

        assert result == 1
        assert not any(
            "completed successfully" in record.message
            for record in caplog.records
        ), "Should not log when no migration was performed"
    finally:
        connection.close()


def test_migration_duration_logging(tmp_path, caplog):
    """run_migrations must log the execution duration for each executed migration function."""
    import logging

    def migration_dummy_test_func(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE dummy_test (id INT)")

    connection = sqlite3.connect(str(tmp_path / "duration-test.db"))
    try:
        with caplog.at_level(logging.INFO, logger="src.db.migrations.common"):
            run_migrations(
                connection,
                migrations={1: migration_dummy_test_func},
                target_version=1,
            )

        assert any(
            "Migration [migration_dummy_test_func] executed in" in record.message
            for record in caplog.records
        ), f"Expected duration log message not found in: {[r.message for r in caplog.records]}"
    finally:
        connection.close()
def test_migration_013_adds_incident_severity_index(tmp_path):
    with connect(tmp_path / "severity-idx-corpus.db") as connection:
        migrate_corpus_database(connection)

        assert index_exists(connection, "idx_incidents_severity_time")


def test_check_table_exists():
    """Verify check_table_exists checks table existence by querying sqlite_master."""
    connection = sqlite3.connect(":memory:")
    try:
        assert check_table_exists(connection, "test_table") is False
        connection.execute("CREATE TABLE test_table (id INTEGER)")
        assert check_table_exists(connection, "test_table") is True
    finally:
        connection.close()


def test_rollback_migration_restores_previous_version():
    """Verify successful rollback of a test migration (issue #1358)."""
    connection = sqlite3.connect(":memory:")

    def migration_up(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE rollback_test (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO rollback_test (name) VALUES (\'before\')")

    def migration_down(conn: sqlite3.Connection) -> None:
        conn.execute("DROP TABLE rollback_test")

    try:
        run_migrations(
            connection,
            migrations={1: migration_up},
            target_version=1,
        )
        assert get_user_version(connection) == 1
        assert table_exists(connection, "rollback_test")

        rollback_migration(
            connection,
            down_migrations={1: migration_down},
            target_version=0,
        )
        assert get_user_version(connection) == 0
        assert not table_exists(connection, "rollback_test")
    finally:
        connection.close()


def test_rollback_migration_no_op_when_already_at_target():
    """Rollback to current version should be a no-op."""
    connection = sqlite3.connect(":memory:")

    def migration_up(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE noop_test (id INTEGER)")

    try:
        run_migrations(
            connection,
            migrations={1: migration_up},
            target_version=1,
        )
        result = rollback_migration(
            connection,
            down_migrations={1: lambda c: None},
            target_version=1,
        )
        assert result == 1
        assert table_exists(connection, "noop_test")
    finally:
        connection.close()


def test_rollback_migration_fails_when_target_higher_than_current():
    """Rollback to a version higher than current must raise RuntimeError."""
    connection = sqlite3.connect(":memory:")

    try:
        run_migrations(
            connection,
            migrations={1: lambda c: c.execute("CREATE TABLE t (id INTEGER)")},
            target_version=1,
        )
        with pytest.raises(RuntimeError, match="Cannot rollback from version 1 to higher version 5"):
            rollback_migration(
                connection,
                down_migrations={},
                target_version=5,
            )
    finally:
        connection.close()


def test_rollback_migration_fails_on_missing_down_definition():
    """Rollback must raise RuntimeError if a down-migration is missing."""
    connection = sqlite3.connect(":memory:")

    def up_1(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE a (id INTEGER)")

    def up_2(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE b (id INTEGER)")

    try:
        run_migrations(
            connection,
            migrations={1: up_1, 2: up_2},
            target_version=2,
        )
        with pytest.raises(RuntimeError, match="Down-migration definitions are missing for versions: 2"):
            rollback_migration(
                connection,
                down_migrations={1: lambda c: c.execute("DROP TABLE a")},
                target_version=1,
            )
    finally:
        connection.close()


def test_failed_rollback_migration_is_atomic():
    """A failed down-migration must not partially modify schema or version."""
    connection = sqlite3.connect(":memory:")

    def up_1(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE atomic_test (id INTEGER)")

    def bad_down_1(conn: sqlite3.Connection) -> None:
        conn.execute("DROP TABLE atomic_test")
        raise RuntimeError("intentional rollback failure")

    try:
        run_migrations(
            connection,
            migrations={1: up_1},
            target_version=1,
        )
        with pytest.raises(RuntimeError, match="intentional rollback failure"):
            rollback_migration(
                connection,
                down_migrations={1: bad_down_1},
                target_version=0,
            )

        assert get_user_version(connection) == 1
        assert table_exists(connection, "atomic_test")
    finally:
        connection.close()
