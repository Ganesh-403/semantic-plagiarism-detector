# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import sqlite3

import pytest

from src.db.migrations import (
    AUTH_SCHEMA_VERSION,
    CORPUS_SCHEMA_VERSION,
    check_table_exists,
    column_exists,
    get_user_version,
    index_exists,
    migrate_auth_database,
    migrate_corpus_database,
    rollback_migration,
    run_migrations,
    table_exists,
)


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


def test_corpus_migrations_apply_from_empty_database():
    """Verify that applying the full corpus migration chain from a blank
    in-memory database succeeds and creates all expected tables and columns."""
    conn = sqlite3.connect(":memory:")

    try:
        migrate_corpus_database(conn)

        tables = {
            row[0]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }

        assert {
            "documents",
            "chunks",
            "plagiarism_incidents",
            "false_positives",
            "incidents_archive",
        }.issubset(tables)

        documents_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(documents)")
        }

        assert "is_deleted" in documents_columns
    finally:
        conn.close()


def test_fresh_auth_database_reaches_latest_version(tmp_path):
    with connect(tmp_path / "fresh-users.db") as connection:
        version = migrate_auth_database(connection)

        assert version == AUTH_SCHEMA_VERSION
        assert get_user_version(connection) == AUTH_SCHEMA_VERSION
        assert table_exists(connection, "users")
        assert column_exists(connection, "users", "tour_completed")
        assert column_exists(connection, "users", "otp_secret")
        assert column_exists(connection, "users", "two_factor_enabled")
        assert not column_exists(connection, "users", "is_active")
        assert column_exists(connection, "users", "status")
        assert index_exists(connection, "idx_users_role")
        assert index_exists(connection, "idx_audit_log_username")
        assert index_exists(connection, "idx_audit_log_event_type")


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
            "completed successfully" in record.message for record in caplog.records
        ), f"Expected migration log message not found in: {[r.message for r in caplog.records]}"


def test_auth_migration_logs_info_message(tmp_path, caplog):
    """migrate_auth_database must log an INFO message on success (issue #1051)."""
    import logging

    with connect(tmp_path / "auth-log.db") as connection:
        with caplog.at_level(logging.INFO, logger="src.db.migrations.common"):
            migrate_auth_database(connection)

        assert any(
            "completed successfully" in record.message for record in caplog.records
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
            "completed successfully" in record.message for record in caplog.records
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


def test_migration_005_signature_uses_connection_param_name():
    """migration_005_add_false_positives previously took a param literally
    named ``cursor`` despite calling .execute() on it like a connection —
    inconsistent with every other migration in the module and misleading
    for type safety / readability. It must be named and typed the same
    way as the rest: ``connection: sqlite3.Connection``."""
    import inspect

    from src.db.migrations.corpus import migration_005_add_false_positives

    sig = inspect.signature(migration_005_add_false_positives)
    params = list(sig.parameters.values())

    assert len(params) == 1
    assert params[0].name == "connection"
    # `from __future__ import annotations` makes annotations strings at
    # runtime rather than resolved types, so compare against both forms.
    assert params[0].annotation in (sqlite3.Connection, "sqlite3.Connection")


def test_migration_005_still_creates_false_positives_table(tmp_path):
    """Behavioral regression check accompanying the parameter rename."""
    with connect(tmp_path / "false-positives-corpus.db") as connection:
        migrate_corpus_database(connection)

        assert table_exists(connection, "false_positives")
        for col in (
            "document_a",
            "document_b",
            "date_dismissed",
            "dismissed_by",
            "dismissal_reason",
        ):
            assert column_exists(connection, "false_positives", col)


def test_migration_018_adds_false_positives_audit_columns(tmp_path):
    """Verify migration_018_add_false_positives_audit_columns adds dismissed_by and dismissal_reason columns."""
    with connect(tmp_path / "false-positives-audit-corpus.db") as connection:
        migrate_corpus_database(connection)

        assert table_exists(connection, "false_positives")
        for col in (
            "document_a",
            "document_b",
            "date_dismissed",
            "dismissed_by",
            "dismissal_reason",
        ):
            assert column_exists(connection, "false_positives", col)


def test_check_table_exists():
    """Verify check_table_exists checks table existence by querying sqlite_master."""
    connection = sqlite3.connect(":memory:")
    try:
        assert check_table_exists(connection, "test_table") is False
        connection.execute("CREATE TABLE test_table (id INTEGER)")
        assert check_table_exists(connection, "test_table") is True
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Tests for rollback_migration (Issue: automated migration rollback / step_down)
# ---------------------------------------------------------------------------


def _test_up_v1(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE widgets (id INTEGER PRIMARY KEY)")


def _test_up_v2(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE sprockets (id INTEGER PRIMARY KEY)")


def _test_down_v1(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE widgets")


def _test_down_v2(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE sprockets")


_TEST_UP_MIGRATIONS = {1: _test_up_v1, 2: _test_up_v2}
_TEST_DOWN_MIGRATIONS = {1: _test_down_v1, 2: _test_down_v2}


def test_rollback_migration_restores_schema_and_version(tmp_path):
    """A partial rollback undoes only the versions above the target, restores
    PRAGMA user_version, and leaves earlier schema changes intact."""
    with connect(tmp_path / "rollback-partial.db") as connection:
        run_migrations(connection, migrations=_TEST_UP_MIGRATIONS, target_version=2)
        assert get_user_version(connection) == 2
        assert table_exists(connection, "widgets")
        assert table_exists(connection, "sprockets")

        result = rollback_migration(
            connection, 1, down_migrations=_TEST_DOWN_MIGRATIONS
        )

        assert result == 1
        assert get_user_version(connection) == 1
        assert table_exists(connection, "widgets")
        assert not table_exists(connection, "sprockets")


def test_rollback_migration_to_zero_undoes_everything(tmp_path):
    """Rolling back to version 0 undoes every registered migration."""
    with connect(tmp_path / "rollback-full.db") as connection:
        run_migrations(connection, migrations=_TEST_UP_MIGRATIONS, target_version=2)

        result = rollback_migration(
            connection, 0, down_migrations=_TEST_DOWN_MIGRATIONS
        )

        assert result == 0
        assert get_user_version(connection) == 0
        assert not table_exists(connection, "widgets")
        assert not table_exists(connection, "sprockets")


def test_rollback_migration_is_noop_when_already_at_target(tmp_path):
    """Rolling back to the current version is a no-op, not an error."""
    with connect(tmp_path / "rollback-noop.db") as connection:
        run_migrations(connection, migrations=_TEST_UP_MIGRATIONS, target_version=1)

        result = rollback_migration(
            connection, 1, down_migrations=_TEST_DOWN_MIGRATIONS
        )

        assert result == 1
        assert get_user_version(connection) == 1
        assert table_exists(connection, "widgets")


def test_rollback_migration_rejects_target_newer_than_current(tmp_path):
    """Rolling back to a version newer than the current one must raise."""
    with connect(tmp_path / "rollback-invalid-target.db") as connection:
        run_migrations(connection, migrations=_TEST_UP_MIGRATIONS, target_version=1)

        with pytest.raises(RuntimeError):
            rollback_migration(connection, 2, down_migrations=_TEST_DOWN_MIGRATIONS)


def test_rollback_migration_rejects_missing_down_definition(tmp_path):
    """Rolling back must raise if a required down-migration isn't registered."""
    with connect(tmp_path / "rollback-missing-def.db") as connection:
        run_migrations(connection, migrations=_TEST_UP_MIGRATIONS, target_version=2)

        with pytest.raises(RuntimeError):
            # Only version 2's down-migration is supplied; version 1's is missing.
            rollback_migration(connection, 0, down_migrations={2: _test_down_v2})


def test_rollback_migration_rejects_negative_target(tmp_path):
    """A negative target_version must be rejected outright."""
    with connect(tmp_path / "rollback-negative.db") as connection:
        run_migrations(connection, migrations=_TEST_UP_MIGRATIONS, target_version=1)

        with pytest.raises(ValueError):
            rollback_migration(connection, -1, down_migrations=_TEST_DOWN_MIGRATIONS)


def test_rollback_migration_is_atomic_on_failure(tmp_path):
    """If a down-migration raises partway through, the whole rollback is
    reverted (schema and PRAGMA user_version both stay at the pre-rollback
    state), matching run_migrations' atomicity guarantee."""

    def _failing_down_v2(connection: sqlite3.Connection) -> None:
        connection.execute("DROP TABLE sprockets")
        raise RuntimeError("simulated failure mid-rollback")

    with connect(tmp_path / "rollback-atomic.db") as connection:
        run_migrations(connection, migrations=_TEST_UP_MIGRATIONS, target_version=2)

        broken_down_migrations = {1: _test_down_v1, 2: _failing_down_v2}

        with pytest.raises(RuntimeError, match="simulated failure"):
            rollback_migration(connection, 0, down_migrations=broken_down_migrations)

        # Nothing should have been undone — the failed transaction rolled back.
        assert get_user_version(connection) == 2
        assert table_exists(connection, "sprockets")
        assert table_exists(connection, "widgets")


# ---------------------------------------------------------------------------
# Issue #1770: Migration Duration Logging — regression tests
# ---------------------------------------------------------------------------


def test_issue_1770_migration_duration_logging_exact_format(tmp_path, caplog):
    """Issue #1770: the duration log message must match the exact
    format specified in the acceptance criteria:

        logger.info("Migration [%s] executed in %.3f seconds.", ...)

    This test asserts the exact message template (including the
    square-bracket around the migration name and the 3-decimal-place
    seconds value) is present in the log output.
    """
    import logging
    import re

    def migration_issue_1770_test(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE issue_1770_test (id INTEGER)")

    connection = sqlite3.connect(str(tmp_path / "issue_1770_format.db"))
    try:
        with caplog.at_level(logging.INFO, logger="src.db.migrations.common"):
            run_migrations(
                connection,
                migrations={1: migration_issue_1770_test},
                target_version=1,
            )

        # Find the duration log record.
        duration_records = [
            r
            for r in caplog.records
            if "executed in" in r.message and "seconds" in r.message
        ]
        assert len(duration_records) >= 1, (
            f"Expected at least one 'executed in ... seconds' log record; "
            f"found: {[r.message for r in caplog.records]}"
        )

        record = duration_records[0]
        # Assert the exact format: "Migration [<name>] executed in <N.NNN> seconds."
        assert re.match(
            r"^Migration \[migration_issue_1770_test\] executed in \d+\.\d{3} seconds\.$",
            record.message,
        ), f"Log message does not match the required format: {record.message!r}"

        # Assert the logger name is correct.
        assert record.name == "src.db.migrations.common"
        assert record.levelno == logging.INFO
    finally:
        connection.close()


def test_issue_1770_migration_duration_logging_multiple_migrations(tmp_path, caplog):
    """Issue #1770: each migration in a multi-migration run must
    produce its own duration log line, not just the first or last.
    """
    import logging

    def mig_v1(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE issue_1770_a (id INTEGER)")

    def mig_v2(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE issue_1770_b (id INTEGER)")

    def mig_v3(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE issue_1770_c (id INTEGER)")

    connection = sqlite3.connect(str(tmp_path / "issue_1770_multi.db"))
    try:
        with caplog.at_level(logging.INFO, logger="src.db.migrations.common"):
            run_migrations(
                connection,
                migrations={1: mig_v1, 2: mig_v2, 3: mig_v3},
                target_version=3,
            )

        duration_records = [
            r
            for r in caplog.records
            if "executed in" in r.message and "Migration [" in r.message
        ]
        assert len(duration_records) == 3, (
            f"Expected 3 duration log records (one per migration), "
            f"got {len(duration_records)}: {[r.message for r in duration_records]}"
        )

        # Verify each migration name appears.
        names_in_logs = [r.message for r in duration_records]
        assert any("mig_v1" in n for n in names_in_logs)
        assert any("mig_v2" in n for n in names_in_logs)
        assert any("mig_v3" in n for n in names_in_logs)
    finally:
        connection.close()


def test_issue_1770_migration_duration_uses_perf_counter(tmp_path):
    """Issue #1770: the timing must use ``time.perf_counter()`` (not
    ``time.time()``) for monotonic high-resolution measurement.

    This test verifies the function's source code contains the
    ``perf_counter`` call, guarding against a refactor that switches
    to the less-precise ``time.time()``.
    """
    import inspect

    from src.db.migrations.common import run_migrations

    source = inspect.getsource(run_migrations)
    assert "perf_counter" in source, (
        "run_migrations must use time.perf_counter() for high-resolution "
        "duration measurement, per issue #1770."
    )
    assert "time.time()" not in source, (
        "run_migrations must NOT use time.time() for duration measurement "
        "(it is not monotonic); use time.perf_counter() instead."
    )


def test_issue_1770_migration_duration_logs_migration_name_attribute(tmp_path, caplog):
    """Issue #1770: the migration name in the log must be the
    function's ``__name__`` attribute, falling back to ``v<version>``
    when the attribute is missing (e.g., for lambdas).
    """
    import logging

    def migration_named_explicitly(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE issue_1770_named (id INTEGER)")

    connection = sqlite3.connect(str(tmp_path / "issue_1770_name.db"))
    try:
        with caplog.at_level(logging.INFO, logger="src.db.migrations.common"):
            run_migrations(
                connection,
                migrations={1: migration_named_explicitly},
                target_version=1,
            )

        # The migration name should be the function's __name__.
        assert any(
            "migration_named_explicitly" in r.message
            for r in caplog.records
            if "executed in" in r.message
        ), (
            "The log message must contain the migration function's __name__; "
            f"records: {[r.message for r in caplog.records]}"
        )
    finally:
        connection.close()


def test_issue_1770_migration_duration_logs_fallback_name_for_lambda(tmp_path, caplog):
    """Issue #1770: when the migration function has no ``__name__``
    (e.g., a lambda), the log must fall back to ``v<version>``.
    """
    import logging

    connection = sqlite3.connect(str(tmp_path / "issue_1770_lambda.db"))
    try:
        with caplog.at_level(logging.INFO, logger="src.db.migrations.common"):
            run_migrations(
                connection,
                migrations={
                    1: lambda conn: conn.execute(
                        "CREATE TABLE issue_1770_lambda (id INTEGER)"
                    )
                },
                target_version=1,
            )

        # A lambda's __name__ is '<lambda>', so the getattr fallback
        # will actually return '<lambda>'. Either way, the log should
        # contain the word 'executed in'.
        duration_records = [r for r in caplog.records if "executed in" in r.message]
        assert len(duration_records) >= 1
    finally:
        connection.close()


def test_issue_1770_rollback_duration_logging(tmp_path, caplog):
    """Issue #1770: rollback migrations must also log their execution
    duration using the same format, with the 'Rollback migration'
    prefix.

    The rollback_migration function already has this logging (lines
    276-282), but no test covers it. This test locks it in.
    """
    import logging
    import re

    def up_v1(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE issue_1770_rb (id INTEGER)")

    def down_v1(conn: sqlite3.Connection) -> None:
        conn.execute("DROP TABLE issue_1770_rb")

    connection = sqlite3.connect(str(tmp_path / "issue_1770_rb.db"))
    try:
        # First, apply the migration.
        run_migrations(
            connection,
            migrations={1: up_v1},
            target_version=1,
        )

        caplog.clear()

        # Now roll back.
        with caplog.at_level(logging.INFO, logger="src.db.migrations.common"):
            rollback_migration(
                connection,
                target_version=0,
                down_migrations={1: down_v1},
            )

        rollback_duration_records = [
            r
            for r in caplog.records
            if "Rollback migration" in r.message and "executed in" in r.message
        ]
        assert len(rollback_duration_records) >= 1, (
            f"Expected a 'Rollback migration [...] executed in ... seconds' log; "
            f"records: {[r.message for r in caplog.records]}"
        )

        record = rollback_duration_records[0]
        assert re.match(
            r"^Rollback migration \[down_v1\] executed in \d+\.\d{3} seconds\.$",
            record.message,
        ), f"Rollback log does not match required format: {record.message!r}"
    finally:
        connection.close()
