"""Integration test suite for SQLite database migration flows.

Verifies that migrate_corpus_database() and migrate_auth_database()
upgrade an in-memory database incrementally from version 1 to each
successive version, asserting the correct tables, columns, and indexes
exist at every step of the schema history.
"""

from __future__ import annotations

import sqlite3

from src.db.migrations import (
    AUTH_MIGRATIONS,
    AUTH_SCHEMA_VERSION,
    CORPUS_MIGRATIONS,
    CORPUS_SCHEMA_VERSION,
    column_exists,
    get_user_version,
    index_exists,
    run_migrations,
    table_exists,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    """Open a fresh in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _apply_up_to(conn: sqlite3.Connection, migrations: dict, version: int) -> int:
    """Run migrations up to *version* only (subset of the full map)."""
    subset = {v: fn for v, fn in migrations.items() if v <= version}
    return run_migrations(conn, migrations=subset, target_version=version)


# ---------------------------------------------------------------------------
# Corpus migration flow — step-by-step assertions
# ---------------------------------------------------------------------------


class TestCorpusMigrationFlow:
    """Incremental integration tests for corpus.db schema upgrades."""

    def test_version_1_creates_base_tables(self):
        conn = _connect()
        try:
            _apply_up_to(conn, CORPUS_MIGRATIONS, 1)
            assert get_user_version(conn) == 1
            assert table_exists(conn, "documents")
            assert table_exists(conn, "chunks")
            # Core columns on documents
            for col in ("id", "filename", "file_hash", "upload_date"):
                assert column_exists(conn, "documents", col), f"Missing column: {col}"
            # Core columns on chunks
            for col in (
                "vector_id",
                "filename",
                "chunk_index",
                "chunk_text",
                "embedding",
            ):
                assert column_exists(conn, "chunks", col), f"Missing column: {col}"
        finally:
            conn.close()

    def test_version_2_adds_document_metadata_columns(self):
        conn = _connect()
        try:
            _apply_up_to(conn, CORPUS_MIGRATIONS, 2)
            assert get_user_version(conn) == 2
            for col in ("class_section", "student_name", "assignment_title"):
                assert column_exists(conn, "documents", col), f"Missing column: {col}"
        finally:
            conn.close()

    def test_version_3_adds_required_indexes(self):
        conn = _connect()
        try:
            _apply_up_to(conn, CORPUS_MIGRATIONS, 3)
            assert get_user_version(conn) == 3
            assert index_exists(conn, "idx_documents_upload_date")
            assert index_exists(conn, "idx_documents_class_section")
            assert index_exists(conn, "idx_chunks_filename")
        finally:
            conn.close()

    def test_version_4_creates_plagiarism_incidents_table(self):
        conn = _connect()
        try:
            _apply_up_to(conn, CORPUS_MIGRATIONS, 4)
            assert get_user_version(conn) == 4
            assert table_exists(conn, "plagiarism_incidents")
            for col in (
                "incident_id",
                "document_a",
                "document_b",
                "similarity_score",
                "severity_rank",
                "review_status",
                "date_flagged",
                "last_seen",
            ):
                assert column_exists(conn, "plagiarism_incidents", col), (
                    f"Missing: {col}"
                )
            assert index_exists(conn, "idx_incidents_status")
        finally:
            conn.close()

    def test_version_5_creates_false_positives_table(self):
        conn = _connect()
        try:
            _apply_up_to(conn, CORPUS_MIGRATIONS, 5)
            assert get_user_version(conn) == 5
            assert table_exists(conn, "false_positives")
            for col in ("document_a", "document_b", "date_dismissed"):
                assert column_exists(conn, "false_positives", col), f"Missing: {col}"
        finally:
            conn.close()

    def test_version_6_adds_threshold_snapshot_column(self):
        conn = _connect()
        try:
            _apply_up_to(conn, CORPUS_MIGRATIONS, 6)
            assert get_user_version(conn) == 6
            assert column_exists(
                conn, "plagiarism_incidents", "threshold_at_time_of_flag"
            )
        finally:
            conn.close()

    def test_version_7_adds_detected_language_column(self):
        conn = _connect()
        try:
            _apply_up_to(conn, CORPUS_MIGRATIONS, 7)
            assert get_user_version(conn) == 7
            assert column_exists(conn, "documents", "detected_language")
        finally:
            conn.close()

    def test_version_8_adds_soft_delete_columns(self):
        conn = _connect()
        try:
            _apply_up_to(conn, CORPUS_MIGRATIONS, 8)
            assert get_user_version(conn) == 8
            assert column_exists(conn, "documents", "is_deleted")
            assert column_exists(conn, "documents", "deleted_at")
        finally:
            conn.close()

    def test_version_9_adds_file_hash_index(self):
        conn = _connect()
        try:
            _apply_up_to(conn, CORPUS_MIGRATIONS, 9)
            assert get_user_version(conn) == 9
            assert index_exists(conn, "idx_documents_file_hash")
        finally:
            conn.close()

    def test_version_10_adds_document_owner(self):
        conn = _connect()
        try:
            _apply_up_to(conn, CORPUS_MIGRATIONS, 10)
            assert get_user_version(conn) == 10
            assert column_exists(conn, "documents", "owner")
            assert index_exists(conn, "idx_documents_owner")
        finally:
            conn.close()

    def test_version_10_backfills_default_owner_for_preexisting_rows(self):
        """Documents inserted before migration 10 ever ran must not end up
        with a NULL owner -- ALTER TABLE ... DEFAULT 'system' backfills
        every pre-existing row at migration time (see acceptance criteria
        of the 'owner column missing DEFAULT' issue)."""
        conn = _connect()
        try:
            _apply_up_to(conn, CORPUS_MIGRATIONS, 9)
            conn.execute(
                "INSERT INTO documents (filename, file_hash, upload_date) "
                "VALUES (?, ?, ?)",
                ("legacy_doc.pdf", "legacy_hash", "2023-01-01T00:00:00"),
            )
            conn.commit()

            _apply_up_to(conn, CORPUS_MIGRATIONS, 10)

            row = conn.execute(
                "SELECT owner FROM documents WHERE filename = ?",
                ("legacy_doc.pdf",),
            ).fetchone()
            assert row[0] == "system"
        finally:
            conn.close()

    def test_version_11_adds_documents_created_at_index(self):
        conn = _connect()
        try:
            _apply_up_to(conn, CORPUS_MIGRATIONS, 11)
            assert get_user_version(conn) == 11
            assert column_exists(conn, "documents", "created_at")
            assert index_exists(conn, "idx_documents_created_at")
        finally:
            conn.close()

    def test_full_corpus_flow_reaches_latest_version(self):
        """Single end-to-end pass: v0 → CORPUS_SCHEMA_VERSION via migrate_corpus_database."""
        conn = _connect()
        try:
            from src.db.migrations import migrate_corpus_database

            version = migrate_corpus_database(conn)

            assert version == CORPUS_SCHEMA_VERSION
            assert get_user_version(conn) == CORPUS_SCHEMA_VERSION

            # Tables
            for tbl in (
                "documents",
                "chunks",
                "plagiarism_incidents",
                "false_positives",
            ):
                assert table_exists(conn, tbl), f"Missing table: {tbl}"

            # All expected columns on documents
            for col in (
                "id",
                "filename",
                "file_hash",
                "upload_date",
                "class_section",
                "student_name",
                "assignment_title",
                "detected_language",
                "owner",
                "created_at",
            ):
                assert column_exists(conn, "documents", col), f"Missing column: {col}"

            # plagiarism_incidents final shape
            assert column_exists(
                conn, "plagiarism_incidents", "threshold_at_time_of_flag"
            )

            # Indexes
            for idx in (
                "idx_documents_upload_date",
                "idx_documents_class_section",
                "idx_chunks_filename",
                "idx_incidents_status",
                "idx_documents_file_hash",
                "idx_documents_owner",
                "idx_documents_created_at",
            ):
                assert index_exists(conn, idx), f"Missing index: {idx}"
        finally:
            conn.close()

    def test_corpus_flow_preserves_existing_rows(self):
        """Rows inserted at v1 survive the full upgrade to the latest version."""
        conn = _connect()
        try:
            _apply_up_to(conn, CORPUS_MIGRATIONS, 1)
            conn.execute(
                "INSERT INTO documents (filename, file_hash, upload_date) VALUES (?, ?, ?)",
                ("sample.pdf", "abc123", "2026-01-01T00:00:00"),
            )
            conn.execute(
                "INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (1, "sample.pdf", 0, "hello world", b"\x00\x01"),
            )
            conn.commit()

            run_migrations(
                conn, migrations=CORPUS_MIGRATIONS, target_version=CORPUS_SCHEMA_VERSION
            )

            row = conn.execute(
                "SELECT filename, file_hash, upload_date FROM documents WHERE filename = ?",
                ("sample.pdf",),
            ).fetchone()
            assert row == ("sample.pdf", "abc123", "2026-01-01T00:00:00")

            chunk = conn.execute(
                "SELECT chunk_text FROM chunks WHERE filename = ?",
                ("sample.pdf",),
            ).fetchone()
            assert chunk == ("hello world",)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Auth migration flow — step-by-step assertions
# ---------------------------------------------------------------------------


class TestAuthMigrationFlow:
    """Incremental integration tests for users.db schema upgrades."""

    def test_version_1_creates_users_table(self):
        conn = _connect()
        try:
            _apply_up_to(conn, AUTH_MIGRATIONS, 1)
            assert get_user_version(conn) == 1
            assert table_exists(conn, "users")
            for col in ("id", "username", "password", "role"):
                assert column_exists(conn, "users", col), f"Missing column: {col}"
        finally:
            conn.close()

    def test_version_2_adds_tour_completed(self):
        conn = _connect()
        try:
            _apply_up_to(conn, AUTH_MIGRATIONS, 2)
            assert get_user_version(conn) == 2
            assert column_exists(conn, "users", "tour_completed")
        finally:
            conn.close()

    def test_version_3_adds_two_factor_fields(self):
        conn = _connect()
        try:
            _apply_up_to(conn, AUTH_MIGRATIONS, 3)
            assert get_user_version(conn) == 3
            assert column_exists(conn, "users", "otp_secret")
            assert column_exists(conn, "users", "two_factor_enabled")
        finally:
            conn.close()

    def test_version_4_adds_role_index(self):
        conn = _connect()
        try:
            _apply_up_to(conn, AUTH_MIGRATIONS, 4)
            assert get_user_version(conn) == 4
            assert index_exists(conn, "idx_users_role")
        finally:
            conn.close()

    def test_version_5_adds_preferences_column(self):
        conn = _connect()
        try:
            _apply_up_to(conn, AUTH_MIGRATIONS, 5)
            assert get_user_version(conn) == 5
            assert column_exists(conn, "users", "preferences")
        finally:
            conn.close()

    def test_version_6_adds_is_active_flag(self):
        conn = _connect()
        try:
            _apply_up_to(conn, AUTH_MIGRATIONS, 6)
            assert get_user_version(conn) == 6
            assert column_exists(conn, "users", "is_active")
        finally:
            conn.close()

    def test_version_7_adds_theme_preference(self):
        conn = _connect()
        try:
            _apply_up_to(conn, AUTH_MIGRATIONS, 7)
            assert get_user_version(conn) == 7
            assert column_exists(conn, "users", "theme")
        finally:
            conn.close()

    def test_version_8_creates_security_audit_log(self):
        conn = _connect()
        try:
            _apply_up_to(conn, AUTH_MIGRATIONS, 8)
            assert get_user_version(conn) == 8
            assert table_exists(conn, "security_audit_log")
            for col in ("id", "event_type", "username", "timestamp", "details"):
                assert column_exists(conn, "security_audit_log", col), f"Missing: {col}"
            assert index_exists(conn, "idx_audit_log_username")
            assert index_exists(conn, "idx_audit_log_event_type")
        finally:
            conn.close()

    def test_version_16_adds_audit_log_indexes(self):
        conn = _connect()
        try:
            # We apply up to version 16 (includes migration 8 creating log and version 16 creating indexes)
            _apply_up_to(conn, AUTH_MIGRATIONS, 16)
            assert get_user_version(conn) == 16
            assert index_exists(conn, "idx_audit_log_username")
            assert index_exists(conn, "idx_audit_log_event_type")
        finally:
            conn.close()

    def test_version_17_drops_is_active(self):
        conn = _connect()
        try:
            _apply_up_to(conn, AUTH_MIGRATIONS, 17)
            assert get_user_version(conn) == 17
            assert not column_exists(conn, "users", "is_active")
            assert column_exists(conn, "users", "status")
        finally:
            conn.close()

    def test_full_auth_flow_reaches_latest_version(self):
        """Single end-to-end pass: v0 → AUTH_SCHEMA_VERSION via migrate_auth_database."""
        conn = _connect()
        try:
            from src.db.migrations import migrate_auth_database

            version = migrate_auth_database(conn)

            assert version == AUTH_SCHEMA_VERSION
            assert get_user_version(conn) == AUTH_SCHEMA_VERSION

            assert table_exists(conn, "users")
            assert table_exists(conn, "security_audit_log")

            for col in (
                "id",
                "username",
                "password",
                "role",
                "tour_completed",
                "otp_secret",
                "two_factor_enabled",
                "preferences",
                "theme",
                "last_login_at",
                "password_changed_at",
                "version",
                "status",
                "must_change_password",
            ):
                assert column_exists(conn, "users", col), f"Missing column: {col}"

            assert not column_exists(conn, "users", "is_active"), "is_active column should be dropped"

            assert index_exists(conn, "idx_users_role")
            assert index_exists(conn, "idx_audit_log_username")
            assert index_exists(conn, "idx_audit_log_event_type")
        finally:
            conn.close()

    def test_auth_flow_preserves_existing_rows(self):
        """Rows inserted at v1 survive the full upgrade to the latest version."""
        conn = _connect()
        try:
            _apply_up_to(conn, AUTH_MIGRATIONS, 1)
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ("alice", "hashed-pw", "teacher"),
            )
            conn.commit()

            run_migrations(
                conn, migrations=AUTH_MIGRATIONS, target_version=AUTH_SCHEMA_VERSION
            )

            row = conn.execute(
                "SELECT username, password, role FROM users WHERE username = ?",
                ("alice",),
            ).fetchone()
            assert row == ("alice", "hashed-pw", "teacher")
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Cross-cutting: column defaults after migration
# ---------------------------------------------------------------------------


class TestMigrationColumnDefaults:
    """Verify that migrated columns carry the correct default values."""

    def test_corpus_detected_language_defaults_to_null(self):
        conn = _connect()
        try:
            from src.db.migrations import migrate_corpus_database

            migrate_corpus_database(conn)
            conn.execute(
                "INSERT INTO documents (filename, file_hash, upload_date) VALUES (?, ?, ?)",
                ("test.pdf", "hash-xyz", "2026-07-01T00:00:00"),
            )
            conn.commit()
            row = conn.execute(
                "SELECT detected_language FROM documents WHERE filename = ?",
                ("test.pdf",),
            ).fetchone()
            assert row == (None,)
        finally:
            conn.close()

    def test_auth_tour_completed_defaults_to_zero(self):
        conn = _connect()
        try:
            from src.db.migrations import migrate_auth_database

            migrate_auth_database(conn)
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ("bob", "pw-hash", "teacher"),
            )
            conn.commit()
            row = conn.execute(
                "SELECT tour_completed FROM users WHERE username = ?",
                ("bob",),
            ).fetchone()
            assert row == (0,)
        finally:
            conn.close()

    def test_auth_status_defaults_to_active(self):
        conn = _connect()
        try:
            from src.db.migrations import migrate_auth_database

            migrate_auth_database(conn)
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ("carol", "pw-hash", "admin"),
            )
            conn.commit()
            row = conn.execute(
                "SELECT status FROM users WHERE username = ?",
                ("carol",),
            ).fetchone()
            assert row == ("active",)
        finally:
            conn.close()

    def test_auth_theme_defaults_to_light(self):
        conn = _connect()
        try:
            from src.db.migrations import migrate_auth_database

            migrate_auth_database(conn)
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ("dave", "pw-hash", "teacher"),
            )
            conn.commit()
            row = conn.execute(
                "SELECT theme FROM users WHERE username = ?",
                ("dave",),
            ).fetchone()
            assert row == ("light",)
        finally:
            conn.close()

    def test_incidents_threshold_defaults_to_zero(self):
        conn = _connect()
        try:
            from src.db.migrations import migrate_corpus_database

            migrate_corpus_database(conn)
            conn.execute(
                "INSERT INTO documents (filename, file_hash, upload_date) VALUES (?, ?, ?)",
                ("doc_a.pdf", "ha", "2026-01-01"),
            )
            conn.execute(
                "INSERT INTO documents (filename, file_hash, upload_date) VALUES (?, ?, ?)",
                ("doc_b.pdf", "hb", "2026-01-01"),
            )
            conn.execute(
                """
                INSERT INTO plagiarism_incidents
                    (incident_id, document_a, document_b, similarity_score,
                     severity_rank, review_status, date_flagged, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "inc-001",
                    "doc_a.pdf",
                    "doc_b.pdf",
                    0.95,
                    "High",
                    "Pending",
                    "2026-07-01",
                    "2026-07-01",
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT threshold_at_time_of_flag FROM plagiarism_incidents WHERE incident_id = ?",
                ("inc-001",),
            ).fetchone()
            assert row == (0.0,)
        finally:
            conn.close()


def test_auth_migrations_auto_discovered():
    """Verify that AUTH_MIGRATIONS is dynamically discovered and sorted by numeric prefix (Issue #2999)."""
    import inspect
    from src.db.migrations import auth

    funcs = [
        f
        for name, f in inspect.getmembers(auth, inspect.isfunction)
        if name.startswith("migration_")
    ]
    assert len(auth.AUTH_MIGRATIONS) == len(funcs)
    assert auth.AUTH_SCHEMA_VERSION == len(funcs)

    for i in range(1, len(funcs) + 1):
        assert i in auth.AUTH_MIGRATIONS
        assert callable(auth.AUTH_MIGRATIONS[i])
        assert auth.AUTH_MIGRATIONS[i].__name__.startswith(f"migration_{i:03d}")

