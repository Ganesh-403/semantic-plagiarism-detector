"""Versioned migrations for corpus.db."""

from __future__ import annotations

import sqlite3

from .common import column_exists, run_migrations, table_exists

CORPUS_SCHEMA_VERSION = 15


def migration_001_create_base_schema(
    connection: sqlite3.Connection,
) -> None:
    """Create the original documents and chunks tables."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT UNIQUE NOT NULL,
            file_hash   TEXT UNIQUE NOT NULL,
            upload_date TEXT NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            vector_id   INTEGER PRIMARY KEY,
            filename    TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text  TEXT NOT NULL,
            embedding   BLOB NOT NULL,
            FOREIGN KEY (filename)
                REFERENCES documents(filename)
                ON DELETE CASCADE
        )
        """)


def migration_002_add_document_metadata(
    connection: sqlite3.Connection,
) -> None:
    """Add optional assignment metadata without removing existing rows."""
    for column_name in (
        "class_section",
        "student_name",
        "assignment_title",
    ):
        if not column_exists(connection, "documents", column_name):
            connection.execute(f'ALTER TABLE documents ADD COLUMN "{column_name}" TEXT')


def migration_003_add_required_indexes(
    connection: sqlite3.Connection,
) -> None:
    """Add indexes used by corpus filtering and chunk lookups."""
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_upload_date
        ON documents(upload_date)
        """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_class_section
        ON documents(class_section)
        """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_filename
        ON chunks(filename)
        """)


def migration_004_add_plagiarism_incidents(
    connection: sqlite3.Connection,
) -> None:
    """Create the incident-review schema stored in corpus.db."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS plagiarism_incidents (
            incident_id TEXT PRIMARY KEY,
            document_a TEXT NOT NULL,
            document_b TEXT NOT NULL,
            similarity_score REAL NOT NULL,
            severity_rank TEXT NOT NULL,
            review_status TEXT NOT NULL DEFAULT 'Pending'
                CHECK (review_status IN ('Pending', 'Resolved')),
            date_flagged TEXT NOT NULL,
            last_seen TEXT NOT NULL
        )
        """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_incidents_status
        ON plagiarism_incidents(review_status)
        """)


def migration_005_add_false_positives(cursor):
    """Adds a table to track dismissed false-positive plagiarism pairs."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS false_positives (
            document_a TEXT,
            document_b TEXT,
            date_dismissed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (document_a, document_b)
        )
    """)


def migration_006_add_incident_threshold_snapshot(
    connection: sqlite3.Connection,
) -> None:
    """Store the threshold that caused each incident to be flagged."""
    if not column_exists(
        connection, "plagiarism_incidents", "threshold_at_time_of_flag"
    ):
        connection.execute(
            "ALTER TABLE plagiarism_incidents "
            "ADD COLUMN threshold_at_time_of_flag REAL NOT NULL DEFAULT 0.0"
        )


def migration_007_add_document_language(
    connection: sqlite3.Connection,
) -> None:
    """Store the primary detected language code of each document."""
    if not column_exists(connection, "documents", "detected_language"):
        connection.execute("ALTER TABLE documents ADD COLUMN detected_language TEXT")


def migration_008_add_soft_delete(
    connection: sqlite3.Connection,
) -> None:
    """Add is_deleted and deleted_at columns to documents for soft-delete support."""
    if not column_exists(connection, "documents", "is_deleted"):
        connection.execute(
            "ALTER TABLE documents ADD COLUMN is_deleted INTEGER DEFAULT 0"
        )
    if not column_exists(connection, "documents", "deleted_at"):
        connection.execute("ALTER TABLE documents ADD COLUMN deleted_at TEXT")


def migration_009_add_file_hash_index(
    connection: sqlite3.Connection,
) -> None:
    """Add index on the file_hash column in documents table."""
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_file_hash
        ON documents(file_hash)
        """)


def migration_010_add_document_owner(
    connection: sqlite3.Connection,
) -> None:
    """Add owner column to documents for per-user document counting (issue #1048).

    The ``owner`` column stores the username of the account that uploaded the
    document, enabling ``get_document_count_by_user()`` analytics without
    requiring a join against the users table.

    Uses ``DEFAULT 'system'`` rather than leaving new rows NULL: SQLite
    backfills that default into every pre-existing row at the moment this
    ``ALTER TABLE`` runs, so a database migrating from an older schema
    version won't end up with a mix of NULL and populated owners for its
    existing documents.

    Note this default only applies retroactively to rows that exist at
    migration time (and to any future INSERT that omits the ``owner``
    column entirely). ``add_document()`` always explicitly includes
    ``owner`` in its INSERT statement and will still pass through
    ``None`` -> SQL NULL for callers that don't supply one, so
    NULL owners remain a real, ongoing possibility going forward. See
    ``get_document_count_by_user()`` in ``src/db/corpus_db.py``, which is
    NULL-safe by construction (``WHERE owner = ?`` never matches a NULL
    row, so it simply excludes them rather than crashing).
    """
    if not column_exists(connection, "documents", "owner"):
        connection.execute(
            "ALTER TABLE documents ADD COLUMN owner TEXT DEFAULT 'system'"
        )
    # Index for fast per-user COUNT queries
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_owner
        ON documents(owner)
        """)


def migration_011_add_documents_created_at_index(
    connection: sqlite3.Connection,
) -> None:
    """Add created_at column and its index to documents table to optimize query performance."""
    if not column_exists(connection, "documents", "created_at"):
        connection.execute("ALTER TABLE documents ADD COLUMN created_at TEXT")
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_created_at
        ON documents(created_at)
        """)


def migration_012_add_fts5_index(
    connection: sqlite3.Connection,
) -> None:
    """Create FTS5 virtual table and sync triggers for full-text search (issue #1359).

    The FTS5 table ``documents_fts`` mirrors the ``filename``,
    ``student_name``, and ``assignment_title`` columns from ``documents``
    so users can keyword-search across the corpus without
    ``LIKE '%query%'`` full table scans.

    Three triggers keep the FTS index in sync:
      - ``documents_ai`` — after INSERT, insert into FTS
      - ``documents_ad`` — after DELETE, delete from FTS
      - ``documents_au`` — after UPDATE, delete+insert in FTS
    """
    # Create the FTS5 virtual table (external content table pointing at documents)
    connection.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            filename,
            student_name,
            assignment_title,
            content='documents',
            content_rowid='id'
        )
        """)

    # Trigger: after INSERT into documents, insert into FTS
    connection.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, filename, student_name, assignment_title)
            VALUES (new.id, new.filename, new.student_name, new.assignment_title);
        END
        """)

    # Trigger: after DELETE from documents, delete from FTS
    connection.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, filename, student_name, assignment_title)
            VALUES ('delete', old.id, old.filename, old.student_name, old.assignment_title);
        END
        """)

    # Trigger: after UPDATE on documents, update FTS (delete old + insert new)
    connection.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, filename, student_name, assignment_title)
            VALUES ('delete', old.id, old.filename, old.student_name, old.assignment_title);
            INSERT INTO documents_fts(rowid, filename, student_name, assignment_title)
            VALUES (new.id, new.filename, new.student_name, new.assignment_title);
        END
        """)

    # Backfill existing rows into the FTS index (for databases that already
    # have documents before this migration runs).
    connection.execute("""
        INSERT INTO documents_fts(documents_fts)
        VALUES ('rebuild')
        """)


def migration_013_add_incident_severity_idx(
    connection: sqlite3.Connection,
) -> None:
    """Add index on severity_rank and date_flagged to speed up
    severity-filtered incident analytics queries (issue #1487)."""
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_incidents_severity_time
        ON plagiarism_incidents(severity_rank, date_flagged DESC)
        """)


def migration_013_add_incident_archive_table(
    connection: sqlite3.Connection,
) -> None:
    """Create incidents_archive table for archived plagiarism incidents
    (issue #1492)."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS incidents_archive (
            incident_id TEXT PRIMARY KEY,
            document_a TEXT NOT NULL,
            document_b TEXT NOT NULL,
            similarity_score REAL NOT NULL,
            severity_rank TEXT NOT NULL,
            review_status TEXT NOT NULL,
            date_flagged TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            threshold_at_time_of_flag REAL NOT NULL DEFAULT 0.0,
            archived_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)


def migration_015_pattern_recognition(
    connection: sqlite3.Connection,
) -> None:
    """Create pattern recognition tables for the intelligent detection system (issue #2840)."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS plagiarism_patterns (
            pattern_id TEXT PRIMARY KEY,
            pattern_type TEXT NOT NULL,
            description TEXT,
            document_group TEXT NOT NULL,
            author_group TEXT,
            assignment_title TEXT,
            class_section TEXT,
            avg_similarity REAL NOT NULL,
            occurrence_count INTEGER NOT NULL DEFAULT 1,
            confidence_score REAL NOT NULL DEFAULT 0.0,
            severity TEXT NOT NULL DEFAULT 'Medium',
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
        )
    """)
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_patterns_type ON plagiarism_patterns(pattern_type)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_patterns_status ON plagiarism_patterns(status)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_patterns_severity ON plagiarism_patterns(severity)"
    )

    connection.execute("""
        CREATE TABLE IF NOT EXISTS pattern_evolution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            occurrence_count INTEGER NOT NULL,
            avg_similarity REAL NOT NULL,
            confidence_score REAL NOT NULL,
            drift_score REAL DEFAULT 0.0,
            FOREIGN KEY (pattern_id) REFERENCES plagiarism_patterns(pattern_id) ON DELETE CASCADE
        )
    """)
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_evolution_pattern ON pattern_evolution(pattern_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_evolution_date ON pattern_evolution(snapshot_date)"
    )

    if not table_exists(connection, "document_risk_scores"):
        connection.execute("""
            CREATE TABLE IF NOT EXISTS document_risk_scores (
                document_name TEXT PRIMARY KEY,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                contributing_factors TEXT,
                model_version TEXT,
                scored_at TEXT NOT NULL
            )
        """)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_risk_level ON document_risk_scores(risk_level)"
        )

    connection.execute("""
        CREATE TABLE IF NOT EXISTS proactive_recommendations (
            recommendation_id TEXT PRIMARY KEY,
            pattern_id TEXT,
            recommendation_type TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 2,
            target TEXT NOT NULL,
            message TEXT NOT NULL,
            action_items TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            FOREIGN KEY (pattern_id) REFERENCES plagiarism_patterns(pattern_id) ON DELETE SET NULL
        )
    """)
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_recommendations_status ON proactive_recommendations(status)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_recommendations_priority ON proactive_recommendations(priority)"
    )


CORPUS_MIGRATIONS = {
    1: migration_001_create_base_schema,
    2: migration_002_add_document_metadata,
    3: migration_003_add_required_indexes,
    4: migration_004_add_plagiarism_incidents,
    5: migration_005_add_false_positives,
    6: migration_006_add_incident_threshold_snapshot,
    7: migration_007_add_document_language,
    8: migration_008_add_soft_delete,
    9: migration_009_add_file_hash_index,
    10: migration_010_add_document_owner,
    11: migration_011_add_documents_created_at_index,
    12: migration_012_add_fts5_index,
    13: migration_013_add_incident_archive_table,
    14: migration_013_add_incident_severity_idx,
    15: migration_015_pattern_recognition,
}


def _drop_column_if_exists(
    connection: sqlite3.Connection, table_name: str, column_name: str
) -> None:
    if column_exists(connection, table_name, column_name):
        try:
            connection.execute(
                f'ALTER TABLE "{table_name}" DROP COLUMN "{column_name}"'
            )
        except sqlite3.OperationalError:
            pass


def down_001_create_base_schema(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS chunks")
    connection.execute("DROP TABLE IF EXISTS documents")


def down_002_add_document_metadata(connection: sqlite3.Connection) -> None:
    for col in ("class_section", "student_name", "assignment_title"):
        _drop_column_if_exists(connection, "documents", col)


def down_003_add_required_indexes(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_documents_upload_date")
    connection.execute("DROP INDEX IF EXISTS idx_documents_class_section")
    connection.execute("DROP INDEX IF EXISTS idx_chunks_filename")


def down_004_add_plagiarism_incidents(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_incidents_status")
    connection.execute("DROP TABLE IF EXISTS plagiarism_incidents")


def down_005_add_false_positives(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS false_positives")


def down_006_add_incident_threshold_snapshot(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(
        connection, "plagiarism_incidents", "threshold_at_time_of_flag"
    )


def down_007_add_document_language(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(connection, "documents", "detected_language")


def down_008_add_soft_delete(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(connection, "documents", "is_deleted")
    _drop_column_if_exists(connection, "documents", "deleted_at")


def down_009_add_file_hash_index(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_documents_file_hash")


def down_010_add_document_owner(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_documents_owner")
    _drop_column_if_exists(connection, "documents", "owner")


def down_011_add_documents_created_at_index(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_documents_created_at")
    _drop_column_if_exists(connection, "documents", "created_at")


def down_012_add_fts5_index(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TRIGGER IF EXISTS documents_ai")
    connection.execute("DROP TRIGGER IF EXISTS documents_ad")
    connection.execute("DROP TRIGGER IF EXISTS documents_au")
    connection.execute("DROP TABLE IF EXISTS documents_fts")


def down_013_add_incident_archive_table(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS incidents_archive")


def down_014_add_incident_severity_idx(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_incidents_severity_time")


def down_015_pattern_recognition(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_recommendations_priority")
    connection.execute("DROP INDEX IF EXISTS idx_recommendations_status")
    connection.execute("DROP TABLE IF EXISTS proactive_recommendations")
    connection.execute("DROP INDEX IF EXISTS idx_risk_level")
    connection.execute("DROP TABLE IF EXISTS document_risk_scores")
    connection.execute("DROP INDEX IF EXISTS idx_evolution_date")
    connection.execute("DROP INDEX IF EXISTS idx_evolution_pattern")
    connection.execute("DROP TABLE IF EXISTS pattern_evolution")
    connection.execute("DROP INDEX IF EXISTS idx_patterns_severity")
    connection.execute("DROP INDEX IF EXISTS idx_patterns_status")
    connection.execute("DROP INDEX IF EXISTS idx_patterns_type")
    connection.execute("DROP TABLE IF EXISTS plagiarism_patterns")


CORPUS_DOWN_MIGRATIONS = {
    1: down_001_create_base_schema,
    2: down_002_add_document_metadata,
    3: down_003_add_required_indexes,
    4: down_004_add_plagiarism_incidents,
    5: down_005_add_false_positives,
    6: down_006_add_incident_threshold_snapshot,
    7: down_007_add_document_language,
    8: down_008_add_soft_delete,
    9: down_009_add_file_hash_index,
    10: down_010_add_document_owner,
    11: down_011_add_documents_created_at_index,
    12: down_012_add_fts5_index,
    13: down_013_add_incident_archive_table,
    14: down_014_add_incident_severity_idx,
    15: down_015_pattern_recognition,
}


def migrate_corpus_database(
    connection: sqlite3.Connection,
) -> int:
    """Upgrade corpus.db to the latest supported schema version."""
    connection.execute("PRAGMA foreign_keys = ON")
    return run_migrations(
        connection,
        migrations=CORPUS_MIGRATIONS,
        target_version=CORPUS_SCHEMA_VERSION,
    )
