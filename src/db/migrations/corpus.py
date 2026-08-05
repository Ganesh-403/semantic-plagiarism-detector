"""Versioned migrations for corpus.db."""

from __future__ import annotations

import sqlite3

from .common import column_exists, run_migrations

CORPUS_SCHEMA_VERSION = 13

def migration_001_create_base_schema(
    connection: sqlite3.Connection,
) -> None:
    """Create the original documents and chunks tables."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT UNIQUE NOT NULL,
            file_hash   TEXT UNIQUE NOT NULL,
            upload_date TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
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
        """
    )


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
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_upload_date
        ON documents(upload_date)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_class_section
        ON documents(class_section)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_filename
        ON chunks(filename)
        """
    )


def migration_004_add_plagiarism_incidents(
    connection: sqlite3.Connection,
) -> None:
    """Create the incident-review schema stored in corpus.db."""
    connection.execute(
        """
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
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_incidents_status
        ON plagiarism_incidents(review_status)
        """
    )


def migration_005_add_false_positives(cursor):
    """Adds a table to track dismissed false-positive plagiarism pairs."""
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS false_positives (
            document_a TEXT,
            document_b TEXT,
            date_dismissed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (document_a, document_b)
        )
    """
    )


def migration_006_add_incident_threshold_snapshot(
    connection: sqlite3.Connection,
) -> None:
    """Store the threshold that caused each incident to be flagged."""
    if not column_exists(connection, "plagiarism_incidents", "threshold_at_time_of_flag"):
        connection.execute(
            "ALTER TABLE plagiarism_incidents "
            "ADD COLUMN threshold_at_time_of_flag REAL NOT NULL DEFAULT 0.0"
        )


def migration_007_add_document_language(
    connection: sqlite3.Connection,
) -> None:
    """Store the primary detected language code of each document."""
    if not column_exists(connection, "documents", "detected_language"):
        connection.execute(
            "ALTER TABLE documents ADD COLUMN detected_language TEXT"
        )


def migration_008_add_soft_delete(
    connection: sqlite3.Connection,
) -> None:
    """Add is_deleted and deleted_at columns to documents for soft-delete support."""
    if not column_exists(connection, "documents", "is_deleted"):
        connection.execute(
            "ALTER TABLE documents ADD COLUMN is_deleted INTEGER DEFAULT 0"
        )
    if not column_exists(connection, "documents", "deleted_at"):
        connection.execute(
            "ALTER TABLE documents ADD COLUMN deleted_at TEXT"
        )


def migration_009_add_file_hash_index(
    connection: sqlite3.Connection,
) -> None:
    """Add index on the file_hash column in documents table."""
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_file_hash
        ON documents(file_hash)
        """
    )


def migration_010_add_document_owner(
    connection: sqlite3.Connection,
) -> None:
    """Add owner column to documents for per-user document counting (issue #1048).

    The ``owner`` column stores the username of the account that uploaded the
    document, enabling ``get_document_count_by_user()`` analytics without
    requiring a join against the users table.
    """
    if not column_exists(connection, "documents", "owner"):
        connection.execute(
            "ALTER TABLE documents ADD COLUMN owner TEXT"
        )
    # Index for fast per-user COUNT queries
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_owner
        ON documents(owner)
        """
    )


def migration_011_add_documents_created_at_index(
    connection: sqlite3.Connection,
) -> None:
    """Add created_at column and its index to documents table to optimize query performance."""
    if not column_exists(connection, "documents", "created_at"):
        connection.execute(
            "ALTER TABLE documents ADD COLUMN created_at TEXT"
        )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_created_at
        ON documents(created_at)
        """
    )

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
    connection.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            filename,
            student_name,
            assignment_title,
            content='documents',
            content_rowid='id'
        )
        """
    )

    # Trigger: after INSERT into documents, insert into FTS
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, filename, student_name, assignment_title)
            VALUES (new.id, new.filename, new.student_name, new.assignment_title);
        END
        """
    )

    # Trigger: after DELETE from documents, delete from FTS
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, filename, student_name, assignment_title)
            VALUES ('delete', old.id, old.filename, old.student_name, old.assignment_title);
        END
        """
    )

    # Trigger: after UPDATE on documents, update FTS (delete old + insert new)
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, filename, student_name, assignment_title)
            VALUES ('delete', old.id, old.filename, old.student_name, old.assignment_title);
            INSERT INTO documents_fts(rowid, filename, student_name, assignment_title)
            VALUES (new.id, new.filename, new.student_name, new.assignment_title);
        END
        """
    )

    # Backfill existing rows into the FTS index (for databases that already
    # have documents before this migration runs).
    connection.execute(
        """
        INSERT INTO documents_fts(documents_fts)
        VALUES ('rebuild')
        """
    )

def migration_013_add_incident_severity_idx(
    connection: sqlite3.Connection,
) -> None:
    """Add index on severity_rank and date_flagged to speed up
    severity-filtered incident analytics queries (issue #1487)."""
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_incidents_severity_time
        ON plagiarism_incidents(severity_rank, date_flagged DESC)
        """
    )


def migration_013_add_incident_archive_table(
    connection: sqlite3.Connection,
) -> None:
    """Create incidents_archive table for archived plagiarism incidents
    (issue #1492)."""
    connection.execute(
        """
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
        """
    )


CORPUS_MIGRATIONS = {   1: migration_001_create_base_schema,
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
    13: migration_013_add_incident_archive_table,    13: migration_013_add_incident_severity_idx,
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
