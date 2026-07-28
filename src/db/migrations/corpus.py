"""Versioned migrations for corpus.db."""

from __future__ import annotations

import sqlite3

from .common import column_exists, run_migrations

CORPUS_SCHEMA_VERSION = 7


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



def migration_007_add_soft_delete(
    connection: sqlite3.Connection,
) -> None:
    """Add is_deleted and deleted_at columns to documents, and create deleted_chunks table."""
    if not column_exists(connection, "documents", "is_deleted"):
        connection.execute(
            "ALTER TABLE documents ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0"
        )
    if not column_exists(connection, "documents", "deleted_at"):
        connection.execute(
            "ALTER TABLE documents ADD COLUMN deleted_at TEXT"
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS deleted_chunks (
            vector_id   INTEGER,
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


CORPUS_MIGRATIONS = {
    1: migration_001_create_base_schema,
    2: migration_002_add_document_metadata,
    3: migration_003_add_required_indexes,
    4: migration_004_add_plagiarism_incidents,
    5: migration_005_add_false_positives,
    6: migration_006_add_incident_threshold_snapshot,
    7: migration_007_add_soft_delete,
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
