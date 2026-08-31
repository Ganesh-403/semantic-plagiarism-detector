"""Versioned migrations for corpus.db."""

from __future__ import annotations

import sqlite3

from .common import column_exists, run_migrations, table_exists

CORPUS_SCHEMA_VERSION = 20


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
                CHECK (review_status IN ('Pending', 'Resolved', 'Dismissed')),
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


def migration_016_add_scheduler_runs(
    connection: sqlite3.Connection,
) -> None:
    """Create the scheduler_runs table.

    Tracks the last-completed run of background scheduled jobs (e.g. the
    scheduled plagiarism rescan job) so a process restart does not lose track
    of when a job last ran. Keyed by job_name.
    """
    connection.execute("""
        CREATE TABLE IF NOT EXISTS scheduler_runs (
            job_name           TEXT PRIMARY KEY,
            last_run_at        TEXT NOT NULL,
            documents_scanned  INTEGER NOT NULL DEFAULT 0,
            new_incidents      INTEGER NOT NULL DEFAULT 0
        )
        """)


def migration_017_add_incident_date_flagged_index(
    connection: sqlite3.Connection,
) -> None:
    """Index plagiarism_incidents(date_flagged) for faster get_recent_incidents queries."""
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_incidents_date_flagged "
        "ON plagiarism_incidents(date_flagged)"
    )


def migration_018_add_false_positives_audit_columns(
    connection: sqlite3.Connection,
) -> None:
    """Add dismissed_by and dismissal_reason audit columns to false_positives table."""
    if not column_exists(connection, "false_positives", "dismissed_by"):
        connection.execute(
            "ALTER TABLE false_positives ADD COLUMN dismissed_by TEXT DEFAULT 'admin'"
        )
    if not column_exists(connection, "false_positives", "dismissal_reason"):
        connection.execute(
            "ALTER TABLE false_positives ADD COLUMN dismissal_reason TEXT"
        )


def migration_019_add_times_flagged(
    connection: sqlite3.Connection,
) -> None:
    """Track how many times a recurring document pair has been re-flagged.

    Issue #3421: when a document pair is flagged again in a later scan, the
    existing incident row is updated in place (see the ON CONFLICT clause in
    sync_flagged_incidents) rather than creating a duplicate row. This adds
    the counter that tracks how many times that has happened.
    """
    if not column_exists(connection, "plagiarism_incidents", "times_flagged"):
        connection.execute(
            "ALTER TABLE plagiarism_incidents "
            "ADD COLUMN times_flagged INTEGER NOT NULL DEFAULT 1"
        )
def migration_020_add_embedding_metadata(
    connection: sqlite3.Connection,
) -> None:
    """Add explicit model/schema metadata to persisted embeddings."""
    columns = (
        ("model_identifier", "TEXT"),
        ("model_version", "TEXT"),
        ("embedding_dimension", "INTEGER"),
        ("normalization_strategy", "TEXT"),
        ("embedding_generated_at", "TEXT"),
        ("vector_schema_version", "INTEGER"),
    )

    for column_name, column_type in columns:
        if not column_exists(connection, "chunks", column_name):
            connection.execute(
                f'ALTER TABLE chunks ADD COLUMN "{column_name}" {column_type}'
            )

    deleted_columns = (
        ("model_identifier", "TEXT"),
        ("model_version", "TEXT"),
        ("embedding_dimension", "INTEGER"),
        ("normalization_strategy", "TEXT"),
        ("embedding_generated_at", "TEXT"),
        ("vector_schema_version", "INTEGER"),
    )

    for column_name, column_type in deleted_columns:
        if not column_exists(connection, "deleted_chunks", column_name):
            connection.execute(
                f'ALTER TABLE deleted_chunks ADD COLUMN "{column_name}" {column_type}'
            )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_model
        ON chunks(model_identifier, model_version)
        """
    )
    def migration_021_add_corpus_duplicate_detection(
    connection: sqlite3.Connection,
) -> None:
    """Create persistent corpus-level duplicate/fingerprint metadata."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS document_fingerprints (
            filename TEXT PRIMARY KEY,
            exact_hash TEXT NOT NULL,
            minhash_signature TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS corpus_duplicate_relationships (
            relationship_id TEXT PRIMARY KEY,
            document_a TEXT NOT NULL,
            document_b TEXT NOT NULL,
            relationship_type TEXT NOT NULL
                CHECK (
                    relationship_type IN (
                        'exact_duplicate',
                        'near_duplicate'
                    )
                ),
            similarity REAL NOT NULL,
            family_id TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            UNIQUE(document_a, document_b)
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_duplicate_relationships_a
        ON corpus_duplicate_relationships(document_a)
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_duplicate_relationships_b
        ON corpus_duplicate_relationships(document_b)
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_duplicate_relationships_family
        ON corpus_duplicate_relationships(family_id)
        """
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
    16: migration_016_add_scheduler_runs,
    17: migration_017_add_incident_date_flagged_index,
    18: migration_018_add_false_positives_audit_columns,
    19: migration_019_add_times_flagged,
    20: migration_020_add_embedding_metadata,
    21: migration_021_add_corpus_duplicate_detection,
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


def down_016_add_scheduler_runs(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS scheduler_runs")


def down_017_add_incident_date_flagged_index(connection: sqlite3.Connection) -> None:
    connection.execute("DROP INDEX IF EXISTS idx_incidents_date_flagged")


def down_018_add_false_positives_audit_columns(
    connection: sqlite3.Connection,
) -> None:
    """Remove dismissed_by and dismissal_reason audit columns from false_positives table."""
    _drop_column_if_exists(connection, "false_positives", "dismissed_by")
    _drop_column_if_exists(connection, "false_positives", "dismissal_reason")


def down_019_add_times_flagged(connection: sqlite3.Connection) -> None:
    _drop_column_if_exists(connection, "plagiarism_incidents", "times_flagged")

def down_020_add_embedding_metadata(
    connection: sqlite3.Connection,
) -> None:
    """Remove embedding metadata columns added by migration 020."""
    columns = (
        "model_identifier",
        "model_version",
        "embedding_dimension",
        "normalization_strategy",
        "embedding_generated_at",
        "vector_schema_version",
    )

    for table in ("chunks", "deleted_chunks"):
        for column_name in columns:
            _drop_column_if_exists(connection, table, column_name)

    connection.execute("DROP INDEX IF EXISTS idx_chunks_embedding_model")
    def down_020_add_corpus_duplicate_detection(
    connection: sqlite3.Connection,
) -> None:
    """Remove corpus duplicate-detection metadata."""
    connection.execute(
        "DROP INDEX IF EXISTS idx_duplicate_relationships_a"
    )
    connection.execute(
        "DROP INDEX IF EXISTS idx_duplicate_relationships_b"
    )
    connection.execute(
        "DROP INDEX IF EXISTS idx_duplicate_relationships_family"
    )
    connection.execute(
        "DROP TABLE IF EXISTS corpus_duplicate_relationships"
    )
    connection.execute(
        "DROP TABLE IF EXISTS document_fingerprints"
    )
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
    16: down_016_add_scheduler_runs,
    17: down_017_add_incident_date_flagged_index,
    18: down_018_add_false_positives_audit_columns,
    19: down_019_add_times_flagged,
    20: down_020_add_corpus_duplicate_detection,
}


def _corpus_db_file_path(connection: sqlite3.Connection) -> Path | None:
    """Return the on-disk path of the connection's "main" database.

    Returns ``None`` for in-memory or temporary databases (no file to
    back up).
    """
    for _, name, filename in connection.execute("PRAGMA database_list"):
        if name == "main" and filename:
            return Path(filename)
    return None


def migrate_corpus_database(
    connection: sqlite3.Connection,
) -> int:
    """Upgrade corpus.db to the latest supported schema version."""
    connection.execute("PRAGMA foreign_keys = ON")

    db_path = _corpus_db_file_path(connection)
    backup_path = db_path.with_name("corpus_pre_migrate.db.bak") if db_path else None

    if backup_path is not None:
        shutil.copy2(db_path, backup_path)

    try:
        return run_migrations(
            connection,
            migrations=CORPUS_MIGRATIONS,
            target_version=CORPUS_SCHEMA_VERSION,
        )
    except Exception:
        if backup_path is not None and backup_path.exists():
            logger.error(
                "Migration failed; restoring corpus.db from backup %s.",
                backup_path,
            )
            connection.close()
            shutil.copy2(backup_path, db_path)
        raise
