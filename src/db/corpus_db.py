"""
corpus_db.py
------------
SQLite database manager to persist document metadata, chunk text, and embeddings.
Enables incremental updates and index rebuilding without re-embedding.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import psutil

from src.core.app_config import CORPUS_DB_PATH, FALLBACK_CORPUS_DB_PATH
from src.core.concurrency import with_sqlite_retry
from src.db.base import BaseRepository
from src.db.migrations.common import column_exists, delete_all_if_table_exists
from src.utils.filename import sanitize_filename

logger = logging.getLogger(__name__)

# Seed the corpus DB path from the centralized app_config.
_DB_PATH = os.path.abspath(str(CORPUS_DB_PATH))

_connection_pool = threading.local()


class CorpusRepository(BaseRepository):
    """Data access repository for corpus documents, text chunks, and vector embeddings."""

    def __init__(self, db_path: str | os.PathLike = CORPUS_DB_PATH) -> None:
        super().__init__(db_path)

    def init_corpus_db(self) -> None:
        """Create or upgrade corpus.db without deleting persisted data."""
        init_corpus_db()


corpus_repo = CorpusRepository(_DB_PATH)


def configure_db_path(db_path: str | os.PathLike) -> None:
    """Configure the SQLite database path used by the corpus module."""
    global _DB_PATH
    close_connections()
    _DB_PATH = os.path.abspath(os.fspath(db_path))
    corpus_repo.configure_db_path(_DB_PATH)


def get_corpus_db_path() -> Path:
    """Return the configured corpus SQLite database path."""
    return Path(_DB_PATH)


def _pool() -> dict[str, sqlite3.Connection]:
    """Return the connection pool belonging to the current thread."""
    pool = getattr(_connection_pool, "connections", None)
    if pool is None:
        pool = {}
        _connection_pool.connections = pool
    return pool


@contextmanager
def _connect():
    """Open a connection for the duration of the operation and always close
    it on exit (success, error, or exception) to avoid leaked file handles
    under concurrent requests."""
    path = os.path.abspath(_DB_PATH)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except (OSError, PermissionError):
        path = str(FALLBACK_CORPUS_DB_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)

    pool = _pool()
    conn = pool.get(path)
    if conn is None:
        try:
            conn = sqlite3.connect(path, check_same_thread=False)
        except sqlite3.OperationalError:
            path = str(FALLBACK_CORPUS_DB_PATH)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode=WAL")
        pool[path] = conn

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def close_connections(all_threads: bool = False) -> None:
    """Close all pooled corpus connections for the current thread (or all threads if specified)."""
    pool = getattr(_connection_pool, "connections", {})
    for conn in pool.values():
        conn.close()
    pool.clear()


def init_corpus_db() -> None:
    """Create or upgrade corpus.db without deleting persisted data."""
    with _connect() as conn:
        # 1. ALWAYS CREATE TABLES FIRST
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                upload_date TEXT NOT NULL,
                class_section TEXT,
                student_name TEXT,
                assignment_title TEXT,
                pdf_author TEXT,
                pdf_creation_date TEXT,
                pdf_title TEXT,
                tags TEXT,
                detected_language TEXT,
                owner TEXT,
                is_deleted INTEGER DEFAULT 0,
                deleted_at TEXT,
                created_at TEXT
            )
            """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                vector_id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (filename)
                REFERENCES documents(filename)
                ON DELETE CASCADE
            )
            """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS deleted_chunks (
                vector_id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                deleted_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS plagiarism_incidents (
                incident_id TEXT PRIMARY KEY,
                document_a TEXT NOT NULL,
                document_b TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                severity_rank TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'Pending'
                    CHECK (review_status IN ('Pending', 'Resolved')),
                date_flagged TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                threshold_at_time_of_flag REAL DEFAULT 0.0
            )
            """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS false_positives (
                document_a TEXT,
                document_b TEXT,
                date_dismissed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (document_a, document_b)
            )
            """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                document_count INTEGER NOT NULL,
                avg_similarity REAL NOT NULL,
                max_similarity REAL NOT NULL,
                flagged_count INTEGER NOT NULL,
                threshold_used REAL NOT NULL
            )
            """)

        # 2. RUN SCHEMA MIGRATIONS / ALTER TABLES AFTER CREATION
        columns_to_ensure = [
            ("class_section", "TEXT"),
            ("student_name", "TEXT"),
            ("assignment_title", "TEXT"),
            ("pdf_author", "TEXT"),
            ("pdf_creation_date", "TEXT"),
            ("pdf_title", "TEXT"),
            ("tags", "TEXT"),
            ("detected_language", "TEXT"),
            ("owner", "TEXT"),
            ("is_deleted", "INTEGER DEFAULT 0"),
            ("deleted_at", "TEXT"),
            ("created_at", "TEXT"),
        ]

        for col_name, col_type in columns_to_ensure:
            if not column_exists(conn, "documents", col_name):
                conn.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_type}")

        if not column_exists(conn, "deleted_chunks", "deleted_at"):
            conn.execute(
                "ALTER TABLE deleted_chunks ADD COLUMN deleted_at TEXT DEFAULT CURRENT_TIMESTAMP"
            )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_incidents_date ON plagiarism_incidents(date_flagged)"
        )

        # Issue #1359: Create FTS5 virtual table + sync triggers for full-text
        # search. Also created by migration_012, but we create it here too
        # so that ``init_corpus_db()`` (which doesn't call
        # ``migrate_corpus_database()``) still sets up FTS.
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                filename,
                student_name,
                assignment_title,
                content='documents',
                content_rowid='id'
            )
            """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, filename, student_name, assignment_title)
                VALUES (new.id, new.filename, new.student_name, new.assignment_title);
            END
            """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, filename, student_name, assignment_title)
                VALUES ('delete', old.id, old.filename, old.student_name, old.assignment_title);
            END
            """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, filename, student_name, assignment_title)
                VALUES ('delete', old.id, old.filename, old.student_name, old.assignment_title);
                INSERT INTO documents_fts(rowid, filename, student_name, assignment_title)
                VALUES (new.id, new.filename, new.student_name, new.assignment_title);
            END
            """)
        # Backfill any existing rows into the FTS index
        try:
            conn.execute("INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')")
        except sqlite3.OperationalError:
            pass  # Table might be empty or already synced

        try:
            os.chmod(_DB_PATH, 0o600)
        except OSError:
            pass


@with_sqlite_retry
def add_document(
    filename: str,
    file_hash: str,
    class_section: str = None,
    student_name: str = None,
    assignment_title: str = None,
    pdf_author: str = None,
    pdf_creation_date: str = None,
    pdf_title: str = None,
    tags: str = None,
    detected_language: str = None,
    owner: str = None,
) -> int | None:
    """Insert a new document metadata row using parameterized execution."""
    filename = sanitize_filename(filename)

    try:
        with _connect() as conn:
            existing = conn.execute(
                """
                SELECT id
                FROM documents
                WHERE file_hash = ?
                  AND (is_deleted IS NULL OR is_deleted = 0)
                """,
                (file_hash,),
            ).fetchone()

            if existing:
                logger.info(
                    "Document %s already exists in corpus; skipping insertion.",
                    file_hash,
                )
                return existing[0]

            cursor = conn.execute(
                "INSERT INTO documents (filename, file_hash, upload_date, class_section, student_name, assignment_title, pdf_author, pdf_creation_date, pdf_title, tags, detected_language, owner) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    filename,
                    file_hash,
                    datetime.now().isoformat(),
                    class_section,
                    student_name,
                    assignment_title,
                    pdf_author,
                    pdf_creation_date,
                    pdf_title,
                    tags,
                    detected_language,
                    owner,
                ),
            )
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None


def get_document_by_hash(file_hash: str) -> str | None:
    """Check if a file with this hash is already indexed and return its filename."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT filename FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return row[0] if row else None


def get_all_documents(include_deleted: bool = False) -> list:
    """Return all indexed documents sorted by upload date descending."""
    from src.db.schemas import Document

    query = (
        "SELECT filename, file_hash, upload_date, class_section, student_name, "
        "assignment_title, pdf_author, pdf_creation_date, pdf_title, detected_language "
        "FROM documents"
    )
    if not include_deleted:
        query += " WHERE is_deleted IS NULL OR is_deleted = 0"
    query += " ORDER BY upload_date DESC"

    with _connect() as conn:
        rows = conn.execute(query).fetchall()
        return [
            Document(
                filename=r[0],
                file_hash=r[1],
                upload_date=r[2],
                class_section=r[3],
                student_name=r[4],
                assignment_title=r[5],
                pdf_author=r[6],
                pdf_creation_date=r[7],
                pdf_title=r[8],
                detected_language=r[9],
            )
            for r in rows
        ]


@with_sqlite_retry
def add_chunks(chunks_to_add: list) -> None:
    """Insert a batch of chunks with their raw text and embedded BLOBs."""
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)
    logger.info("Memory usage before batch chunk insertion: %.2f MB", mem_before)

    formatted_chunks = []
    for vid, fname, idx, text, emb in chunks_to_add:
        emb_blob = emb.astype(np.float32).tobytes()
        formatted_chunks.append((vid, fname, idx, text, emb_blob))

    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
            formatted_chunks,
        )

    mem_after = process.memory_info().rss / (1024 * 1024)
    logger.info("Memory usage after batch chunk insertion: %.2f MB", mem_after)


def get_chunk_registry() -> list:
    """Reconstructs the registry of ChunkRecord objects ordered by vector_id."""
    from src.core.faiss_index import ChunkRecord

    with _connect() as conn:
        rows = conn.execute(
            "SELECT filename, chunk_index, chunk_text FROM chunks ORDER BY vector_id ASC"
        ).fetchall()
        return [ChunkRecord(r[0], r[1], r[2]) for r in rows]


def get_all_embeddings() -> np.ndarray:
    """Load all chunk embeddings from the database to rebuild the FAISS index."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT embedding FROM chunks ORDER BY vector_id ASC"
        ).fetchall()

    if not rows:
        return np.empty((0, 384), dtype=np.float32)

    embeddings = [np.frombuffer(r[0], dtype=np.float32) for r in rows]
    return np.vstack(embeddings)


@with_sqlite_retry
def delete_document(filename: str) -> None:
    """Delete a document and all its associated chunks (cascade)."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM plagiarism_incidents WHERE document_a = ? OR document_b = ?",
            (filename, filename),
        )
        conn.execute(
            "DELETE FROM false_positives WHERE document_a = ? OR document_b = ?",
            (filename, filename),
        )
        conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        _compact_vector_ids()


@with_sqlite_retry
def soft_delete_document(filename: str) -> None:
    """Soft delete a document by setting is_deleted=1 and moving chunks to deleted_chunks."""
    with _connect() as conn:
        conn.execute(
            "UPDATE documents SET is_deleted = 1, deleted_at = ? WHERE filename = ?",
            (datetime.now().isoformat(), filename),
        )
        conn.execute(
            """
            INSERT INTO deleted_chunks (vector_id, filename, chunk_index, chunk_text, embedding)
            SELECT vector_id, filename, chunk_index, chunk_text, embedding
            FROM chunks
            WHERE filename = ?
            """,
            (filename,),
        )
        conn.execute("DELETE FROM chunks WHERE filename = ?", (filename,))
        _compact_vector_ids()


def get_deleted_documents() -> list:
    """Return all soft-deleted documents sorted by deleted_at descending."""
    from src.db.schemas import Document

    with _connect() as conn:
        rows = conn.execute(
            "SELECT filename, file_hash, upload_date, class_section, student_name, assignment_title, pdf_author, pdf_creation_date, pdf_title, deleted_at FROM documents WHERE is_deleted = 1 ORDER BY deleted_at DESC"
        ).fetchall()
        return [
            Document(
                filename=r[0],
                file_hash=r[1],
                upload_date=r[2],
                class_section=r[3],
                student_name=r[4],
                assignment_title=r[5],
                pdf_author=r[6],
                pdf_creation_date=r[7],
                pdf_title=r[8],
                deleted_at=r[9],
            )
            for r in rows
        ]


def get_deleted_documents_count() -> int:
    """Return the count of documents currently in the trash."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(1) FROM documents WHERE is_deleted = 1"
        ).fetchone()
        return row[0] if row else 0


@with_sqlite_retry
def restore_document(filename: str) -> None:
    """Restore a soft-deleted document by setting is_deleted=0 and moving chunks back."""
    with _connect() as conn:
        conn.execute(
            "UPDATE documents SET is_deleted = 0, deleted_at = NULL WHERE filename = ?",
            (filename,),
        )
        restored = conn.execute(
            "SELECT filename, chunk_index, chunk_text, embedding FROM deleted_chunks WHERE filename = ?",
            (filename,),
        ).fetchall()
        max_id_row = conn.execute(
            "SELECT COALESCE(MAX(vector_id), -1) FROM chunks"
        ).fetchone()
        next_id = max_id_row[0] + 1
        for i, row in enumerate(restored):
            conn.execute(
                "INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
                (next_id + i, row[0], row[1], row[2], row[3]),
            )
        conn.execute("DELETE FROM deleted_chunks WHERE filename = ?", (filename,))
        _compact_vector_ids()


@with_sqlite_retry
def permanently_delete_document(filename: str) -> None:
    """Permanently delete a document (alias to delete_document)."""
    delete_document(filename)


@with_sqlite_retry
def empty_trash() -> None:
    """Permanently delete all soft-deleted documents."""
    with _connect() as conn:
        deleted_docs = [
            r[0]
            for r in conn.execute(
                "SELECT filename FROM documents WHERE is_deleted = 1"
            ).fetchall()
        ]
        for filename in deleted_docs:
            conn.execute(
                "DELETE FROM plagiarism_incidents WHERE document_a = ? OR document_b = ?",
                (filename, filename),
            )
            conn.execute(
                "DELETE FROM false_positives WHERE document_a = ? OR document_b = ?",
                (filename, filename),
            )
        conn.execute("DELETE FROM documents WHERE is_deleted = 1")


@with_sqlite_retry
def batch_soft_delete_documents(doc_ids: list[int]) -> int:
    """
    Batch soft delete documents by updating their is_deleted flag and moving chunks to deleted_chunks.
    Returns the number of rows updated.
    """
    if not doc_ids:
        return 0

    placeholders = ",".join(["?"] * len(doc_ids))
    now = datetime.now().isoformat()

    with _connect() as conn:
        conn.execute(
            f"""
            INSERT INTO deleted_chunks (vector_id, filename, chunk_index, chunk_text, embedding)
            SELECT vector_id, filename, chunk_index, chunk_text, embedding
            FROM chunks
            WHERE filename IN (SELECT filename FROM documents WHERE id IN ({placeholders}))
            """,
            tuple(doc_ids),
        )
        conn.execute(
            f"""
            DELETE FROM chunks
            WHERE filename IN (SELECT filename FROM documents WHERE id IN ({placeholders}))
            """,
            tuple(doc_ids),
        )
        cursor = conn.execute(
            f"UPDATE documents SET is_deleted = 1, deleted_at = ? WHERE id IN ({placeholders})",
            (now, *doc_ids),
        )
        rowcount = cursor.rowcount

    _compact_vector_ids()
    return rowcount


@with_sqlite_retry
def batch_permanently_delete_documents(doc_ids: list[int]) -> int:
    """
    Batch permanently delete multiple document records in a single transaction.

    Hard-deletes the matching documents together with their chunks (removed via
    the ``chunks.filename`` ON DELETE CASCADE), any soft-deleted chunks, and any
    related plagiarism incident / false positive records.

    Args:
        doc_ids: List of document IDs to permanently delete.

    Returns:
        The total number of deleted document records.
    """
    if not doc_ids:
        return 0

    placeholders = ",".join(["?"] * len(doc_ids))
    doc_id_tuple = tuple(doc_ids)

    with _connect() as conn:
        # Resolve affected filenames before the documents are removed so related
        # records can be purged from tables without cascade constraints.
        filenames = [
            row[0]
            for row in conn.execute(
                f"SELECT filename FROM documents WHERE id IN ({placeholders})",
                doc_id_tuple,
            ).fetchall()
        ]

        for filename in filenames:
            conn.execute(
                "DELETE FROM plagiarism_incidents WHERE document_a = ? OR document_b = ?",
                (filename, filename),
            )
            conn.execute(
                "DELETE FROM false_positives WHERE document_a = ? OR document_b = ?",
                (filename, filename),
            )
            conn.execute("DELETE FROM deleted_chunks WHERE filename = ?", (filename,))

        cursor = conn.execute(
            f"DELETE FROM documents WHERE id IN ({placeholders})",
            doc_id_tuple,
        )
        rowcount = cursor.rowcount

    if filenames:
        _compact_vector_ids()
    return rowcount


@with_sqlite_retry
def _compact_vector_ids() -> None:
    """Re-index the vector_id column to remove any gaps left by deleted documents."""
    with _connect() as conn:
        chunks = conn.execute(
            "SELECT filename, chunk_index, chunk_text, embedding FROM chunks ORDER BY vector_id ASC"
        ).fetchall()

        conn.execute("DELETE FROM chunks")

        if chunks:
            formatted = [(i, r[0], r[1], r[2], r[3]) for i, r in enumerate(chunks)]
            conn.executemany(
                "INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
                formatted,
            )


def get_document_chunks_count(filename: str) -> int:
    """Return the number of chunks for a given document."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(1) FROM chunks WHERE filename = ?", (filename,)
        ).fetchone()
        return row[0] if row else 0


def get_document_word_counts() -> dict[str, int]:
    """Calculate and return the total word count for each document currently in the database."""
    import re

    with _connect() as conn:
        rows = conn.execute("SELECT filename, chunk_text FROM chunks").fetchall()

    word_counts = {}
    for filename, chunk_text in rows:
        words = len(re.findall(r"\b\w+\b", chunk_text or ""))
        word_counts[filename] = word_counts.get(filename, 0) + words
    return word_counts


def get_document_char_counts() -> dict[str, int]:
    """Calculate and return the total character count for each document currently in the database."""
    with _connect() as conn:
        rows = conn.execute("SELECT filename, chunk_text FROM chunks").fetchall()

    char_counts = {}
    for filename, chunk_text in rows:
        chars = len(chunk_text or "")
        char_counts[filename] = char_counts.get(filename, 0) + chars
    return char_counts


@with_sqlite_retry
def clear_all_data() -> None:
    """Clear known corpus tables while tolerating partial schemas."""
    with _connect() as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        delete_all_if_table_exists(conn, "chunks")
        delete_all_if_table_exists(conn, "deleted_chunks")
        delete_all_if_table_exists(conn, "documents")
        delete_all_if_table_exists(conn, "plagiarism_incidents")
        delete_all_if_table_exists(conn, "false_positives")


def get_unique_class_sections() -> list:
    """Return all unique class sections from the documents table."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT class_section FROM documents WHERE class_section IS NOT NULL AND class_section != ''"
        ).fetchall()
        return [r[0] for r in rows]


def get_documents_by_class(class_section: str) -> list:
    """Return all document filenames belonging to a class section."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT filename FROM documents WHERE class_section = ?", (class_section,)
        ).fetchall()
        return [r[0] for r in rows]


def get_documents_since(since_iso: str) -> list[str]:
    """Return filenames of non-deleted documents uploaded at/after ``since_iso``.

    Used by the scheduled rescan job (``src.core.processing.rescan_recent_documents``)
    to find documents added within the configured grace period so they can
    be re-checked against the full corpus, rather than rescanning everything.

    ``upload_date`` is stored as an ISO-8601 string (see ``add_document``),
    so a lexicographic comparison is sufficient.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT filename FROM documents
            WHERE upload_date >= ?
              AND (is_deleted IS NULL OR is_deleted = 0)
            ORDER BY upload_date ASC
            """,
            (since_iso,),
        ).fetchall()
        return [r[0] for r in rows]


def get_chunks_for_documents(
    filenames: list[str],
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Load chunk texts and embeddings for a set of documents, grouped by filename.

    Returns a dict mapping ``filename -> (chunk_texts, embeddings)`` where
    ``chunk_texts`` is ordered by ``chunk_index`` and ``embeddings`` is a
    ``(num_chunks, 384)`` float32 array in the same order — matching the
    ``chunked_docs`` / ``embeddings`` shapes expected by
    ``src.core.similarity`` and ``src.core.faiss_index``.

    Documents with no stored chunks are omitted from the result.
    """
    if not filenames:
        return {}

    placeholders = ",".join("?" for _ in filenames)
    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT filename, chunk_text, embedding
            FROM chunks
            WHERE filename IN ({placeholders})
            ORDER BY filename ASC, chunk_index ASC
            """,
            tuple(filenames),
        ).fetchall()

    grouped: dict[str, tuple[list[str], list[np.ndarray]]] = {}
    for filename, chunk_text, embedding_blob in rows:
        texts, vectors = grouped.setdefault(filename, ([], []))
        texts.append(chunk_text)
        vectors.append(np.frombuffer(embedding_blob, dtype=np.float32))

    result: dict[str, tuple[list[str], np.ndarray]] = {}
    for filename, (texts, vectors) in grouped.items():
        result[filename] = (
            texts,
            np.vstack(vectors) if vectors else np.empty((0, 384), dtype=np.float32),
        )
    return result


def get_document_count_fast(include_deleted: bool = False) -> int:
    """Return the total document count using SELECT COUNT(*) query."""
    query = "SELECT COUNT(1) FROM documents"
    if not include_deleted:
        query += " WHERE is_deleted IS NULL OR is_deleted = 0"
    with _connect() as conn:
        row = conn.execute(query).fetchone()
        return int(row[0]) if row else 0


def get_embedding_count() -> int:
    """Return the number of durable chunk embeddings in the corpus."""
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(1) FROM chunks").fetchone()
        return int(row[0]) if row else 0


def get_document_count_by_user(owner_username: str) -> int:
    """Return the number of non-deleted documents owned by a specific user."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(1) FROM documents WHERE owner = ? AND is_deleted = 0",
            (owner_username,),
        ).fetchone()
        return int(row[0]) if row else 0


@with_sqlite_retry
def add_documents_bulk(documents: list) -> int:
    """Insert a batch of new documents in a single transaction using executemany."""
    formatted_docs = []
    now = datetime.now().isoformat()
    for doc in documents:
        if not doc.get("file_hash"):
            raise sqlite3.IntegrityError(
                "NOT NULL constraint failed: documents.file_hash"
            )
        if not doc.get("filename"):
            raise sqlite3.IntegrityError(
                "NOT NULL constraint failed: documents.filename"
            )
        formatted_docs.append(
            (
                doc.get("filename"),
                doc.get("file_hash"),
                now,
                doc.get("class_section"),
                doc.get("student_name"),
                doc.get("assignment_title"),
                doc.get("pdf_author"),
                doc.get("pdf_creation_date"),
                doc.get("pdf_title"),
                doc.get("tags"),
                doc.get("detected_language"),
            )
        )

    success_count = 0
    with _connect() as conn:
        try:
            before = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            conn.executemany(
                "INSERT OR IGNORE INTO documents (filename, file_hash, upload_date, class_section, student_name, assignment_title, pdf_author, pdf_creation_date, pdf_title, tags, detected_language) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                formatted_docs,
            )
            conn.commit()
            after = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            success_count = after - before
        except sqlite3.Error as e:
            conn.rollback()
            raise e
    return success_count


def get_all_tags() -> list[str]:
    """Fetches all unique document tags from the database."""
    try:
        with _connect() as conn:
            cursor = conn.execute(
                "SELECT tags FROM documents WHERE tags IS NOT NULL AND tags != ''"
            )
            all_tags_lists = [row[0] for row in cursor.fetchall()]

        from src.core.tag_manager import TagManager

        return TagManager.extract_unique_tags(all_tags_lists)
    except Exception:
        return []


def get_document_tags(filename: str) -> str:
    """Fetches the tags string for a specific document."""
    try:
        with _connect() as conn:
            cursor = conn.execute(
                "SELECT tags FROM documents WHERE filename = ?", (filename,)
            )
            row = cursor.fetchone()
            return row[0] if row and row[0] else ""
    except Exception:
        return ""


@with_sqlite_retry
def update_document_tags(filename: str, tags: str) -> bool:
    """Updates the tags for a specific document."""
    try:
        with _connect() as conn:
            conn.execute(
                "UPDATE documents SET tags = ? WHERE filename = ?",
                (tags, filename),
            )
            return True
    except Exception as e:
        logger.error(f"Failed to update tags for '{filename}': {e}")
        return False


def get_tag_document_count(tag: str) -> int:
    """Counts how many documents currently have the given tag."""
    if not tag or not isinstance(tag, str):
        return 0
    tag = tag.strip()
    if not tag:
        return 0

    count = 0
    try:
        with _connect() as conn:
            cursor = conn.execute(
                "SELECT tags FROM documents WHERE tags IS NOT NULL AND tags != ''"
            )
            for (tags_str,) in cursor.fetchall():
                individual_tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                if tag in individual_tags:
                    count += 1
    except Exception as e:
        logger.error(f"Failed to count documents for tag '{tag}': {e}")
    return count


@with_sqlite_retry
def delete_tag(tag: str) -> int:
    """Removes a specific tag from ALL documents in the database."""
    if not tag or not isinstance(tag, str):
        return 0
    tag = tag.strip()
    if not tag:
        return 0

    affected_count = 0
    try:
        with _connect() as conn:
            cursor = conn.execute(
                "SELECT filename, tags FROM documents WHERE tags IS NOT NULL AND tags != ''"
            )
            rows = cursor.fetchall()
            for filename, tags_str in rows:
                if not tags_str:
                    continue
                individual_tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                if tag in individual_tags:
                    updated_tags = [t for t in individual_tags if t != tag]
                    new_tags_str = (
                        ",".join(sorted(updated_tags)) if updated_tags else ""
                    )
                    conn.execute(
                        "UPDATE documents SET tags = ? WHERE filename = ?",
                        (new_tags_str, filename),
                    )
                    affected_count += 1
    except Exception as e:
        logger.error(f"Failed to delete tag '{tag}': {e}")
        raise
    return affected_count


def check_database_integrity() -> list[str]:
    """Execute PRAGMA integrity_check and return the result."""
    try:
        with _connect() as conn:
            cursor = conn.execute("PRAGMA integrity_check;")
            rows = cursor.fetchall()
            return [row[0] for row in rows]
    except Exception as e:
        logger.error(f"Integrity check failed: {e}")
        return [f"Error: {e}"]


@with_sqlite_retry
def optimize_database() -> dict[str, any]:
    """Executes SQLite VACUUM to reclaim database storage space."""
    path = get_corpus_db_path()
    try:
        size_before = path.stat().st_size if path.exists() else 0

        conn = sqlite3.connect(os.path.abspath(path))
        conn.isolation_level = None
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()

        size_after = path.stat().st_size if path.exists() else 0
        reclaimed_bytes = max(0, size_before - size_after)

        return {
            "size_before": size_before,
            "size_after": size_after,
            "reclaimed_bytes": reclaimed_bytes,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Database optimization (VACUUM) failed: {e}")
        return {
            "size_before": 0,
            "size_after": 0,
            "reclaimed_bytes": 0,
            "error": str(e),
        }


@with_sqlite_retry
def purge_stale_trash(days_in_trash: int = 30) -> int:
    """Automatically purge documents that have been soft-deleted for over a specified number of days."""
    threshold_date = (datetime.now() - timedelta(days=days_in_trash)).isoformat()

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT filename FROM documents
            WHERE is_deleted = 1 AND deleted_at < ?
            """,
            (threshold_date,),
        ).fetchall()

        filenames_to_purge = [row[0] for row in rows]

    deleted_count = 0
    for filename in filenames_to_purge:
        try:
            permanently_delete_document(filename)
            deleted_count += 1
            logger.info(f"Purged stale trashed document: {filename}")
        except Exception as e:
            logger.error(f"Failed to purge stale trashed document {filename}: {e}")

    return deleted_count


def get_total_document_count(include_deleted: bool = False) -> int:
    """Return the total count of non-deleted (or all) indexed documents in the corpus database."""
    with _connect() as conn:
        if include_deleted:
            row = conn.execute("SELECT COUNT(1) FROM documents").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(1) FROM documents WHERE is_deleted IS NULL OR is_deleted = 0"
            ).fetchone()
        return int(row[0]) if row else 0


def search_documents_fts(query_text: str) -> list[dict]:
    """Search the document corpus using the FTS5 full-text index (issue #1359).

    Replaces slow ``LIKE '%query%'`` full table scans with a fast FTS5
    MATCH query against the ``documents_fts`` virtual table. Searches
    across ``filename``, ``student_name``, and ``assignment_title`` columns.

    Args:
        query_text: The search query string. Must be a non-empty string.
                    FTS5 query syntax is supported (e.g., ``"machine learning"``
                    for phrase search, ``machine OR learning`` for boolean).

    Returns:
        A list of dicts, each containing:
        - ``id`` (int): Document row ID.
        - ``filename`` (str): Document filename.
        - ``student_name`` (str|None): Student name if set.
        - ``assignment_title`` (str|None): Assignment title if set.
        - ``upload_date`` (str): ISO timestamp of upload.
        - ``snippet`` (str): FTS5-generated snippet with matched terms highlighted.

        Returns an empty list if the query is empty or no matches are found.
    """
    if not query_text or not query_text.strip():
        return []

    # Sanitize the query for FTS5: wrap in double quotes to prevent
    # FTS5 syntax errors from special characters (e.g., *, :, OR).
    # This treats the query as a phrase match which is the safest
    # default for user-supplied search terms.
    sanitized = query_text.strip().replace('"', '""')
    fts_query = f'"{sanitized}"'

    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    d.id,
                    d.filename,
                    d.student_name,
                    d.assignment_title,
                    d.upload_date,
                    snippet(documents_fts, 0, '<mark>', '</mark>', '...', 32) as snippet
                FROM documents_fts
                JOIN documents d ON d.id = documents_fts.rowid
                WHERE documents_fts MATCH ?
                  AND (d.is_deleted IS NULL OR d.is_deleted = 0)
                ORDER BY rank
                """,
                (fts_query,),
            ).fetchall()

            return [
                {
                    "id": r[0],
                    "filename": r[1],
                    "student_name": r[2],
                    "assignment_title": r[3],
                    "upload_date": r[4],
                    "snippet": r[5],
                }
                for r in rows
            ]
    except sqlite3.OperationalError as e:
        logger.warning("FTS5 search failed: %s. Returning empty list.", e)
        return []


def record_scan_summary(
    document_count: int,
    avg_similarity: float,
    max_similarity: float,
    flagged_count: int,
    threshold_used: float,
) -> bool:
    """Record a scan session summary for historical trend analysis.

    Args:
        document_count: Number of documents processed in the scan.
        avg_similarity: Average similarity score across all document pairs.
        max_similarity: Highest similarity score detected in the scan.
        flagged_count: Number of document pairs flagged as plagiarism.
        threshold_used: The similarity threshold used for flagging.

    Returns:
        True if the record was successfully inserted, False otherwise.
    """
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO scan_history
                (timestamp, document_count, avg_similarity, max_similarity, flagged_count, threshold_used)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    int(document_count),
                    float(avg_similarity),
                    float(max_similarity),
                    int(flagged_count),
                    float(threshold_used),
                ),
            )
        logger.info(
            "Recorded scan summary: %d docs, %.2f avg sim, %d flagged.",
            document_count,
            avg_similarity,
            flagged_count,
        )
        return True
    except Exception as exc:
        logger.error("Failed to record scan summary: %s", exc)
        return False


def get_scan_history(
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Retrieve historical scan summaries with optional date filtering.

    Args:
        start_date: Optional ISO format start date string (YYYY-MM-DD).
        end_date: Optional ISO format end date string (YYYY-MM-DD).
        limit: Maximum number of records to return (default 100).

    Returns:
        List of dictionaries containing scan history records, ordered by timestamp descending.
    """
    query = "SELECT * FROM scan_history WHERE 1=1"
    params = []

    if start_date:
        query += " AND timestamp >= ?"
        params.append(f"{start_date}T00:00:00")
    if end_date:
        query += " AND timestamp <= ?"
        params.append(f"{end_date}T23:59:59")

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(int(limit))

    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
    except Exception as exc:
        logger.error("Failed to retrieve scan history: %s", exc)
        return []
