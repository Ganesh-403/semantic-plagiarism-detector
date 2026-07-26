from __future__ import annotations

"""
corpus_db.py
------------
SQLite database manager to persist document metadata, chunk text, and embeddings.
Enables incremental updates and index rebuilding without re-embedding.
"""

import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np

from src.db.migrations import (delete_all_if_table_exists,
                               migrate_corpus_database)
from src.utils.filename import sanitize_filename

_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "corpus.db")
)

_connection_pool = threading.local()


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
    """Borrow a reusable connection and manage the operation transaction.

    Connections are kept per thread and database path so consecutive database
    operations reuse the same SQLite handle.  The context manager commits on
    success and rolls back on failure; :func:`close_connections` closes the
    handles when the process or a test is finished with the database.
    """
    path = os.path.abspath(_DB_PATH)
    pool = _pool()
    conn = pool.get(path)
    if conn is None:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        pool[path] = conn

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def close_connections() -> None:
    """Close all pooled corpus connections for the current thread."""
    pool = getattr(_connection_pool, "connections", {})
    for conn in pool.values():
        conn.close()
    pool.clear()


def init_corpus_db() -> None:
    """Create or upgrade corpus.db without deleting persisted data."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                filename         TEXT    UNIQUE NOT NULL,
                file_hash        TEXT    UNIQUE NOT NULL,
                upload_date      TEXT    NOT NULL,
                class_section    TEXT,
                student_name     TEXT,
                assignment_title TEXT,
                pdf_author       TEXT,
                pdf_creation_date TEXT,
                pdf_title        TEXT,
                tags             TEXT
            )
        """
        )

        # Schema migration fallback logic: add missing columns if documents table already existed
        cursor = conn.execute("PRAGMA table_info(documents)")
        columns = [row[1] for row in cursor.fetchall()]
        if "class_section" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN class_section TEXT")
        if "student_name" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN student_name TEXT")
        if "assignment_title" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN assignment_title TEXT")
        if "pdf_author" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN pdf_author TEXT")
        if "pdf_creation_date" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN pdf_creation_date TEXT")
        if "pdf_title" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN pdf_title TEXT")
        if "tags" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN tags TEXT")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                vector_id    INTEGER PRIMARY KEY,
                filename     TEXT    NOT NULL,
                chunk_index  INTEGER NOT NULL,
                chunk_text   TEXT    NOT NULL,
                embedding    BLOB    NOT NULL,
                FOREIGN KEY (filename) REFERENCES documents(filename) ON DELETE CASCADE
            )
        """
        )
        conn.commit()
        migrate_corpus_database(conn)

    # Restrict database file permissions to owner read/write only
    # Prevents other local users on the server from reading the corpus data
    try:
        os.chmod(_DB_PATH, 0o600)
    except OSError:
        pass  # Best-effort; some platforms (e.g., Windows) may not support chmod


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
) -> bool:
    """
    Insert a new document metadata row using parameterized execution.
    Returns True if successfully inserted, False if it already exists.

    The filename is sanitized again here so direct database callers cannot
    persist HTML, JavaScript, traversal components, or control characters.
    """
    filename = sanitize_filename(filename)

    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO documents (filename, file_hash, upload_date, class_section, student_name, assignment_title, pdf_author, pdf_creation_date, pdf_title, tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                ),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def get_document_by_hash(file_hash: str) -> str | None:
    """Check if a file with this hash is already indexed and return its filename."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT filename FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()
    return row[0] if row else None


def get_all_documents() -> list:
    """Return all indexed documents sorted by upload date descending."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT filename, file_hash, upload_date, class_section, student_name, assignment_title, pdf_author, pdf_creation_date, pdf_title FROM documents ORDER BY upload_date DESC"
        ).fetchall()
    return [
        {
            "filename": r[0],
            "file_hash": r[1],
            "upload_date": r[2],
            "class_section": r[3],
            "student_name": r[4],
            "assignment_title": r[5],
            "pdf_author": r[6],
            "pdf_creation_date": r[7],
            "pdf_title": r[8],
        }
        for r in rows
    ]


def add_chunks(chunks_to_add: list) -> None:
    """
    Insert a batch of chunks with their raw text and embedded BLOBs.

    chunks_to_add: list of tuples: (vector_id, filename, chunk_index, chunk_text, embedding_np_array)
    """
    formatted_chunks = []
    for vid, fname, idx, text, emb in chunks_to_add:
        # Convert float32 numpy array to raw bytes BLOB
        emb_blob = emb.astype(np.float32).tobytes()
        formatted_chunks.append((vid, fname, idx, text, emb_blob))

    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
            formatted_chunks,
        )
        conn.commit()


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


def delete_document(filename: str) -> None:
    """
    Delete a document and all its associated chunks (cascade).
    After deletion, vector_ids will have gaps, so we need to compact the vector IDs.
    """
    with _connect() as conn:
        # Delete document (triggers cascading delete on chunks)
        conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        conn.commit()

    # Re-index all remaining chunks so vector_ids are sequential [0, 1, ..., N-1]
    _compact_vector_ids()


def _compact_vector_ids() -> None:
    """Re-index the vector_id column to remove any gaps left by deleted documents."""
    with _connect() as conn:
        # Retrieve all chunks ordered by current vector_id
        chunks = conn.execute(
            "SELECT filename, chunk_index, chunk_text, embedding FROM chunks ORDER BY vector_id ASC"
        ).fetchall()

        # Clear chunks table
        conn.execute("DELETE FROM chunks")

        # Insert them back with fresh sequential IDs starting at 0
        if chunks:
            formatted = [(i, r[0], r[1], r[2], r[3]) for i, r in enumerate(chunks)]
            conn.executemany(
                "INSERT INTO chunks (vector_id, filename, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?, ?)",
                formatted,
            )
        conn.commit()


def get_document_chunks_count(filename: str) -> int:
    """Return the number of chunks for a given document."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(1) FROM chunks WHERE filename = ?", (filename,)
        ).fetchone()
    return row[0] if row else 0


def get_document_word_counts() -> dict[str, int]:
    """Calculate and return the total word count for each document currently in the database based on its chunks."""
    import re

    with _connect() as conn:
        rows = conn.execute("SELECT filename, chunk_text FROM chunks").fetchall()

    word_counts = {}
    for filename, chunk_text in rows:
        words = len(re.findall(r"\b\w+\b", chunk_text or ""))
        word_counts[filename] = word_counts.get(filename, 0) + words
    return word_counts


def clear_all_data() -> None:
    """Clear known corpus tables while tolerating partial schemas."""
    with _connect() as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        delete_all_if_table_exists(conn, "chunks")
        delete_all_if_table_exists(conn, "documents")
        delete_all_if_table_exists(conn, "plagiarism_incidents")
        conn.commit()


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


def get_embedding_count() -> int:
    """Return the number of durable chunk embeddings in the corpus."""
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(1) FROM chunks").fetchone()
    return int(row[0]) if row else 0


def add_documents_bulk(documents: list) -> int:
    """
    Insert a batch of new documents in a single transaction using executemany.
    documents: list of dicts containing metadata (filename, file_hash, class_section, etc).
    Returns the number of documents successfully inserted.
    """
    formatted_docs = []
    now = datetime.now().isoformat()
    for doc in documents:
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
            )
        )

    success_count = 0
    with _connect() as conn:
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO documents (filename, file_hash, upload_date, class_section, student_name, assignment_title, pdf_author, pdf_creation_date, pdf_title, tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                formatted_docs,
            )
            success_count = conn.execute("SELECT changes()").fetchone()[0]
            conn.commit()
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

            # Use TagManager to extract unique
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


def update_document_tags(filename: str, tags: str) -> bool:
    """Updates the tags for a specific document."""
    try:
        with _connect() as conn:
            conn.execute(
                "UPDATE documents SET tags = ? WHERE filename = ?",
                (tags, filename)
            )
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to update tags for '{filename}': {e}")
        return False


def delete_tag(tag: str) -> int:
    """
    Removes a specific tag from ALL documents in the database.
    Returns the number of documents that were modified.
    """
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
            conn.commit()
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
