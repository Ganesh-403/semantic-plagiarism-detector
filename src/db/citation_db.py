"""
src/db/citation_db.py
---------------------
SQLite database manager for bibliography citations and structural plagiarism.

Persists extracted citations and maps them to source documents, enabling
cross-document citation graph analysis and "citation lifting" detection.

Recent Additions (Issue #1958):
- Created citations and document_citations tables.
- Implemented helpers to store, retrieve, and compare citation sets.
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from typing import Dict, List

from src.db.corpus_db import _DB_PATH, FALLBACK_CORPUS_DB_PATH

logger = logging.getLogger(__name__)

_connection_pool = threading.local()


def _pool() -> dict:
    pool = getattr(_connection_pool, "connections", None)
    if pool is None:
        pool = {}
        _connection_pool.connections = pool
    return pool


@contextmanager
def _connect(readonly: bool = False):
    """Borrow a reusable SQLite connection."""
    import os

    path = os.path.abspath(_DB_PATH)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except (OSError, PermissionError):
        path = str(FALLBACK_CORPUS_DB_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)

    pool = _pool()
    conn = pool.get(path)
    if conn is None:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        pool[path] = conn

    try:
        yield conn
        if not readonly:
            conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_citation_db() -> None:
    """Create the citation tables if they do not exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                hash TEXT PRIMARY KEY,
                author TEXT,
                year TEXT,
                title TEXT,
                raw_text TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_citations (
                doc_name TEXT NOT NULL,
                citation_hash TEXT NOT NULL,
                is_ghost INTEGER DEFAULT 0,
                PRIMARY KEY (doc_name, citation_hash),
                FOREIGN KEY (citation_hash) REFERENCES citations(hash) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_citations_doc
            ON document_citations(doc_name)
        """)
    logger.info("Citation database tables initialized.")


def add_document_citations(doc_name: str, citations: list[dict[str, str]]) -> int:
    """Insert extracted citations into the database and link them to a document.

    Args:
        doc_name: The filename of the source document.
        citations: List of citation dictionaries from extract_citations().

    Returns:
        The number of new citation links created.
    """
    if not citations:
        return 0

    added_count = 0
    try:
        with _connect() as conn:
            for cit in citations:
                # Insert into master citations table (ignore if exists)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO citations (hash, author, year, title, raw_text)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        cit["hash"],
                        cit["author"],
                        cit["year"],
                        cit["title"],
                        cit["raw_text"],
                    ),
                )

                # Link to document
                try:
                    cursor = conn.execute(
                        """
                        INSERT OR IGNORE INTO document_citations (doc_name, citation_hash)
                        VALUES (?, ?)
                        """,
                        (doc_name, cit["hash"]),
                    )
                    if cursor.rowcount > 0:
                        added_count += 1
                except sqlite3.IntegrityError:
                    pass
    except Exception as exc:
        logger.error("Failed to add citations for %s: %s", doc_name, exc)

    return added_count


def get_shared_citations(doc_a: str, doc_b: str) -> list[dict[str, str]]:
    """Find citations that appear in both Document A and Document B.

    Args:
        doc_a: Filename of the first document.
        doc_b: Filename of the second document.

    Returns:
        List of citation dictionaries that are shared between the two documents.
    """
    query = """
        SELECT c.hash, c.author, c.year, c.title
        FROM citations c
        JOIN document_citations dc1 ON c.hash = dc1.citation_hash
        JOIN document_citations dc2 ON c.hash = dc2.citation_hash
        WHERE dc1.doc_name = ? AND dc2.doc_name = ?
    """
    try:
        with _connect(readonly=True) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, (doc_a, doc_b))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as exc:
        logger.error("Failed to get shared citations: %s", exc)
        return []


def compute_citation_jaccard(doc_a: str, doc_b: str) -> float:
    """Compute the Jaccard similarity index of the bibliographies of two documents.

    Jaccard = |Intersection| / |Union|

    Args:
        doc_a: Filename of the first document.
        doc_b: Filename of the second document.

    Returns:
        Float between 0.0 and 1.0 representing bibliography overlap.
    """
    query_a = "SELECT citation_hash FROM document_citations WHERE doc_name = ?"
    query_b = "SELECT citation_hash FROM document_citations WHERE doc_name = ?"

    try:
        with _connect(readonly=True) as conn:
            set_a = {row[0] for row in conn.execute(query_a, (doc_a,)).fetchall()}
            set_b = {row[0] for row in conn.execute(query_b, (doc_b,)).fetchall()}

            if not set_a and not set_b:
                return 0.0

            intersection = len(set_a & set_b)
            union = len(set_a | set_b)

            return float(intersection / union) if union > 0 else 0.0
    except Exception as exc:
        logger.error("Failed to compute citation Jaccard: %s", exc)
        return 0.0
