"""
src/db/citation_graph_db.py
---------------------------
SQLite database manager for the Citation Graph.

Stores nodes (papers) and edges (citations) to build a network of
academic references. Computes similarity between student bibliographies
to detect citation laundering and shared reference rings.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from contextlib import contextmanager
from datetime import datetime

from src.core.citation_extractor import Citation

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/citation_graph.db")


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Context manager for SQLite connections."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_citation_db(db_path: Optional[Path] = None) -> None:
    """Create the citation graph database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS citation_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_key TEXT UNIQUE NOT NULL,
                authors TEXT,
                year TEXT,
                title TEXT,
                source TEXT
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_citations (
                document_id TEXT NOT NULL,
                node_key TEXT NOT NULL,
                cited_at TEXT NOT NULL,
                PRIMARY KEY (document_id, node_key),
                FOREIGN KEY (node_key) REFERENCES citation_nodes(node_key)
            )
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_doc_citations_doc 
            ON document_citations(document_id)
        """
        )

    logger.info("Citation graph database initialized at %s", db_path or DEFAULT_DB_PATH)


def ingest_citations(
    document_id: str, citations: list[Citation], db_path: Optional[Path] = None
) -> int:
    """Ingest a list of citations for a specific document."""
    inserted = 0
    try:
        with get_connection(db_path) as conn:
            for cite in citations:
                node_key = cite.get_normalized_key()

                # Insert node if it doesn't exist
                conn.execute(
                    """
                    INSERT OR IGNORE INTO citation_nodes 
                    (node_key, authors, year, title, source)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (node_key, cite.authors, cite.year, cite.title, cite.source),
                )

                # Link document to node
                conn.execute(
                    """
                    INSERT OR IGNORE INTO document_citations 
                    (document_id, node_key, cited_at)
                    VALUES (?, ?, ?)
                    """,
                    (document_id, node_key, datetime.utcnow().isoformat()),
                )
                inserted += 1
        return inserted
    except sqlite3.Error as e:
        logger.error("Failed to ingest citations for %s: %s", document_id, e)
        return 0


def get_document_citation_keys(
    document_id: str, db_path: Optional[Path] = None
) -> set[str]:
    """Retrieve the set of citation node keys for a document."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT node_key FROM document_citations WHERE document_id = ?",
                (document_id,),
            )
            return {row["node_key"] for row in cursor.fetchall()}
    except sqlite3.Error as e:
        logger.error("Failed to get citation keys for %s: %s", document_id, e)
        return set()
