"""
src/db/citation_context_db.py
-----------------------------
SQLite database manager for Citation Context Alignment Logs.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/citation_context.db")


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Context manager for SQLite connections."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_citation_context_db(db_path: Optional[Path] = None) -> None:
    """Create the citation context database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS citation_context_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                citation_id TEXT NOT NULL,
                alignment_score REAL NOT NULL,
                is_bluffing INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info(
        "Citation context database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_citation_alignment(
    document_id: str,
    citation_id: str,
    alignment_score: float,
    is_bluffing: bool,
    db_path: Optional[Path] = None,
) -> bool:
    """Persist a citation alignment result."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO citation_context_logs 
                (document_id, citation_id, alignment_score, is_bluffing, analyzed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    citation_id,
                    alignment_score,
                    1 if is_bluffing else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log citation alignment: %s", e)
        return False
