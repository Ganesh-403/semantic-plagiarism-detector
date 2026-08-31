"""
src/db/code_comment_db.py
-------------------------
SQLite database manager for Code Comment Alignment Logs.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data/code_comment.db")


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_code_comment_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS code_comment_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                overall_coherence REAL NOT NULL,
                is_mismatch INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info("Code comment database initialized at %s", db_path or DEFAULT_DB_PATH)


def log_code_comment_alignment(
    document_id: str,
    overall_coherence: float,
    is_mismatch: bool,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO code_comment_logs (document_id, overall_coherence, is_mismatch, analyzed_at) VALUES (?, ?, ?, ?)",
                (
                    document_id,
                    overall_coherence,
                    1 if is_mismatch else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log code comment alignment: %s", e)
        return False
