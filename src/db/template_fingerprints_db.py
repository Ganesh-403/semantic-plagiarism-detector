"""
src/db/template_fingerprints_db.py
----------------------------------
SQLite database manager for Template Fingerprint Logs.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data/template_fingerprints.db")


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


def initialize_template_fingerprints_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS template_fingerprint_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_a_id TEXT NOT NULL,
                doc_b_id TEXT NOT NULL,
                is_template_plagiarism INTEGER NOT NULL,
                entropy_delta REAL NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info(
        "Template fingerprints database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_template_comparison(
    doc_a_id: str,
    doc_b_id: str,
    is_template_plagiarism: bool,
    entropy_delta: float,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO template_fingerprint_logs (doc_a_id, doc_b_id, is_template_plagiarism, entropy_delta, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    doc_a_id,
                    doc_b_id,
                    1 if is_template_plagiarism else 0,
                    entropy_delta,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log template comparison: %s", e)
        return False
