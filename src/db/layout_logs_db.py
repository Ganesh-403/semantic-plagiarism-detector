"""
src/db/layout_logs_db.py
------------------------
SQLite database manager for Layout Plagiarism Logs.

Persists layout hashes and structural match reports.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

from src.core.app_config import DATA_DIR
from src.db.connection import get_connection as managed_connection

DEFAULT_DB_PATH = DATA_DIR / "layout_logs.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_layout_logs_db(db_path: Optional[Path] = None) -> None:
    """Create the layout logs database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS layout_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_a_id TEXT NOT NULL,
                document_b_id TEXT NOT NULL,
                edit_distance INTEGER NOT NULL,
                structural_similarity REAL NOT NULL,
                is_structural_clone INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info("Layout logs database initialized at %s", db_path or DEFAULT_DB_PATH)


def log_layout_comparison(
    doc_a_id: str,
    doc_b_id: str,
    edit_distance: int,
    similarity: float,
    is_clone: bool,
    db_path: Optional[Path] = None,
) -> bool:
    """Persist a layout comparison result."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO layout_logs
                (document_a_id, document_b_id, edit_distance, structural_similarity, is_structural_clone, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_a_id,
                    doc_b_id,
                    edit_distance,
                    similarity,
                    1 if is_clone else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log layout comparison: %s", e)
        return False
