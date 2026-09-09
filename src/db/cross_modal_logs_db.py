"""
src/db/cross_modal_logs_db.py
-----------------------------
SQLite database manager for Cross-Modal Alignment Logs.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
from src.core.app_config import DATA_DIR
from src.db.connection import get_connection as managed_connection

DEFAULT_DB_PATH = DATA_DIR / "cross_modal_logs.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_cross_modal_logs_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cross_modal_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_doc_id TEXT NOT NULL,
                code_doc_id TEXT NOT NULL,
                overall_score REAL NOT NULL,
                is_translation INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info(
        "Cross-modal logs database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_cross_modal_alignment(
    text_doc_id: str,
    code_doc_id: str,
    overall_score: float,
    is_translation: bool,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO cross_modal_logs (text_doc_id, code_doc_id, overall_score, is_translation, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    text_doc_id,
                    code_doc_id,
                    overall_score,
                    1 if is_translation else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log cross-modal alignment: %s", e)
        return False
