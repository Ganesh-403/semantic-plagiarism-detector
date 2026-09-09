"""
src/db/presentation_plagiarism_db.py
------------------------------------
SQLite database manager for Presentation Plagiarism Logs.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
from src.core.app_config import DATA_DIR
from src.db.connection import get_connection as managed_connection

DEFAULT_DB_PATH = DATA_DIR / "presentation_plagiarism.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_presentation_plagiarism_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS presentation_plagiarism_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deck_a_id TEXT NOT NULL,
                deck_b_id TEXT NOT NULL,
                overall_score REAL NOT NULL,
                is_cloned_deck INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info(
        "Presentation plagiarism database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_presentation_alignment(
    deck_a_id: str,
    deck_b_id: str,
    overall_score: float,
    is_cloned: bool,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO presentation_plagiarism_logs (deck_a_id, deck_b_id, overall_score, is_cloned_deck, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    deck_a_id,
                    deck_b_id,
                    overall_score,
                    1 if is_cloned else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log presentation alignment: %s", e)
        return False
