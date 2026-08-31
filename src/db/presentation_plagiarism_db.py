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
DEFAULT_DB_PATH = Path("data/presentation_plagiarism.db")


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
