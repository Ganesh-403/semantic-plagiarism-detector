"""
src/db/cognitive_load_db.py
---------------------------
SQLite database manager for Cognitive Load Logs.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data/cognitive_load.db")


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


def initialize_cognitive_load_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cognitive_load_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                ai_probability REAL NOT NULL,
                is_ai_generated INTEGER NOT NULL,
                fk_variance REAL NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info("Cognitive load database initialized at %s", db_path or DEFAULT_DB_PATH)


def log_cognitive_load_analysis(
    document_id: str,
    ai_probability: float,
    is_ai_generated: bool,
    fk_variance: float,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO cognitive_load_logs (document_id, ai_probability, is_ai_generated, fk_variance, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    document_id,
                    ai_probability,
                    1 if is_ai_generated else 0,
                    fk_variance,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log cognitive load analysis: %s", e)
        return False
