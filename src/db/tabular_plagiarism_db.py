"""
src/db/tabular_plagiarism_db.py
-------------------------------
SQLite database manager for Tabular Plagiarism Logs.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data/tabular_plagiarism.db")


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


def initialize_tabular_plagiarism_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tabular_plagiarism_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_a_id TEXT NOT NULL,
                table_b_id TEXT NOT NULL,
                overall_score REAL NOT NULL,
                is_cloned_dataset INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info(
        "Tabular plagiarism database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_table_alignment(
    table_a_id: str,
    table_b_id: str,
    overall_score: float,
    is_cloned: bool,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO tabular_plagiarism_logs (table_a_id, table_b_id, overall_score, is_cloned_dataset, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    table_a_id,
                    table_b_id,
                    overall_score,
                    1 if is_cloned else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log table alignment: %s", e)
        return False
