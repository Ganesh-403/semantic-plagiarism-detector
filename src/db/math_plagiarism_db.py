"""
src/db/math_plagiarism_db.py
----------------------------
SQLite database manager for Math Plagiarism Logs.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data/math_plagiarism.db")


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


def initialize_math_plagiarism_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS math_plagiarism_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                eq_a_id TEXT NOT NULL,
                eq_b_id TEXT NOT NULL,
                structural_similarity REAL NOT NULL,
                is_structural_plagiarism INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info(
        "Math plagiarism database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_math_alignment(
    eq_a_id: str,
    eq_b_id: str,
    structural_similarity: float,
    is_plagiarism: bool,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO math_plagiarism_logs (eq_a_id, eq_b_id, structural_similarity, is_structural_plagiarism, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    eq_a_id,
                    eq_b_id,
                    structural_similarity,
                    1 if is_plagiarism else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log math alignment: %s", e)
        return False
