"""
src/db/git_forensics_db.py
--------------------------
SQLite database manager for Git Forensics Logs.
"""

import sqlite3, logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data/git_forensics.db")


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


def initialize_git_forensics_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS git_forensics_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, log_a_id TEXT NOT NULL, log_b_id TEXT NOT NULL,
            overall_score REAL NOT NULL, is_covert_collaboration INTEGER NOT NULL, analyzed_at TEXT NOT NULL)"""
        )


def log_git_forensics(
    log_a_id: str,
    log_b_id: str,
    score: float,
    is_covert: bool,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO git_forensics_logs (log_a_id, log_b_id, overall_score, is_covert_collaboration, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    log_a_id,
                    log_b_id,
                    score,
                    1 if is_covert else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log Git forensics: %s", e)
        return False
