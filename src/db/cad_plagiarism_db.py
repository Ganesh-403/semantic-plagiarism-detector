"""
src/db/cad_plagiarism_db.py
---------------------------
SQLite database manager for CAD Plagiarism Logs.
"""

import sqlite3, logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data/cad_plagiarism.db")


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


def initialize_cad_plagiarism_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS cad_plagiarism_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, model_a_id TEXT NOT NULL, model_b_id TEXT NOT NULL,
            overall_score REAL NOT NULL, is_cloned_geometry INTEGER NOT NULL, analyzed_at TEXT NOT NULL)"""
        )


def log_cad_alignment(
    model_a_id: str,
    model_b_id: str,
    score: float,
    is_cloned: bool,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO cad_plagiarism_logs (model_a_id, model_b_id, overall_score, is_cloned_geometry, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    model_a_id,
                    model_b_id,
                    score,
                    1 if is_cloned else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log CAD alignment: %s", e)
        return False
