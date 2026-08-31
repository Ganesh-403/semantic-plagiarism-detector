"""
src/db/api_graph_db.py
----------------------
SQLite database manager for API Call Graph Logs.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data/api_graphs.db")


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


def initialize_api_graph_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_graph_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_a_id TEXT NOT NULL,
                code_b_id TEXT NOT NULL,
                overall_score REAL NOT NULL,
                is_clone INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info("API graph database initialized at %s", db_path or DEFAULT_DB_PATH)


def log_api_graph_alignment(
    code_a_id: str,
    code_b_id: str,
    overall_score: float,
    is_clone: bool,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO api_graph_logs (code_a_id, code_b_id, overall_score, is_clone, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    code_a_id,
                    code_b_id,
                    overall_score,
                    1 if is_clone else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log API graph alignment: %s", e)
        return False
