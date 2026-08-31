"""
src/db/revision_bursts_db.py
-----------------------------
SQLite database manager for Revision Burst Logs.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data/revision_bursts.db")


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_revision_bursts_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS revision_burst_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                risk_score REAL NOT NULL,
                is_ghostwritten INTEGER NOT NULL,
                burst_ratio REAL NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info(
        "Revision bursts database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_revision_burst_analysis(
    document_id: str,
    risk_score: float,
    is_ghostwritten: bool,
    burst_ratio: float,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO revision_burst_logs (document_id, risk_score, is_ghostwritten, burst_ratio, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    document_id,
                    risk_score,
                    1 if is_ghostwritten else 0,
                    burst_ratio,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log revision burst analysis: %s", e)
        return False
