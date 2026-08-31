"""
src/db/steganography_logs_db.py
-------------------------------
SQLite database manager for Steganography Logs.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path("data/steganography_logs.db")


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


def initialize_steganography_logs_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS steganography_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                is_injection INTEGER NOT NULL,
                risk_score REAL NOT NULL,
                matched_patterns TEXT NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info(
        "Steganography logs database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_steganography_analysis(
    document_id: str,
    is_injection: bool,
    risk_score: float,
    matched_patterns: list,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO steganography_logs (document_id, is_injection, risk_score, matched_patterns, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    document_id,
                    1 if is_injection else 0,
                    risk_score,
                    json.dumps(matched_patterns),
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log steganography analysis: %s", e)
        return False
