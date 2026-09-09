"""
src/db/multimedia_forensics_db.py
---------------------------------
SQLite database manager for Multimedia Forensics Logs.
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

DEFAULT_DB_PATH = DATA_DIR / "multimedia_forensics.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_multimedia_forensics_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS multimedia_forensics_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, media_a_id TEXT NOT NULL, media_b_id TEXT NOT NULL,
            dubbing_probability REAL NOT NULL, is_dubbed INTEGER NOT NULL, analyzed_at TEXT NOT NULL)"""
        )


def log_av_forensics(
    media_a_id: str,
    media_b_id: str,
    prob: float,
    is_dubbed: bool,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO multimedia_forensics_logs (media_a_id, media_b_id, dubbing_probability, is_dubbed, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    media_a_id,
                    media_b_id,
                    prob,
                    1 if is_dubbed else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log AV forensics: %s", e)
        return False
