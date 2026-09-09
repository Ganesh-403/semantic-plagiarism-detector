"""
src/db/audio_prosody_db.py
--------------------------
SQLite database manager for Audio Prosody Logs.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
from src.core.app_config import DATA_DIR
from src.db.connection import get_connection as managed_connection

DEFAULT_DB_PATH = DATA_DIR / "audio_prosody.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_audio_prosody_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_prosody_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                is_synthetic INTEGER NOT NULL,
                pause_variance REAL NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info("Audio prosody database initialized at %s", db_path or DEFAULT_DB_PATH)


def log_prosody_analysis(
    document_id: str,
    is_synthetic: bool,
    pause_variance: float,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO audio_prosody_logs (document_id, is_synthetic, pause_variance, analyzed_at) VALUES (?, ?, ?, ?)",
                (
                    document_id,
                    1 if is_synthetic else 0,
                    pause_variance,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log prosody analysis: %s", e)
        return False
