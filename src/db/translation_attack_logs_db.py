"""
src/db/translation_attack_logs_db.py
------------------------------------
SQLite database manager for Translation Attack Logs.

Persists detected back-translation attacks, drift metrics, and invariance scores.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

from src.core.app_config import DATA_DIR
from src.db.connection import get_connection as managed_connection

DEFAULT_DB_PATH = DATA_DIR / "translation_attack_logs.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_translation_logs_db(db_path: Optional[Path] = None) -> None:
    """Create the translation attack logs database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translation_attack_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                lexical_drift REAL NOT NULL,
                structural_variance REAL NOT NULL,
                invariance_score REAL NOT NULL,
                is_obfuscated INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_translation_doc
            ON translation_attack_logs(document_id)
        """
        )
    logger.info(
        "Translation attack logs database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_translation_attack(
    document_id: str,
    lexical_drift: float,
    structural_variance: float,
    invariance_score: float,
    is_obfuscated: bool,
    db_path: Optional[Path] = None,
) -> bool:
    """Persist a translation attack detection result."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO translation_attack_logs
                (document_id, lexical_drift, structural_variance, invariance_score, is_obfuscated, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    lexical_drift,
                    structural_variance,
                    invariance_score,
                    1 if is_obfuscated else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log translation attack for %s: %s", document_id, e)
        return False
