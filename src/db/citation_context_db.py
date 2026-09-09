"""
src/db/citation_context_db.py
-----------------------------
SQLite database manager for Citation Context Alignment Logs.
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

DEFAULT_DB_PATH = DATA_DIR / "citation_context.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_citation_context_db(db_path: Optional[Path] = None) -> None:
    """Create the citation context database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS citation_context_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                citation_id TEXT NOT NULL,
                alignment_score REAL NOT NULL,
                is_bluffing INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info(
        "Citation context database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_citation_alignment(
    document_id: str,
    citation_id: str,
    alignment_score: float,
    is_bluffing: bool,
    db_path: Optional[Path] = None,
) -> bool:
    """Persist a citation alignment result."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO citation_context_logs
                (document_id, citation_id, alignment_score, is_bluffing, analyzed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    citation_id,
                    alignment_score,
                    1 if is_bluffing else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log citation alignment: %s", e)
        return False
