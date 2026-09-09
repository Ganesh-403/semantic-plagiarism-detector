"""
src/db/provenance_logs_db.py
----------------------------
SQLite database manager for Document Provenance Logs.

Persists forensic flags, metadata extractions, and historical provenance
baselines for audit and anomaly detection.
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

DEFAULT_DB_PATH = DATA_DIR / "provenance_logs.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_provenance_db(db_path: Optional[Path] = None) -> None:
    """Create the provenance logs database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS provenance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                file_type TEXT NOT NULL,
                risk_score REAL NOT NULL,
                is_suspicious INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_provenance_doc
            ON provenance_logs(document_id)
        """
        )
    logger.info(
        "Provenance logs database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_provenance_analysis(
    document_id: str,
    file_type: str,
    risk_score: float,
    is_suspicious: bool,
    metadata: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> bool:
    """Persist a provenance analysis result."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO provenance_logs
                (document_id, file_type, risk_score, is_suspicious, metadata_json, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    file_type,
                    risk_score,
                    1 if is_suspicious else 0,
                    json.dumps(metadata),
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log provenance analysis for %s: %s", document_id, e)
        return False
