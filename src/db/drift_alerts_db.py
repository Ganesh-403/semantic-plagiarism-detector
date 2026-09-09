"""
src/db/drift_alerts_db.py
-------------------------
SQLite database manager for Style Drift Alerts.

Persists detected drift boundaries, change-points, and confidence scores
for intra-document contract cheating analysis.
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

DEFAULT_DB_PATH = DATA_DIR / "drift_alerts.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_drift_db(db_path: Optional[Path] = None) -> None:
    """Create the drift alerts database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                changepoints_json TEXT NOT NULL,
                max_confidence REAL NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_drift_doc
            ON drift_alerts(document_id)
        """
        )
    logger.info("Drift alerts database initialized at %s", db_path or DEFAULT_DB_PATH)


def log_drift_alert(
    document_id: str, changepoints: List[Dict[str, Any]], db_path: Optional[Path] = None
) -> bool:
    """Persist a drift detection alert."""
    max_conf = max([cp.get("confidence", 0.0) for cp in changepoints], default=0.0)
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO drift_alerts
                (document_id, changepoints_json, max_confidence, analyzed_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    document_id,
                    json.dumps(changepoints),
                    max_conf,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log drift alert for %s: %s", document_id, e)
        return False
