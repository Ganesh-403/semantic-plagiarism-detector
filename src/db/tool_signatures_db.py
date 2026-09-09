"""
src/db/tool_signatures_db.py
----------------------------
SQLite database manager for Paraphrase Tool Signatures.

Persists known statistical signatures of commercial paraphrase tools
and logs attribution results for historical analysis.
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

DEFAULT_DB_PATH = DATA_DIR / "tool_signatures.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_signatures_db(db_path: Optional[Path] = None) -> None:
    """Create the tool signatures database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_signatures (
                tool_name TEXT PRIMARY KEY,
                signature_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attribution_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                attributed_tool TEXT NOT NULL,
                confidence REAL NOT NULL,
                fingerprint_json TEXT NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )

    logger.info(
        "Tool signatures database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_attribution(
    document_id: str,
    attributed_tool: str,
    confidence: float,
    fingerprint: dict[str, float],
    db_path: Optional[Path] = None,
) -> bool:
    """Log a paraphrase tool attribution result."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO attribution_logs
                (document_id, attributed_tool, confidence, fingerprint_json, analyzed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    attributed_tool,
                    confidence,
                    json.dumps(fingerprint),
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log attribution for %s: %s", document_id, e)
        return False
