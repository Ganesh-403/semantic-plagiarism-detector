"""
src/db/evasion_logs_db.py
-------------------------
SQLite database manager for Test-Case Evasion Logs.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/evasion_logs.db")


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Context manager for SQLite connections."""
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


def initialize_evasion_logs_db(db_path: Optional[Path] = None) -> None:
    """Create the evasion logs database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evasion_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                evasion_risk_score REAL NOT NULL,
                is_suspicious INTEGER NOT NULL,
                evasion_patterns TEXT NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info("Evasion logs database initialized at %s", db_path or DEFAULT_DB_PATH)


def log_evasion_analysis(
    document_id: str,
    evasion_risk_score: float,
    is_suspicious: bool,
    evasion_patterns: List[str],
    db_path: Optional[Path] = None,
) -> bool:
    """Persist an evasion analysis result."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO evasion_logs 
                (document_id, evasion_risk_score, is_suspicious, evasion_patterns, analyzed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    evasion_risk_score,
                    1 if is_suspicious else 0,
                    json.dumps(evasion_patterns),
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log evasion analysis: %s", e)
        return False
