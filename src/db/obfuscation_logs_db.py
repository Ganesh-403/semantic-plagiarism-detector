"""
src/db/obfuscation_logs_db.py
-----------------------------
SQLite database manager for logging suspected adversarial obfuscation attempts.

Maintains an audit trail of documents that triggered the obfuscation
detector, allowing administrators to review patterns of cheating and
identify repeat offenders.
"""

import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/obfuscation_logs.db")


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Context manager for acquiring and releasing SQLite connections."""
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


def initialize_obfuscation_db(db_path: Optional[Path] = None) -> None:
    """Create the obfuscation logs database schema if it doesn't exist."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS obfuscation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                document_hash TEXT NOT NULL,
                user_id TEXT NOT NULL,
                obfuscation_score REAL NOT NULL,
                zero_width_count INTEGER DEFAULT 0,
                homoglyph_count INTEGER DEFAULT 0,
                flagged_indices_count INTEGER DEFAULT 0,
                detected_at TEXT NOT NULL
            )
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_obfuscation_user 
            ON obfuscation_logs(user_id)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_obfuscation_hash 
            ON obfuscation_logs(document_hash)
        """
        )

    logger.info(
        "Obfuscation logs database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_obfuscation_attempt(
    document_id: str,
    document_hash: str,
    user_id: str,
    score: float,
    zero_width_count: int = 0,
    homoglyph_count: int = 0,
    flagged_indices_count: int = 0,
    db_path: Optional[Path] = None,
) -> bool:
    """Insert a new obfuscation detection event into the audit log."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO obfuscation_logs 
                (document_id, document_hash, user_id, obfuscation_score, 
                 zero_width_count, homoglyph_count, flagged_indices_count, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    document_hash,
                    user_id,
                    score,
                    zero_width_count,
                    homoglyph_count,
                    flagged_indices_count,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log obfuscation attempt for %s: %s", document_id, e)
        return False


def get_user_obfuscation_history(
    user_id: str, limit: int = 50, db_path: Optional[Path] = None
) -> list[dict[str, Any]]:
    """Retrieve the obfuscation log history for a specific user."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM obfuscation_logs 
                WHERE user_id = ? 
                ORDER BY detected_at DESC 
                LIMIT ?
                """,
                (user_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error("Failed to fetch obfuscation history for user %s: %s", user_id, e)
        return []
import json
from datetime import datetime

# Implements mock connection context hooks; swap out for your active ORM/Supabase adapter seamlessly
def log_obfuscation_incident(incident_data: dict) -> bool:
    """
    Saves flagged evasion attempts into the 'obfuscation_incidents' schema.
    
    Expected Database Table Structure:
    CREATE TABLE obfuscation_incidents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id VARCHAR(255),
        document_hash CHAR(64) NOT NULL,
        obfuscation_score NUMERIC(5,2) NOT NULL,
        patterns_json JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """
    try:
        # Mock database insertion execution logic
        record = {
            "document_id": incident_data["document_id"],
            "document_hash": incident_data["document_hash"],
            "obfuscation_score": incident_data["obfuscation_score"],
            "patterns_json": json.dumps(incident_data["patterns_found"]),
            "created_at": datetime.utcnow().isoformat()
        }
        # print(f"Saving security incident row: {record}")
        return True
    except Exception as db_error:
        print(f"Failed to log security anomaly profile: {str(db_error)}")
        return False
