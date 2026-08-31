"""
src/db/stylometry_profiles_db.py
--------------------------------
SQLite database manager for Stylometric Profiles.

Persists historical stylometric baselines per student, allowing the system
to compare new submissions against an author's established writing habits.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime

from src.core.stylometry_engine import StylometricProfile

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/stylometry_profiles.db")


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


def initialize_stylometry_db(db_path: Optional[Path] = None) -> None:
    """Create the stylometry profiles database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stylometry_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                profile_data TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_stylometry_user 
            ON stylometry_profiles(user_id)
        """
        )

    logger.info(
        "Stylometry profiles database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def save_profile(
    user_id: str,
    document_id: str,
    profile: StylometricProfile,
    db_path: Optional[Path] = None,
) -> bool:
    """Persist a stylometric profile for a specific user and document."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO stylometry_profiles 
                (user_id, document_id, profile_data, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    document_id,
                    json.dumps(profile.to_dict()),
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to save stylometric profile: %s", e)
        return False


def get_user_baseline(
    user_id: str, db_path: Optional[Path] = None
) -> Optional[StylometricProfile]:
    """Compute the average stylometric baseline for a user across all historical documents.

    Returns:
        A StylometricProfile representing the averaged baseline, or None if no history exists.
    """
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT profile_data FROM stylometry_profiles WHERE user_id = ?",
                (user_id,),
            )
            rows = cursor.fetchall()

            if not rows:
                return None

            # Accumulate sums for averaging
            sums = {
                "type_token_ratio": 0.0,
                "avg_sentence_length": 0.0,
                "sentence_length_variance": 0.0,
                "avg_word_length": 0.0,
                "punctuation_frequency": 0.0,
                "yules_k": 0.0,
            }

            count = len(rows)
            for row in rows:
                data = json.loads(row["profile_data"])
                for key in sums:
                    sums[key] += data.get(key, 0.0)

            # Compute averages
            avg_data = {k: round(v / count, 4) for k, v in sums.items()}
            return StylometricProfile(**avg_data)

    except sqlite3.Error as e:
        logger.error("Failed to compute baseline for user %s: %s", user_id, e)
        return None
