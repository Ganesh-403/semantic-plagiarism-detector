"""
src/db/reviewer_calibration_db.py
---------------------------------
SQLite database manager for Reviewer Calibration and IRR tracking.

Persists historical review overrides, computes reviewer bias metrics,
and stores Inter-Rater Reliability scores for review committees.
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

DEFAULT_DB_PATH = DATA_DIR / "reviewer_calibration.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_calibration_db(db_path: Optional[Path] = None) -> None:
    """Create the reviewer calibration database schema."""
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reviewer_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                automated_score REAL NOT NULL,
                manual_score REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviewer_metrics (
                reviewer_id TEXT PRIMARY KEY,
                mean_error REAL,
                mean_absolute_error REAL,
                variance REAL,
                calibration_weight REAL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_overrides_reviewer
            ON review_overrides(reviewer_id)
        """)

    logger.info(
        "Reviewer calibration database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_review_override(
    reviewer_id: str,
    document_id: str,
    automated_score: float,
    manual_score: float,
    db_path: Optional[Path] = None,
) -> bool:
    """Log a manual review override against an automated score."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO review_overrides
                (reviewer_id, document_id, automated_score, manual_score, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    reviewer_id,
                    document_id,
                    automated_score,
                    manual_score,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log review override: %s", e)
        return False


def update_reviewer_metrics(
    reviewer_id: str,
    metrics: dict[str, float],
    weight: float,
    db_path: Optional[Path] = None,
) -> bool:
    """Update the aggregated calibration metrics for a reviewer."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reviewer_metrics
                (reviewer_id, mean_error, mean_absolute_error, variance, calibration_weight, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    reviewer_id,
                    metrics["mean_error"],
                    metrics["mean_absolute_error"],
                    metrics["variance"],
                    weight,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to update reviewer metrics for %s: %s", reviewer_id, e)
        return False


def get_reviewer_weight(reviewer_id: str, db_path: Optional[Path] = None) -> float:
    """Retrieve the current calibration weight for a reviewer. Defaults to 1.0."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT calibration_weight FROM reviewer_metrics WHERE reviewer_id = ?",
                (reviewer_id,),
            )
            row = cursor.fetchone()
            return row["calibration_weight"] if row else 1.0
    except sqlite3.Error as e:
        logger.error("Failed to get weight for %s: %s", reviewer_id, e)
        return 1.0


# semantic-plagiarism-detector/src/db/reviewer_calibration_db.py

from typing import List, Dict, Any


class ReviewerCalibrationDB:
    """
    Persists historical review overrides and computes reviewer bias metrics.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        initialize_calibration_db(self.db_path)

    def save_review_override(
        self,
        submission_id: str,
        reviewer_id: str,
        assigned_score: float,
        consensus_score: float,
    ) -> None:
        """Persists a reviewer override event along with its deviation from consensus."""
        if not 0 <= assigned_score <= 1 or not 0 <= consensus_score <= 1:
            raise ValueError("review scores must be between zero and one")
        if not log_review_override(
            reviewer_id, submission_id, consensus_score, assigned_score, self.db_path
        ):
            raise sqlite3.OperationalError("Failed to persist review override")

    def fetch_reviewer_history(self, reviewer_id: str) -> list[dict[str, Any]]:
        """Retrieves all historical review overrides for a specific reviewer."""
        with get_connection(self.db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM review_overrides WHERE reviewer_id = ? ORDER BY id",
                (reviewer_id,),
            ).fetchall()
        return [self._record(row) for row in rows]

    def fetch_all_overrides(self) -> list[dict[str, Any]]:
        """Retrieves entire override dataset for committee IRR calculations."""
        with get_connection(self.db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM review_overrides ORDER BY id"
            ).fetchall()
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row) -> dict[str, Any]:
        return {
            "submission_id": row["document_id"],
            "reviewer_id": row["reviewer_id"],
            "assigned_score": row["manual_score"],
            "consensus_score": row["automated_score"],
            "consensus_deviation": row["manual_score"] - row["automated_score"],
        }
