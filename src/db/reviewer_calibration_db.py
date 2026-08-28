# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
src/db/reviewer_calibration_db.py
---------------------------------
SQLite database manager for Reviewer Calibration and IRR tracking.

Persists historical review overrides, computes reviewer bias metrics,
and stores Inter-Rater Reliability scores for review committees.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/reviewer_calibration.db")


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


def initialize_calibration_db(db_path: Optional[Path] = None) -> None:
    """Create the reviewer calibration database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS review_overrides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reviewer_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                automated_score REAL NOT NULL,
                manual_score REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviewer_metrics (
                reviewer_id TEXT PRIMARY KEY,
                mean_error REAL,
                mean_absolute_error REAL,
                variance REAL,
                calibration_weight REAL,
                updated_at TEXT NOT NULL
            )
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_overrides_reviewer
            ON review_overrides(reviewer_id)
        """
        )

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

from typing import Any, Dict, List


class ReviewerCalibrationDB:
    """
    Persists historical review overrides and computes reviewer bias metrics.
    """

    def __init__(self):
        # In-memory storage mock for demonstration (replace with SQL/ORM in production)
        self.overrides_store: list[dict[str, Any]] = []

    def save_review_override(
        self,
        submission_id: str,
        reviewer_id: str,
        assigned_score: float,
        consensus_score: float,
    ) -> None:
        """Persists a reviewer override event along with its deviation from consensus."""
        deviation = assigned_score - consensus_score
        record = {
            "submission_id": submission_id,
            "reviewer_id": reviewer_id,
            "assigned_score": assigned_score,
            "consensus_score": consensus_score,
            "consensus_deviation": deviation,
        }
        self.overrides_store.append(record)

    def fetch_reviewer_history(self, reviewer_id: str) -> list[dict[str, Any]]:
        """Retrieves all historical review overrides for a specific reviewer."""
        return [r for r in self.overrides_store if r["reviewer_id"] == reviewer_id]

    def fetch_all_overrides(self) -> list[dict[str, Any]]:
        """Retrieves entire override dataset for committee IRR calculations."""
        return self.overrides_store
