"""
src/db/patchwriting_logs_db.py
------------------------------
SQLite database manager for Patchwriting Detection Logs.

Persists detected structural clones and the specific POS patterns matched,
allowing administrators to review mosaic plagiarism cases.
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

DEFAULT_DB_PATH = DATA_DIR / "patchwriting_logs.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_patchwriting_db(db_path: Optional[Path] = None) -> None:
    """Create the patchwriting logs database schema."""
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patchwriting_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_a_id TEXT NOT NULL,
                document_b_id TEXT NOT NULL,
                syntactic_jaccard REAL NOT NULL,
                ngram_overlap REAL NOT NULL,
                is_flagged INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_patchwriting_docs
            ON patchwriting_logs(document_a_id, document_b_id)
        """)

    logger.info(
        "Patchwriting logs database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_patchwriting_detection(
    doc_a_id: str,
    doc_b_id: str,
    jaccard: float,
    ngram_overlap: float,
    is_flagged: bool,
    db_path: Optional[Path] = None,
) -> bool:
    """Log a patchwriting detection event."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO patchwriting_logs
                (document_a_id, document_b_id, syntactic_jaccard, ngram_overlap, is_flagged, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_a_id,
                    doc_b_id,
                    jaccard,
                    ngram_overlap,
                    1 if is_flagged else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log patchwriting detection: %s", e)
        return False


# semantic-plagiarism-detector/src/db/patchwriting_logs_db.py

from typing import List, Dict, Any
from datetime import datetime


class PatchwritingLogsDB:
    """
    Logs detected structural clones and the specific POS patterns matched
    during mosaic plagiarism detection scans.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        initialize_patchwriting_db(self.db_path)
        with get_connection(self.db_path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS structural_clone_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    submission_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    metrics_json TEXT NOT NULL
                )
            """)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_structural_clone_submission ON structural_clone_logs(submission_id)"
            )

    def log_structural_clone(
        self,
        submission_id: str,
        source_id: str,
        similarity_score: float,
        metrics: dict[str, Any],
    ) -> None:
        """Persists a structural clone detection record."""
        if not 0 <= similarity_score <= 1:
            raise ValueError("similarity_score must be between zero and one")
        with get_connection(self.db_path) as connection:
            connection.execute(
                "INSERT INTO structural_clone_logs (timestamp, submission_id, source_id, similarity_score, metrics_json) VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.utcnow().isoformat(),
                    submission_id,
                    source_id,
                    similarity_score,
                    json.dumps(metrics, allow_nan=False),
                ),
            )

    def fetch_logs_by_submission(self, submission_id: str) -> list[dict[str, Any]]:
        """Retrieves all patchwriting logs for a given submission."""
        with get_connection(self.db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM structural_clone_logs WHERE submission_id = ? ORDER BY id",
                (submission_id,),
            ).fetchall()
        return [self._record(row) for row in rows]

    def fetch_all_logs(self) -> list[dict[str, Any]]:
        """Retrieves all recorded patchwriting logs."""
        with get_connection(self.db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM structural_clone_logs ORDER BY id"
            ).fetchall()
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row) -> dict[str, Any]:
        return {
            "timestamp": row["timestamp"],
            "submission_id": row["submission_id"],
            "source_id": row["source_id"],
            "similarity_score": row["similarity_score"],
            "metrics": json.loads(row["metrics_json"]),
        }
