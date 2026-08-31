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

DEFAULT_DB_PATH = Path("data/patchwriting_logs.db")


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
        
    logger.info("Patchwriting logs database initialized at %s", db_path or DEFAULT_DB_PATH)


def log_patchwriting_detection(
    doc_a_id: str,
    doc_b_id: str,
    jaccard: float,
    ngram_overlap: float,
    is_flagged: bool,
    db_path: Optional[Path] = None
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
                (doc_a_id, doc_b_id, jaccard, ngram_overlap, 1 if is_flagged else 0, datetime.utcnow().isoformat())
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
    def __init__(self):
        self.logs_store: list[dict[str, Any]] = []

    def log_structural_clone(self, submission_id: str, source_id: str, similarity_score: float, metrics: dict[str, Any]) -> None:
        """Persists a structural clone detection record."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "submission_id": submission_id,
            "source_id": source_id,
            "similarity_score": similarity_score,
            "metrics": metrics
        }
        self.logs_store.append(record)

    def fetch_logs_by_submission(self, submission_id: str) -> list[dict[str, Any]]:
        """Retrieves all patchwriting logs for a given submission."""
        return [log for log in self.logs_store if log["submission_id"] == submission_id]

    def fetch_all_logs(self) -> list[dict[str, Any]]:
        """Retrieves all recorded patchwriting logs."""
        return self.logs_store
