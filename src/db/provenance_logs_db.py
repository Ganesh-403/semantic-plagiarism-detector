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
src/db/provenance_logs_db.py
----------------------------
SQLite database manager for Document Provenance Logs.

Persists forensic flags, metadata extractions, and historical provenance
baselines for audit and anomaly detection.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/provenance_logs.db")


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
