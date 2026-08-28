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
src/db/sandbox_logs_db.py
-------------------------
SQLite database manager for Code Execution Sandbox Logs.

Persists execution traces and maps them to submission hashes, allowing
the system to detect behavioral clones across different code submissions.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/sandbox_logs.db")


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


def initialize_sandbox_db(db_path: Optional[Path] = None) -> None:
    """Create the sandbox logs database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                submission_hash TEXT UNIQUE NOT NULL,
                behavioral_hash TEXT NOT NULL,
                trace_json TEXT NOT NULL,
                executed_at TEXT NOT NULL
            )
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_traces_behavioral
            ON execution_traces(behavioral_hash)
        """
        )

    logger.info("Sandbox logs database initialized at %s", db_path or DEFAULT_DB_PATH)


def log_execution_trace(
    submission_hash: str,
    behavioral_hash: str,
    trace_data: dict[str, Any],
    db_path: Optional[Path] = None,
) -> bool:
    """Persist an execution trace and its behavioral hash."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO execution_traces
                (submission_hash, behavioral_hash, trace_json, executed_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    submission_hash,
                    behavioral_hash,
                    json.dumps(trace_data),
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log execution trace for %s: %s", submission_hash, e)
        return False


def find_behavioral_clones(
    behavioral_hash: str,
    exclude_submission_hash: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> list[str]:
    """Find all submission hashes that share the same behavioral fingerprint."""
    try:
        with get_connection(db_path) as conn:
            query = (
                "SELECT submission_hash FROM execution_traces WHERE behavioral_hash = ?"
            )
            params = [behavioral_hash]

            if exclude_submission_hash:
                query += " AND submission_hash != ?"
                params.append(exclude_submission_hash)

            cursor = conn.execute(query, params)
            return [row["submission_hash"] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error("Failed to find behavioral clones for %s: %s", behavioral_hash, e)
        return []
