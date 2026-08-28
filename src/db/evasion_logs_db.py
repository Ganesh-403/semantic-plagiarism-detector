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
src/db/evasion_logs_db.py
-------------------------
SQLite database manager for Test-Case Evasion Logs.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
