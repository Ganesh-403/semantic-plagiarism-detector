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
src/db/watermark_verification_db.py
-----------------------------------
SQLite database manager for AI Watermark Verification Logs.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/watermark_verification.db")


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


def initialize_watermark_verification_db(db_path: Optional[Path] = None) -> None:
    """Create the watermark verification database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watermark_verification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                z_score REAL NOT NULL,
                p_value REAL NOT NULL,
                is_watermarked INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info(
        "Watermark verification database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_watermark_verification(
    document_id: str,
    z_score: float,
    p_value: float,
    is_watermarked: bool,
    db_path: Optional[Path] = None,
) -> bool:
    """Persist a watermark verification result."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO watermark_verification_logs
                (document_id, z_score, p_value, is_watermarked, analyzed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    z_score,
                    p_value,
                    1 if is_watermarked else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log watermark verification: %s", e)
        return False
