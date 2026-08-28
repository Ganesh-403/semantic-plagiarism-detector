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
src/db/watermark_logs_db.py
---------------------------
SQLite database manager for tracking document watermarks.

Maintains an audit trail mapping generated watermark IDs to specific
users, documents, and timestamps. This allows administrators to trace
leaked documents back to their original recipients.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/watermark_logs.db")


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


def initialize_watermark_db(db_path: Optional[Path] = None) -> None:
    """Create the watermark logs database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watermark_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                watermark_id TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                document_hash TEXT NOT NULL,
                strategy TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_watermark_user
            ON watermark_logs(user_id)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_watermark_doc
            ON watermark_logs(document_hash)
        """
        )

    logger.info("Watermark logs database initialized at %s", db_path or DEFAULT_DB_PATH)


def log_watermark_generation(
    watermark_id: str,
    user_id: str,
    document_hash: str,
    strategy: str = "append",
    db_path: Optional[Path] = None,
) -> bool:
    """Record the generation of a new watermark."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO watermark_logs
                (watermark_id, user_id, document_hash, strategy, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    watermark_id,
                    user_id,
                    document_hash,
                    strategy,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log watermark %s: %s", watermark_id, e)
        return False


def identify_leak_source(
    watermark_id: str, db_path: Optional[Path] = None
) -> Optional[dict[str, Any]]:
    """Look up the user and document associated with a leaked watermark ID."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM watermark_logs WHERE watermark_id = ?", (watermark_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    except sqlite3.Error as e:
        logger.error("Failed to identify leak source for %s: %s", watermark_id, e)
        return None
