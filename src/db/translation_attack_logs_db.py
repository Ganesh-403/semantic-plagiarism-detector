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
src/db/translation_attack_logs_db.py
------------------------------------
SQLite database manager for Translation Attack Logs.

Persists detected back-translation attacks, drift metrics, and invariance scores.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/translation_attack_logs.db")


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


def initialize_translation_logs_db(db_path: Optional[Path] = None) -> None:
    """Create the translation attack logs database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS translation_attack_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                lexical_drift REAL NOT NULL,
                structural_variance REAL NOT NULL,
                invariance_score REAL NOT NULL,
                is_obfuscated INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_translation_doc
            ON translation_attack_logs(document_id)
        """
        )
    logger.info(
        "Translation attack logs database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def log_translation_attack(
    document_id: str,
    lexical_drift: float,
    structural_variance: float,
    invariance_score: float,
    is_obfuscated: bool,
    db_path: Optional[Path] = None,
) -> bool:
    """Persist a translation attack detection result."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO translation_attack_logs
                (document_id, lexical_drift, structural_variance, invariance_score, is_obfuscated, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    lexical_drift,
                    structural_variance,
                    invariance_score,
                    1 if is_obfuscated else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log translation attack for %s: %s", document_id, e)
        return False
