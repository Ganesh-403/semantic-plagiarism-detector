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
src/db/essay_scores_db.py
-------------------------
SQLite database manager for Essay Scores and Rubrics.

Persists trait scores, holistic grades, and custom rubric templates.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/essay_scores.db")


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


def initialize_essay_scores_db(db_path: Optional[Path] = None) -> None:
    """Create the essay scores database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS essay_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                rubric_name TEXT NOT NULL,
                final_grade REAL NOT NULL,
                traits_json TEXT NOT NULL,
                criterion_scores_json TEXT NOT NULL,
                scored_at TEXT NOT NULL
            )
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_essay_doc
            ON essay_scores(document_id)
        """
        )
    logger.info("Essay scores database initialized at %s", db_path or DEFAULT_DB_PATH)


def log_essay_score(
    document_id: str,
    rubric_name: str,
    final_grade: float,
    traits: Dict[str, Any],
    criterion_scores: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> bool:
    """Persist an essay scoring result."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO essay_scores
                (document_id, rubric_name, final_grade, traits_json, criterion_scores_json, scored_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    rubric_name,
                    final_grade,
                    json.dumps(traits),
                    json.dumps(criterion_scores),
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log essay score for %s: %s", document_id, e)
        return False
