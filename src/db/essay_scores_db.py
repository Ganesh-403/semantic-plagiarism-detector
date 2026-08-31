"""
src/db/essay_scores_db.py
-------------------------
SQLite database manager for Essay Scores and Rubrics.

Persists trait scores, holistic grades, and custom rubric templates.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime

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
