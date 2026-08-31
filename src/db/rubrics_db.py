"""
src/db/rubrics_db.py
--------------------
SQLite database manager for Rubrics and Grading Records.

Persists custom rubric templates and historical grading records,
allowing instructors to reuse rubrics across assignments and track
student progress over time.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime

from src.core.rubric_engine import Rubric, RubricCriterion, CriterionType

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/rubrics.db")


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


def initialize_rubrics_db(db_path: Optional[Path] = None) -> None:
    """Create the rubrics database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rubrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                criteria_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS grading_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rubric_id INTEGER NOT NULL,
                document_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                result_json TEXT NOT NULL,
                graded_at TEXT NOT NULL,
                FOREIGN KEY (rubric_id) REFERENCES rubrics(id)
            )
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_grading_user 
            ON grading_records(user_id)
        """
        )

    logger.info("Rubrics database initialized at %s", db_path or DEFAULT_DB_PATH)


def save_rubric(rubric: Rubric, db_path: Optional[Path] = None) -> bool:
    """Persist a rubric template to the database."""
    criteria_dicts = [
        {
            "name": c.name,
            "type": c.type.value,
            "weight": c.weight,
            "max_points": c.max_points,
            "thresholds": c.thresholds,
        }
        for c in rubric.criteria
    ]

    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO rubrics (name, criteria_json, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    rubric.name,
                    json.dumps(criteria_dicts),
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to save rubric %s: %s", rubric.name, e)
        return False


def get_rubric(name: str, db_path: Optional[Path] = None) -> Optional[Rubric]:
    """Retrieve a rubric template by name."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute("SELECT * FROM rubrics WHERE name = ?", (name,))
            row = cursor.fetchone()
            if not row:
                return None

            criteria_data = json.loads(row["criteria_json"])
            criteria = [
                RubricCriterion(
                    name=c["name"],
                    type=CriterionType(c["type"]),
                    weight=c["weight"],
                    max_points=c["max_points"],
                    thresholds=c.get("thresholds", {}),
                )
                for c in criteria_data
            ]
            return Rubric(name=row["name"], criteria=criteria)

    except sqlite3.Error as e:
        logger.error("Failed to retrieve rubric %s: %s", name, e)
        return None
