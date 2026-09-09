"""
src/db/concept_graphs_db.py
---------------------------
SQLite database manager for Concept Graph Logs.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)
from src.core.app_config import DATA_DIR
from src.db.connection import get_connection as managed_connection

DEFAULT_DB_PATH = DATA_DIR / "concept_graphs.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_concept_graphs_db(db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS concept_graph_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_a_id TEXT NOT NULL,
                doc_b_id TEXT NOT NULL,
                conceptual_score REAL NOT NULL,
                is_conceptual_plagiarism INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
        """
        )
    logger.info("Concept graphs database initialized at %s", db_path or DEFAULT_DB_PATH)


def log_concept_alignment(
    doc_a_id: str,
    doc_b_id: str,
    conceptual_score: float,
    is_plagiarism: bool,
    db_path: Optional[Path] = None,
) -> bool:
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO concept_graph_logs (doc_a_id, doc_b_id, conceptual_score, is_conceptual_plagiarism, analyzed_at) VALUES (?, ?, ?, ?, ?)",
                (
                    doc_a_id,
                    doc_b_id,
                    conceptual_score,
                    1 if is_plagiarism else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log concept alignment: %s", e)
        return False
