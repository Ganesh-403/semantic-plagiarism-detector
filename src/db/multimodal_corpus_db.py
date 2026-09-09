"""
src/db/multimodal_corpus_db.py
------------------------------
SQLite database manager for Multimodal Plagiarism Corpus.

Persists image perceptual hashes and equation ASTs for cross-document matching.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

from src.core.app_config import DATA_DIR
from src.db.connection import get_connection as managed_connection

DEFAULT_DB_PATH = DATA_DIR / "multimodal_corpus.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Use shared connection settings and commit or roll back one transaction."""
    with managed_connection(db_path or DEFAULT_DB_PATH) as connection:
        with connection:
            yield connection


def initialize_multimodal_db(db_path: Optional[Path] = None) -> None:
    """Create the multimodal corpus database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS image_hashes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                image_id TEXT NOT NULL,
                phash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS equation_asts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                equation_id TEXT NOT NULL,
                normalized_tokens TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """
        )
    logger.info(
        "Multimodal corpus database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def store_image_hash(
    document_id: str, image_id: str, phash: str, db_path: Optional[Path] = None
) -> bool:
    """Store an image perceptual hash."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO image_hashes (document_id, image_id, phash, created_at) VALUES (?, ?, ?, ?)",
                (document_id, image_id, phash, datetime.utcnow().isoformat()),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to store image hash: %s", e)
        return False
