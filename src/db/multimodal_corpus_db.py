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
src/db/multimodal_corpus_db.py
------------------------------
SQLite database manager for Multimodal Plagiarism Corpus.

Persists image perceptual hashes and equation ASTs for cross-document matching.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/multimodal_corpus.db")


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
