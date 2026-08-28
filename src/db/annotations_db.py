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
src/db/annotations_db.py
------------------------
SQLite database manager for persisting collaborative review annotations.

Handles the storage, retrieval, and lifecycle management of highlights
and comments attached to documents during grading committee sessions.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.annotations import AnnotationRecord, AnnotationType

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/annotations.db")


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Context manager for acquiring and releasing SQLite connections."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_annotations_db(db_path: Optional[Path] = None) -> None:
    """Create the annotations database schema if it doesn't exist."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                username TEXT NOT NULL,
                type TEXT NOT NULL,
                highlight_data TEXT,
                comment_data TEXT,
                parent_annotation_id TEXT,
                is_resolved INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (parent_annotation_id) REFERENCES annotations(id) ON DELETE CASCADE
            )
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_annotations_document
            ON annotations(document_id)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_annotations_user
            ON annotations(user_id)
        """
        )

    logger.info("Annotations database initialized at %s", db_path or DEFAULT_DB_PATH)


def create_annotation(record: AnnotationRecord, db_path: Optional[Path] = None) -> bool:
    """Insert a new annotation into the database."""
    highlight_json = (
        json.dumps(record.highlight.model_dump()) if record.highlight else None
    )
    comment_json = json.dumps(record.comment.model_dump()) if record.comment else None

    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO annotations
                (id, document_id, user_id, username, type, highlight_data, comment_data,
                 parent_annotation_id, is_resolved, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.document_id,
                    record.user_id,
                    record.username,
                    record.type.value,
                    highlight_json,
                    comment_json,
                    record.comment.parent_annotation_id if record.comment else None,
                    1 if record.is_resolved else 0,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to create annotation %s: %s", record.id, e)
        return False


def get_annotations_for_document(
    document_id: str, db_path: Optional[Path] = None
) -> list[AnnotationRecord]:
    """Retrieve all annotations for a specific document."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM annotations WHERE document_id = ? ORDER BY created_at ASC",
                (document_id,),
            )
            rows = cursor.fetchall()

            records = []
            for row in rows:
                highlight_data = (
                    json.loads(row["highlight_data"]) if row["highlight_data"] else None
                )
                comment_data = (
                    json.loads(row["comment_data"]) if row["comment_data"] else None
                )

                records.append(
                    AnnotationRecord(
                        id=row["id"],
                        document_id=row["document_id"],
                        user_id=row["user_id"],
                        username=row["username"],
                        type=AnnotationType(row["type"]),
                        highlight=highlight_data,
                        comment=comment_data,
                        is_resolved=bool(row["is_resolved"]),
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )
            return records
    except sqlite3.Error as e:
        logger.error("Failed to fetch annotations for document %s: %s", document_id, e)
        return []


def resolve_annotation(annotation_id: str, db_path: Optional[Path] = None) -> bool:
    """Mark an annotation as resolved."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                "UPDATE annotations SET is_resolved = 1, updated_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), annotation_id),
            )
            return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error("Failed to resolve annotation %s: %s", annotation_id, e)
        return False


def delete_annotation(annotation_id: str, db_path: Optional[Path] = None) -> bool:
    """Delete an annotation and its replies (via CASCADE)."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM annotations WHERE id = ?", (annotation_id,)
            )
            return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error("Failed to delete annotation %s: %s", annotation_id, e)
        return False
