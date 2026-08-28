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

import hashlib
from datetime import datetime

# In-memory document storage mock engine; replace with your active Prisma/Supabase models seamlessly
VERSION_LINEAGE_CACHE = {}


def register_document_draft(user_id: str, document_text: str, filename: str) -> dict:
    """
    Saves and maps draft relationships inside the version control schema.

    Expected Database Schema Target:
    CREATE TABLE document_drafts (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID NOT NULL,
        doc_hash CHAR(64) UNIQUE NOT NULL,
        parent_hash CHAR(64) REFERENCES document_drafts(doc_hash),
        version_number INT DEFAULT 1,
        filename VARCHAR(255),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """
    doc_hash = hashlib.sha256(document_text.encode("utf-8")).hexdigest()

    # Query history to locate existing user versions to build parent connections
    user_history = [
        v for v in VERSION_LINEAGE_CACHE.values() if v["user_id"] == user_id
    ]

    parent_hash = None
    version_number = 1

    if user_history:
        # Sort to find the immediate prior draft configuration
        user_history.sort(key=lambda x: x["version_number"], reverse=True)
        latest_draft = user_history[0]

        if latest_draft["doc_hash"] != doc_hash:
            parent_hash = latest_draft["doc_hash"]
            version_number = latest_draft["version_number"] + 1
        else:
            return latest_draft  # Document already archived, bypass insertion loops

    draft_record = {
        "user_id": user_id,
        "doc_hash": doc_hash,
        "parent_hash": parent_hash,
        "version_number": version_number,
        "filename": filename,
        "created_at": datetime.utcnow().isoformat(),
    }

    VERSION_LINEAGE_CACHE[doc_hash] = draft_record
    return draft_record


"""
src/db/version_history_db.py
----------------------------
SQLite database manager for tracking document version lineage.

Maps document hashes to user IDs and tracks parent/child draft
relationships, enabling the historical diff analysis engine.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/version_history.db")


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


def initialize_version_history_db(db_path: Optional[Path] = None) -> None:
    """Create the version history database schema."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_hash TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                assignment_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                parent_hash TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (parent_hash) REFERENCES document_versions(document_hash)
            )
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_versions_user_assignment
            ON document_versions(user_id, assignment_id)
        """
        )

    logger.info(
        "Version history database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def register_document_version(
    document_hash: str,
    user_id: str,
    assignment_id: str,
    parent_hash: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Register a new document version and compute its version number.

    Args:
        document_hash: SHA-256 hash of the document content.
        user_id: ID of the user who uploaded the document.
        assignment_id: ID of the assignment this draft belongs to.
        parent_hash: Hash of the previous draft (if any).

    Returns:
        The computed version number (1-indexed).
    """
    try:
        with get_connection(db_path) as conn:
            # Determine the next version number
            cursor = conn.execute(
                """
                SELECT MAX(version_number) FROM document_versions
                WHERE user_id = ? AND assignment_id = ?
                """,
                (user_id, assignment_id),
            )
            max_version = cursor.fetchone()[0] or 0
            next_version = max_version + 1

            conn.execute(
                """
                INSERT INTO document_versions
                (document_hash, user_id, assignment_id, version_number, parent_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_hash,
                    user_id,
                    assignment_id,
                    next_version,
                    parent_hash,
                    datetime.utcnow().isoformat(),
                ),
            )
            return next_version
    except sqlite3.Error as e:
        logger.error("Failed to register document version: %s", e)
        return -1


def get_version_lineage(
    user_id: str, assignment_id: str, db_path: Optional[Path] = None
) -> list[dict[str, Any]]:
    """Retrieve the complete version lineage for a specific assignment."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM document_versions
                WHERE user_id = ? AND assignment_id = ?
                ORDER BY version_number ASC
                """,
                (user_id, assignment_id),
            )
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error("Failed to fetch version lineage: %s", e)
        return []
