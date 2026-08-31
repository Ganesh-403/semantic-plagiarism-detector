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
src/db/version_repo.py
---------------------
Enhanced document versioning repository with diff tracking, similarity
trends across drafts, and analytics.

Extends the basic version_history_db with:
  • Content snapshot storage (text snapshots per version)
  • Pairwise diff summaries between adjacent versions
  • Similarity trend tracking across a document's lifecycle
  • Lineage graph queries (parent→child chains)
  • Bulk ingestion and maintenance operations
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/version_repo.db")


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


@contextmanager
def get_connection(db_path: Path | None = None):
    """Context manager for SQLite connections with WAL and FK enforcement."""
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


def _db_path() -> Path:
    """Return the configured database path."""
    return DEFAULT_DB_PATH


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


def init_version_repo_db(db_path: Path | None = None) -> None:
    """Create all tables for the version repository."""
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS document_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                document_hash   TEXT    UNIQUE NOT NULL,
                user_id         TEXT    NOT NULL,
                assignment_id   TEXT    NOT NULL,
                filename        TEXT    NOT NULL DEFAULT 'untitled',
                content_text    TEXT    NOT NULL DEFAULT '',
                content_length  INTEGER NOT NULL DEFAULT 0,
                word_count      INTEGER NOT NULL DEFAULT 0,
                version_number  INTEGER NOT NULL DEFAULT 1,
                parent_hash     TEXT,
                similarity_to_parent REAL DEFAULT NULL,
                created_at      TEXT    NOT NULL,
                FOREIGN KEY (parent_hash) REFERENCES document_snapshots(document_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_snap_user_assignment
                ON document_snapshots(user_id, assignment_id);

            CREATE INDEX IF NOT EXISTS idx_snap_hash
                ON document_snapshots(document_hash);

            CREATE TABLE IF NOT EXISTS version_diffs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_hash     TEXT    NOT NULL,
                child_hash      TEXT    NOT NULL UNIQUE,
                similarity      REAL    NOT NULL DEFAULT 0.0,
                added_words     INTEGER NOT NULL DEFAULT 0,
                removed_words   INTEGER NOT NULL DEFAULT 0,
                changed_words   INTEGER NOT NULL DEFAULT 0,
                jaccard_index   REAL    NOT NULL DEFAULT 0.0,
                diff_summary    TEXT    NOT NULL DEFAULT '{}',
                computed_at     TEXT    NOT NULL,
                FOREIGN KEY (parent_hash) REFERENCES document_snapshots(document_hash),
                FOREIGN KEY (child_hash)  REFERENCES document_snapshots(document_hash)
            );

            CREATE TABLE IF NOT EXISTS version_lineage (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                assignment_id   TEXT    NOT NULL,
                user_id         TEXT    NOT NULL,
                head_hash       TEXT    NOT NULL,
                total_versions  INTEGER NOT NULL DEFAULT 0,
                avg_similarity  REAL    NOT NULL DEFAULT 0.0,
                min_similarity  REAL    NOT NULL DEFAULT 1.0,
                max_similarity  REAL    NOT NULL DEFAULT 0.0,
                first_created   TEXT    NOT NULL,
                last_created    TEXT    NOT NULL,
                FOREIGN KEY (head_hash) REFERENCES document_snapshots(document_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_lineage_assignment
                ON version_lineage(assignment_id, user_id);
        """
        )
    logger.info("Version repo DB initialized at %s", db_path or DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# DocumentSnapshotRepository
# ---------------------------------------------------------------------------


class DocumentSnapshotRepository:
    """CRUD + analytics for document version snapshots."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path

    def _conn(self):
        return get_connection(self._db_path)

    # -- Create ---------------------------------------------------------------

    def register_version(
        self,
        user_id: str,
        assignment_id: str,
        filename: str,
        content_text: str,
        parent_hash: str | None = None,
        similarity_to_parent: float | None = None,
    ) -> dict[str, Any]:
        """Register a new document version and return the snapshot record."""
        content_hash = hashlib.sha256(content_text.encode("utf-8")).hexdigest()
        word_count = len(content_text.split())
        now = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:
            # Determine version number
            cursor = conn.execute(
                """SELECT MAX(version_number) FROM document_snapshots
                   WHERE user_id = ? AND assignment_id = ?""",
                (user_id, assignment_id),
            )
            max_ver = cursor.fetchone()[0] or 0
            next_ver = max_ver + 1

            conn.execute(
                """INSERT INTO document_snapshots
                   (document_hash, user_id, assignment_id, filename,
                    content_text, content_length, word_count,
                    version_number, parent_hash, similarity_to_parent, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    content_hash,
                    user_id,
                    assignment_id,
                    filename,
                    content_text,
                    len(content_text),
                    word_count,
                    next_ver,
                    parent_hash,
                    similarity_to_parent,
                    now,
                ),
            )

            # Update lineage
            self._upsert_lineage(
                conn,
                assignment_id,
                user_id,
                content_hash,
                now,
                similarity_to_parent,
            )

            return {
                "document_hash": content_hash,
                "user_id": user_id,
                "assignment_id": assignment_id,
                "filename": filename,
                "content_length": len(content_text),
                "word_count": word_count,
                "version_number": next_ver,
                "parent_hash": parent_hash,
                "similarity_to_parent": similarity_to_parent,
                "created_at": now,
            }

    def register_diff(
        self,
        parent_hash: str,
        child_hash: str,
        similarity: float,
        added_words: int,
        removed_words: int,
        changed_words: int,
        jaccard_index: float,
        diff_summary: dict | None = None,
    ) -> int:
        """Store a pairwise diff record between two versions."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT OR REPLACE INTO version_diffs
                   (parent_hash, child_hash, similarity, added_words,
                    removed_words, changed_words, jaccard_index,
                    diff_summary, computed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    parent_hash,
                    child_hash,
                    similarity,
                    added_words,
                    removed_words,
                    changed_words,
                    jaccard_index,
                    json.dumps(diff_summary or {}),
                    now,
                ),
            )
            return cursor.lastrowid

    # -- Read -----------------------------------------------------------------

    def get_snapshot(self, doc_hash: str) -> dict[str, Any] | None:
        """Retrieve a single snapshot by hash."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM document_snapshots WHERE document_hash = ?",
                (doc_hash,),
            ).fetchone()
            return dict(row) if row else None

    def list_versions(
        self,
        user_id: str | None = None,
        assignment_id: str | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> dict[str, Any]:
        """List document snapshots with optional filtering and pagination."""
        conditions: list[str] = []
        params: list[Any] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if assignment_id:
            conditions.append("assignment_id = ?")
            params.append(assignment_id)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""

        with self._conn() as conn:
            count_row = conn.execute(
                f"SELECT COUNT(*) FROM document_snapshots{where}",  # nosec
                params,  # nosec
            ).fetchone()
            total = count_row[0] if count_row else 0

            offset = (page - 1) * per_page
            cursor = conn.execute(
                f"""SELECT * FROM document_snapshots{where}  # nosec
                    ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                params + [per_page, offset],
            )
            items = [dict(r) for r in cursor.fetchall()]

        total_pages = max(1, (total + per_page - 1) // per_page)
        return {
            "items": items,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1,
        }

    def get_lineage(self, user_id: str, assignment_id: str) -> list[dict[str, Any]]:
        """Get the full version lineage for a user + assignment."""
        with self._conn() as conn:
            cursor = conn.execute(
                """SELECT * FROM document_snapshots
                   WHERE user_id = ? AND assignment_id = ?
                   ORDER BY version_number ASC""",
                (user_id, assignment_id),
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_diff(self, parent_hash: str, child_hash: str) -> dict[str, Any] | None:
        """Get the diff record between two specific versions."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM version_diffs
                   WHERE parent_hash = ? AND child_hash = ?""",
                (parent_hash, child_hash),
            ).fetchone()
            return dict(row) if row else None

    def get_diffs_for_version(self, doc_hash: str) -> list[dict[str, Any]]:
        """Get all diffs involving a specific version (as parent or child)."""
        with self._conn() as conn:
            cursor = conn.execute(
                """SELECT * FROM version_diffs
                   WHERE parent_hash = ? OR child_hash = ?
                   ORDER BY computed_at ASC""",
                (doc_hash, doc_hash),
            )
            return [dict(r) for r in cursor.fetchall()]

    def list_lineages(
        self,
        user_id: str | None = None,
        assignment_id: str | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> dict[str, Any]:
        """List all tracked lineages with optional filtering."""
        conditions: list[str] = []
        params: list[Any] = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if assignment_id:
            conditions.append("assignment_id = ?")
            params.append(assignment_id)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""

        with self._conn() as conn:
            count_row = conn.execute(
                f"SELECT COUNT(*) FROM version_lineage{where}",  # nosec
                params,  # nosec
            ).fetchone()
            total = count_row[0] if count_row else 0

            offset = (page - 1) * per_page
            cursor = conn.execute(
                f"""SELECT * FROM version_lineage{where}  # nosec
                    ORDER BY last_created DESC LIMIT ? OFFSET ?""",
                params + [per_page, offset],
            )
            items = [dict(r) for r in cursor.fetchall()]

        total_pages = max(1, (total + per_page - 1) // per_page)
        return {
            "items": items,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1,
        }

    # -- Analytics ------------------------------------------------------------

    def analytics_summary(self) -> dict[str, Any]:
        """Return aggregate statistics across all versions."""
        with self._conn() as conn:
            total_versions = conn.execute(
                "SELECT COUNT(*) FROM document_snapshots"
            ).fetchone()[0]
            total_lineages = conn.execute(
                "SELECT COUNT(*) FROM version_lineage"
            ).fetchone()[0]
            total_diffs = conn.execute("SELECT COUNT(*) FROM version_diffs").fetchone()[
                0
            ]

            avg_sim_row = conn.execute(
                "SELECT AVG(similarity) FROM version_diffs"
            ).fetchone()
            avg_similarity = avg_sim_row[0] if avg_sim_row[0] else 0.0

            avg_versions_row = conn.execute(
                "SELECT AVG(total_versions) FROM version_lineage"
            ).fetchone()
            avg_versions_per_doc = avg_versions_row[0] if avg_versions_row[0] else 0.0

            unique_users = conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM document_snapshots"
            ).fetchone()[0]

            return {
                "total_versions": total_versions,
                "total_lineages": total_lineages,
                "total_diffs": total_diffs,
                "avg_similarity": round(avg_similarity, 4),
                "avg_versions_per_document": round(avg_versions_per_doc, 2),
                "unique_users": unique_users,
            }

    def similarity_trend(
        self, user_id: str, assignment_id: str
    ) -> list[dict[str, Any]]:
        """Return the similarity trend across versions for a lineage."""
        lineage = self.get_lineage(user_id, assignment_id)
        if len(lineage) < 2:
            return []

        trend: list[dict[str, Any]] = []
        for i in range(1, len(lineage)):
            parent = lineage[i - 1]
            child = lineage[i]
            diff = self.get_diff(parent["document_hash"], child["document_hash"])
            trend.append(
                {
                    "from_version": parent["version_number"],
                    "to_version": child["version_number"],
                    "similarity": diff["similarity"] if diff else None,
                    "added_words": diff["added_words"] if diff else None,
                    "removed_words": diff["removed_words"] if diff else None,
                    "created_at": child["created_at"],
                }
            )
        return trend

    def most_revised_documents(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return documents with the most versions."""
        with self._conn() as conn:
            cursor = conn.execute(
                """SELECT assignment_id, user_id, total_versions,
                          avg_similarity, last_created
                   FROM version_lineage
                   ORDER BY total_versions DESC
                   LIMIT ?""",
                (limit,),
            )
            return [dict(r) for r in cursor.fetchall()]

    def highest_drift_documents(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return documents with the most drift (lowest avg similarity)."""
        with self._conn() as conn:
            cursor = conn.execute(
                """SELECT assignment_id, user_id, total_versions,
                          avg_similarity, last_created
                   FROM version_lineage
                   WHERE total_versions >= 2
                   ORDER BY avg_similarity ASC
                   LIMIT ?""",
                (limit,),
            )
            return [dict(r) for r in cursor.fetchall()]

    # -- Delete ----------------------------------------------------------------

    def delete_version(self, doc_hash: str) -> bool:
        """Delete a specific version snapshot."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM document_snapshots WHERE document_hash = ?",
                (doc_hash,),
            )
            return cursor.rowcount > 0

    def delete_lineage(self, user_id: str, assignment_id: str) -> bool:
        """Delete an entire lineage (all versions for a user + assignment)."""
        with self._conn() as conn:
            conn.execute(
                """DELETE FROM version_diffs WHERE parent_hash IN
                   (SELECT document_hash FROM document_snapshots
                    WHERE user_id = ? AND assignment_id = ?)""",
                (user_id, assignment_id),
            )
            conn.execute(
                """DELETE FROM version_diffs WHERE child_hash IN
                   (SELECT document_hash FROM document_snapshots
                    WHERE user_id = ? AND assignment_id = ?)""",
                (user_id, assignment_id),
            )
            cursor = conn.execute(
                """DELETE FROM document_snapshots
                   WHERE user_id = ? AND assignment_id = ?""",
                (user_id, assignment_id),
            )
            conn.execute(
                """DELETE FROM version_lineage
                   WHERE user_id = ? AND assignment_id = ?""",
                (user_id, assignment_id),
            )
            return cursor.rowcount > 0

    # -- Internal helpers -----------------------------------------------------

    @staticmethod
    def _upsert_lineage(
        conn: sqlite3.Connection,
        assignment_id: str,
        user_id: str,
        head_hash: str,
        now: str,
        similarity: float | None,
    ) -> None:
        """Insert or update the lineage record for a user + assignment."""
        existing = conn.execute(
            """SELECT * FROM version_lineage
               WHERE user_id = ? AND assignment_id = ?""",
            (user_id, assignment_id),
        ).fetchone()

        if existing:
            total = existing["total_versions"] + 1
            sims = [existing["avg_similarity"]]
            if similarity is not None:
                sims.append(similarity)
            avg_sim = sum(sims) / len(sims) if sims else 0.0
            min_sim = min(existing["min_similarity"], similarity or 1.0)
            max_sim = max(existing["max_similarity"], similarity or 0.0)

            conn.execute(
                """UPDATE version_lineage
                   SET head_hash = ?, total_versions = ?,
                       avg_similarity = ?, min_similarity = ?,
                       max_similarity = ?, last_created = ?
                   WHERE user_id = ? AND assignment_id = ?""",
                (
                    head_hash,
                    total,
                    avg_sim,
                    min_sim,
                    max_sim,
                    now,
                    user_id,
                    assignment_id,
                ),
            )
        else:
            conn.execute(
                """INSERT INTO version_lineage
                   (assignment_id, user_id, head_hash, total_versions,
                    avg_similarity, min_similarity, max_similarity,
                    first_created, last_created)
                   VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)""",
                (
                    assignment_id,
                    user_id,
                    head_hash,
                    similarity or 0.0,
                    similarity or 1.0,
                    similarity or 0.0,
                    now,
                    now,
                ),
            )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

version_repo = DocumentSnapshotRepository()
