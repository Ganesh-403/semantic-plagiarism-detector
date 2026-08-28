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

"""Data access layer for the pattern recognition & prediction system.

Provides CRUD operations for plagiarism patterns, evolution snapshots,
document risk scores, and proactive recommendations backed by SQLite.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.app_config import CORPUS_DB_PATH, FALLBACK_DATA_DIR
from src.core.concurrency import with_sqlite_retry
from src.db.base import BaseRepository
from src.db.migrations import migrate_corpus_database

DEFAULT_DB_PATH = CORPUS_DB_PATH


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_connection(db_path: str | Path) -> sqlite3.Connection:
    abs_path = os.path.abspath(str(db_path))
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        conn = sqlite3.connect(abs_path)
    except (sqlite3.OperationalError, OSError, PermissionError):
        fallback_path = str(FALLBACK_DATA_DIR / os.path.basename(abs_path))
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        conn = sqlite3.connect(fallback_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        migrate_corpus_database(conn)
    except Exception:
        pass
    return conn


class PatternRepository(BaseRepository):
    """Repository for pattern recognition data access."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        super().__init__(db_path)

    # ── Pattern CRUD ───────────────────────────────────────────────────────

    @with_sqlite_retry
    def upsert_pattern(
        self,
        pattern_id: str,
        pattern_type: str,
        document_group: list[str],
        avg_similarity: float,
        occurrence_count: int,
        confidence_score: float,
        severity: str,
        first_seen: str,
        last_seen: str,
        description: str = "",
        author_group: list[str] | None = None,
        assignment_title: str | None = None,
        class_section: str | None = None,
        status: str = "active",
    ) -> None:
        with closing(_get_connection(self._db_path)) as conn:
            conn.execute(
                """
                INSERT INTO plagiarism_patterns (
                    pattern_id, pattern_type, description, document_group,
                    author_group, assignment_title, class_section,
                    avg_similarity, occurrence_count, confidence_score,
                    severity, first_seen, last_seen, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pattern_id) DO UPDATE SET
                    occurrence_count = excluded.occurrence_count,
                    avg_similarity = excluded.avg_similarity,
                    confidence_score = excluded.confidence_score,
                    severity = excluded.severity,
                    last_seen = excluded.last_seen,
                    status = excluded.status
                """,
                (
                    pattern_id,
                    pattern_type,
                    description,
                    json.dumps(document_group),
                    json.dumps(author_group) if author_group else None,
                    assignment_title,
                    class_section,
                    avg_similarity,
                    occurrence_count,
                    confidence_score,
                    severity,
                    first_seen,
                    last_seen,
                    status,
                ),
            )
            conn.commit()

    @with_sqlite_retry
    def get_patterns(
        self,
        status: str | None = None,
        pattern_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        with closing(_get_connection(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            where_clauses = []
            params: list[Any] = []
            if status:
                where_clauses.append("status = ?")
                params.append(status)
            if pattern_type:
                where_clauses.append("pattern_type = ?")
                params.append(pattern_type)
            where_sql = (
                ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            )
            rows = conn.execute(
                f"""  # nosec
                SELECT * FROM plagiarism_patterns
                {where_sql}
                ORDER BY confidence_score DESC, occurrence_count DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
            return [dict(r) for r in rows]

    @with_sqlite_retry
    def get_pattern_by_id(self, pattern_id: str) -> dict[str, Any] | None:
        with closing(_get_connection(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM plagiarism_patterns WHERE pattern_id = ?",
                (pattern_id,),
            ).fetchone()
            return dict(row) if row else None

    @with_sqlite_retry
    def update_pattern_status(self, pattern_id: str, status: str) -> bool:
        with closing(_get_connection(self._db_path)) as conn:
            cursor = conn.execute(
                "UPDATE plagiarism_patterns SET status = ? WHERE pattern_id = ?",
                (status, pattern_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    @with_sqlite_retry
    def get_pattern_summary(self) -> dict[str, Any]:
        with closing(_get_connection(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM plagiarism_patterns"
            ).fetchone()["cnt"]
            by_type = conn.execute(
                "SELECT pattern_type, COUNT(*) as cnt FROM plagiarism_patterns "
                "GROUP BY pattern_type"
            ).fetchall()
            by_severity = conn.execute(
                "SELECT severity, COUNT(*) as cnt FROM plagiarism_patterns "
                "GROUP BY severity"
            ).fetchall()
            by_status = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM plagiarism_patterns "
                "GROUP BY status"
            ).fetchall()
            return {
                "total": total,
                "by_type": {r["pattern_type"]: r["cnt"] for r in by_type},
                "by_severity": {r["severity"]: r["cnt"] for r in by_severity},
                "by_status": {r["status"]: r["cnt"] for r in by_status},
            }

    # ── Evolution ──────────────────────────────────────────────────────────

    @with_sqlite_retry
    def record_evolution_snapshot(
        self,
        pattern_id: str,
        occurrence_count: int,
        avg_similarity: float,
        confidence_score: float,
        drift_score: float = 0.0,
        snapshot_date: str | None = None,
    ) -> None:
        date = snapshot_date or _utc_now_iso()
        with closing(_get_connection(self._db_path)) as conn:
            conn.execute(
                """
                INSERT INTO pattern_evolution (
                    pattern_id, snapshot_date, occurrence_count,
                    avg_similarity, confidence_score, drift_score
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    pattern_id,
                    date,
                    occurrence_count,
                    avg_similarity,
                    confidence_score,
                    drift_score,
                ),
            )
            conn.commit()

    @with_sqlite_retry
    def get_evolution_timeseries(
        self, pattern_id: str, days: int = 90
    ) -> list[dict[str, Any]]:
        with closing(_get_connection(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM pattern_evolution
                WHERE pattern_id = ?
                  AND snapshot_date >= datetime('now', '-' || ? || ' days')
                ORDER BY snapshot_date ASC
                """,
                (pattern_id, days),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Risk Scores ────────────────────────────────────────────────────────

    @with_sqlite_retry
    def upsert_risk_score(
        self,
        document_name: str,
        risk_score: float,
        risk_level: str,
        contributing_factors: list[str] | None = None,
        model_version: str | None = None,
    ) -> None:
        factors_json = (
            json.dumps(contributing_factors) if contributing_factors else None
        )
        with closing(_get_connection(self._db_path)) as conn:
            conn.execute(
                """
                INSERT INTO document_risk_scores (
                    document_name, risk_score, risk_level,
                    contributing_factors, model_version, scored_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_name) DO UPDATE SET
                    risk_score = excluded.risk_score,
                    risk_level = excluded.risk_level,
                    contributing_factors = excluded.contributing_factors,
                    model_version = excluded.model_version,
                    scored_at = excluded.scored_at
                """,
                (
                    document_name,
                    risk_score,
                    risk_level,
                    factors_json,
                    model_version,
                    _utc_now_iso(),
                ),
            )
            conn.commit()

    @with_sqlite_retry
    def get_risk_score(self, document_name: str) -> dict[str, Any] | None:
        with closing(_get_connection(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM document_risk_scores WHERE document_name = ?",
                (document_name,),
            ).fetchone()
            return dict(row) if row else None

    @with_sqlite_retry
    def get_high_risk_documents(self, threshold: float = 0.7) -> list[dict[str, Any]]:
        with closing(_get_connection(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM document_risk_scores
                WHERE risk_score >= ?
                ORDER BY risk_score DESC
                """,
                (threshold,),
            ).fetchall()
            return [dict(r) for r in rows]

    @with_sqlite_retry
    def get_risk_distribution(self) -> dict[str, int]:
        with closing(_get_connection(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT risk_level, COUNT(*) as cnt FROM document_risk_scores GROUP BY risk_level"
            ).fetchall()
            return {r["risk_level"]: r["cnt"] for r in rows}

    # ── Recommendations ────────────────────────────────────────────────────

    @with_sqlite_retry
    def create_recommendation(
        self,
        recommendation_id: str,
        recommendation_type: str,
        priority: int,
        target: str,
        message: str,
        pattern_id: str | None = None,
        action_items: list[str] | None = None,
    ) -> None:
        items_json = json.dumps(action_items) if action_items else None
        with closing(_get_connection(self._db_path)) as conn:
            conn.execute(
                """
                INSERT INTO proactive_recommendations (
                    recommendation_id, pattern_id, recommendation_type,
                    priority, target, message, action_items, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                ON CONFLICT(recommendation_id) DO NOTHING
                """,
                (
                    recommendation_id,
                    pattern_id,
                    recommendation_type,
                    priority,
                    target,
                    message,
                    items_json,
                    _utc_now_iso(),
                ),
            )
            conn.commit()

    @with_sqlite_retry
    def get_recommendations(
        self,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        with closing(_get_connection(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM proactive_recommendations
                    WHERE status = ?
                    ORDER BY priority ASC, created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (status, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM proactive_recommendations
                    ORDER BY priority ASC, created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
            return [dict(r) for r in rows]

    @with_sqlite_retry
    def update_recommendation_status(self, recommendation_id: str, status: str) -> bool:
        with closing(_get_connection(self._db_path)) as conn:
            cursor = conn.execute(
                "UPDATE proactive_recommendations SET status = ? WHERE recommendation_id = ?",
                (status, recommendation_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    @with_sqlite_retry
    def get_recommendation_stats(self) -> dict[str, int]:
        with closing(_get_connection(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM proactive_recommendations GROUP BY status"
            ).fetchall()
            return {r["status"]: r["cnt"] for r in rows}
