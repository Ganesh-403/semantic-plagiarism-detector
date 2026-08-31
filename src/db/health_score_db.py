"""
health_score_db.py
------------------
SQLite persistence layer for document health scores.

Stores:
  - Per-document health score snapshots (latest + history)
  - Quality gate decisions
  - Aggregate quality trends over time
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Optional

from src.db.base import BaseRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection pool (mirrors corpus_db / batch_history_db pattern)
# ---------------------------------------------------------------------------
_connection_pool = threading.local()
_all_connections: set[sqlite3.Connection] = set()
_pool_lock = threading.Lock()

import atexit

_DB_PATH: str | os.PathLike = "plagiarism_detector.db"


def _cleanup_all_connections() -> None:
    with _pool_lock:
        for conn in _all_connections:
            try:
                conn.close()
            except Exception:
                pass
        _all_connections.clear()


atexit.register(_cleanup_all_connections)


def _pool() -> dict[str, sqlite3.Connection]:
    pool = getattr(_connection_pool, "connections", None)
    if pool is None:
        pool = {}
        _connection_pool.connections = pool
    return pool


@contextmanager
def _connect():
    """Open a pooled connection for the duration of the operation."""
    path = os.path.abspath(_DB_PATH)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except (OSError, PermissionError):
        path = os.path.abspath("plagiarism_detector.db")

    pool = _pool()
    conn = pool.get(path)

    if conn is not None:
        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            conn = None

    if conn is None:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode=WAL")
        pool[path] = conn
        with _pool_lock:
            _all_connections.add(conn)

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def close_connections(all_threads: bool = False) -> None:
    """Close pooled connections."""
    if all_threads:
        _cleanup_all_connections()
        pool = getattr(_connection_pool, "connections", {})
        pool.clear()
    else:
        pool = getattr(_connection_pool, "connections", {})
        for conn in pool.values():
            try:
                conn.close()
            except Exception:
                pass
        pool.clear()


def configure_db_path(db_path: str | os.PathLike) -> None:
    """Override the database file path (useful for testing)."""
    global _DB_PATH
    close_connections()
    _DB_PATH = os.path.abspath(os.fspath(db_path))


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_health_score_db() -> None:
    """Create the document health score tables if they do not exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_health_scores (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                filename        TEXT    NOT NULL,
                overall_score   REAL    NOT NULL,
                grade           TEXT    NOT NULL,
                dimension_data  TEXT    NOT NULL,
                metadata_json   TEXT,
                checked_at      TEXT    NOT NULL,
                gate_passed     INTEGER NOT NULL DEFAULT 1,
                gate_reason     TEXT,
                FOREIGN KEY (filename)
                    REFERENCES documents(filename)
                    ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quality_gate_config (
                config_key   TEXT PRIMARY KEY,
                config_value TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quality_trend_snapshots (
                snapshot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT    NOT NULL,
                total_docs    INTEGER NOT NULL DEFAULT 0,
                avg_score     REAL    NOT NULL DEFAULT 0.0,
                median_score  REAL    NOT NULL DEFAULT 0.0,
                pass_rate     REAL    NOT NULL DEFAULT 0.0,
                grade_dist    TEXT    NOT NULL DEFAULT '{}',
                dimension_avgs TEXT   NOT NULL DEFAULT '{}',
                created_at    TEXT    NOT NULL
            )
            """
        )

        # Indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hs_filename ON document_health_scores(filename)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hs_score ON document_health_scores(overall_score)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hs_checked ON document_health_scores(checked_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hs_gate ON document_health_scores(gate_passed)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_qt_date ON quality_trend_snapshots(snapshot_date)"
        )

        # Seed default quality-gate config
        now = datetime.now().isoformat()
        defaults = {
            "min_score": "60.0",
            "min_grade": "D",
            "enabled": "true",
        }
        for key, value in defaults.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO quality_gate_config (config_key, config_value, updated_at)
                VALUES (?, ?, ?)
                """,
                (key, value, now),
            )


# ---------------------------------------------------------------------------
# Repository class
# ---------------------------------------------------------------------------

class HealthScoreRepository(BaseRepository):
    """Data access object for document health score tables."""

    def __init__(self, db_path: str | os.PathLike = _DB_PATH) -> None:
        super().__init__(db_path)

    # -- Score persistence ---------------------------------------------------

    def save_score(
        self,
        filename: str,
        overall_score: float,
        grade: str,
        dimensions: list[dict],
        metadata: dict | None = None,
        gate_passed: bool = True,
        gate_reason: str = "",
    ) -> int:
        """Persist a health score snapshot and return the record ID."""
        now = datetime.now().isoformat()
        dim_json = json.dumps(dimensions)
        meta_json = json.dumps(metadata) if metadata else None
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO document_health_scores
                    (filename, overall_score, grade, dimension_data, metadata_json,
                     checked_at, gate_passed, gate_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    overall_score,
                    grade,
                    dim_json,
                    meta_json,
                    now,
                    1 if gate_passed else 0,
                    gate_reason,
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_latest_score(self, filename: str) -> dict[str, Any] | None:
        """Retrieve the most recent health score for a document."""
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM document_health_scores
                WHERE filename = ?
                ORDER BY checked_at DESC LIMIT 1
                """,
                (filename,),
            ).fetchone()
            return self._hydrate_row(row) if row else None

    def get_score_history(
        self, filename: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Retrieve historical health scores for a document."""
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM document_health_scores
                WHERE filename = ?
                ORDER BY checked_at DESC LIMIT ?
                """,
                (filename, limit),
            ).fetchall()
            return [self._hydrate_row(r) for r in rows]

    def list_scores(
        self,
        *,
        min_score: float | None = None,
        max_score: float | None = None,
        grade: str | None = None,
        gate_passed: bool | None = None,
        sort_by: str = "overall_score",
        sort_order: str = "DESC",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List health scores with filtering and pagination."""
        allowed_sorts = {"overall_score", "checked_at", "grade", "filename"}
        if sort_by not in allowed_sorts:
            sort_by = "overall_score"
        sort_dir = "DESC" if sort_order.upper() == "DESC" else "ASC"

        query = "SELECT * FROM document_health_scores WHERE 1=1"
        params: list[Any] = []

        if min_score is not None:
            query += " AND overall_score >= ?"
            params.append(min_score)
        if max_score is not None:
            query += " AND overall_score <= ?"
            params.append(max_score)
        if grade:
            query += " AND grade = ?"
            params.append(grade)
        if gate_passed is not None:
            query += " AND gate_passed = ?"
            params.append(1 if gate_passed else 0)

        query += f" ORDER BY {sort_by} {sort_dir} LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [self._hydrate_row(r) for r in rows]

    def count_scores(
        self,
        *,
        min_score: float | None = None,
        gate_passed: bool | None = None,
    ) -> int:
        """Count health score records matching optional filters."""
        query = "SELECT COUNT(1) FROM document_health_scores WHERE 1=1"
        params: list[Any] = []
        if min_score is not None:
            query += " AND overall_score >= ?"
            params.append(min_score)
        if gate_passed is not None:
            query += " AND gate_passed = ?"
            params.append(1 if gate_passed else 0)

        with _connect() as conn:
            row = conn.execute(query, params).fetchone()
            return int(row[0]) if row else 0

    def delete_score(self, score_id: int) -> bool:
        """Delete a specific health score record."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM document_health_scores WHERE id = ?", (score_id,)
            )
            return cursor.rowcount > 0

    def delete_scores_for_document(self, filename: str) -> int:
        """Delete all health scores for a document."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM document_health_scores WHERE filename = ?",
                (filename,),
            )
            return cursor.rowcount

    # -- Quality gate config -------------------------------------------------

    def get_gate_config(self) -> dict[str, str]:
        """Retrieve the current quality gate configuration."""
        with _connect() as conn:
            rows = conn.execute(
                "SELECT config_key, config_value FROM quality_gate_config"
            ).fetchall()
            return {row[0]: row[1] for row in rows}

    def set_gate_config(self, key: str, value: str) -> None:
        """Update a quality gate configuration value."""
        now = datetime.now().isoformat()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO quality_gate_config (config_key, config_value, updated_at)
                VALUES (?, ?, ?)
                """,
                (key, value, now),
            )

    # -- Aggregate analytics -------------------------------------------------

    def get_score_summary(self) -> dict[str, Any]:
        """Compute aggregate statistics across all health scores."""
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(1)                           AS total,
                    AVG(overall_score)                 AS avg_score,
                    MIN(overall_score)                 AS min_score,
                    MAX(overall_score)                 AS max_score,
                    SUM(CASE WHEN gate_passed = 1 THEN 1 ELSE 0 END) AS passed,
                    SUM(CASE WHEN gate_passed = 0 THEN 1 ELSE 0 END) AS failed
                FROM document_health_scores
                """
            ).fetchone()

            total = int(row[0]) if row[0] else 0
            avg = float(row[1]) if row[1] else 0.0
            min_s = float(row[2]) if row[2] else 0.0
            max_s = float(row[3]) if row[3] else 0.0
            passed = int(row[4]) if row[4] else 0
            failed = int(row[5]) if row[5] else 0

            # Grade distribution
            grade_rows = conn.execute(
                """
                SELECT grade, COUNT(1) AS cnt
                FROM document_health_scores
                GROUP BY grade
                ORDER BY cnt DESC
                """
            ).fetchall()
            grade_dist = {r[0]: r[1] for r in grade_rows}

            # Latest check timestamp
            latest = conn.execute(
                "SELECT MAX(checked_at) FROM document_health_scores"
            ).fetchone()
            last_checked = latest[0] if latest and latest[0] else None

        return {
            "total_scored": total,
            "avg_score": round(avg, 2),
            "min_score": round(min_s, 2),
            "max_score": round(max_s, 2),
            "passed_gate": passed,
            "failed_gate": failed,
            "pass_rate": round(passed / total * 100, 1) if total > 0 else 0.0,
            "grade_distribution": grade_dist,
            "last_checked_at": last_checked,
        }

    def get_dimension_averages(self) -> dict[str, float]:
        """Compute average score per dimension across all documents."""
        with _connect() as conn:
            rows = conn.execute(
                "SELECT dimension_data FROM document_health_scores"
            ).fetchall()

        dim_totals: dict[str, float] = {}
        dim_counts: dict[str, int] = {}

        for (dim_json,) in rows:
            try:
                dims = json.loads(dim_json)
                for d in dims:
                    name = d.get("name", "unknown")
                    score = d.get("score", 0)
                    dim_totals[name] = dim_totals.get(name, 0) + score
                    dim_counts[name] = dim_counts.get(name, 0) + 1
            except (json.JSONDecodeError, TypeError):
                continue

        return {
            name: round(dim_totals[name] / dim_counts[name], 2)
            for name in dim_totals
            if dim_counts[name] > 0
        }

    def get_worst_documents(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the documents with the lowest health scores."""
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM document_health_scores
                WHERE id IN (
                    SELECT MAX(id) FROM document_health_scores
                    GROUP BY filename
                )
                ORDER BY overall_score ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [self._hydrate_row(r) for r in rows]

    def get_best_documents(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the documents with the highest health scores."""
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM document_health_scores
                WHERE id IN (
                    SELECT MAX(id) FROM document_health_scores
                    GROUP BY filename
                )
                ORDER BY overall_score DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [self._hydrate_row(r) for r in rows]

    # -- Trend snapshots -----------------------------------------------------

    def save_trend_snapshot(self, snapshot_data: dict) -> int:
        """Persist a daily trend snapshot."""
        now = datetime.now().isoformat()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO quality_trend_snapshots
                    (snapshot_date, total_docs, avg_score, median_score,
                     pass_rate, grade_dist, dimension_avgs, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_data.get("snapshot_date", now[:10]),
                    snapshot_data.get("total_docs", 0),
                    snapshot_data.get("avg_score", 0.0),
                    snapshot_data.get("median_score", 0.0),
                    snapshot_data.get("pass_rate", 0.0),
                    json.dumps(snapshot_data.get("grade_distribution", {})),
                    json.dumps(snapshot_data.get("dimension_averages", {})),
                    now,
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_trend_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Retrieve trend snapshots for the last N days."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM quality_trend_snapshots
                WHERE created_at >= ?
                ORDER BY snapshot_date DESC
                """,
                (since,),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                for key in ("grade_dist", "dimension_avgs"):
                    if d.get(key):
                        try:
                            d[key] = json.loads(d[key])
                        except (json.JSONDecodeError, TypeError):
                            pass
                results.append(d)
            return results

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _hydrate_row(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a Row to a dict, parsing JSON fields."""
        d = dict(row)
        for key in ("dimension_data", "metadata_json"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

health_repo = HealthScoreRepository(_DB_PATH)
