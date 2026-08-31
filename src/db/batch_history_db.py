"""
batch_history_db.py
-------------------
SQLite database manager for batch analysis history.

Tracks batch scan runs including:
- Batch run metadata (timing, documents scanned, thresholds)
- Per-document results within each batch
- Timeline events for the audit trail
- Aggregate statistics for trend analysis
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
# Connection pool (mirrors corpus_db pattern)
# ---------------------------------------------------------------------------
_connection_pool = threading.local()
_all_connections: set[sqlite3.Connection] = set()
_pool_lock = threading.Lock()

# Use the same DB as corpus for simplicity; in production this could be
# a dedicated file via src.core.app_config.
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

def init_batch_history_db() -> None:
    """Create the batch analysis history tables if they do not exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_runs (
                run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at      TEXT    NOT NULL,
                completed_at    TEXT,
                status          TEXT    NOT NULL DEFAULT 'running'
                                    CHECK (status IN ('running','completed','failed','cancelled')),
                trigger_source  TEXT    NOT NULL DEFAULT 'manual'
                                    CHECK (trigger_source IN ('manual','scheduled','api','webhook')),
                documents_scanned  INTEGER NOT NULL DEFAULT 0,
                documents_flagged  INTEGER NOT NULL DEFAULT 0,
                avg_similarity     REAL    NOT NULL DEFAULT 0.0,
                max_similarity     REAL    NOT NULL DEFAULT 0.0,
                threshold_used     REAL    NOT NULL DEFAULT 0.75,
                duration_ms        INTEGER,
                error_message      TEXT,
                created_by         TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_run_documents (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL,
                document_name   TEXT    NOT NULL,
                similarity_score REAL   NOT NULL DEFAULT 0.0,
                severity        TEXT    NOT NULL DEFAULT 'none'
                                    CHECK (severity IN ('high','medium','low','none')),
                flagged         INTEGER NOT NULL DEFAULT 0,
                matched_docs    TEXT,
                processing_ms   INTEGER,
                FOREIGN KEY (run_id) REFERENCES batch_runs(run_id)
                    ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_timeline_events (
                event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER,
                event_type  TEXT    NOT NULL
                                CHECK (event_type IN (
                                    'batch_started','batch_completed','batch_failed',
                                    'batch_cancelled','document_uploaded','document_scanned',
                                    'threshold_changed','system_maintenance','alert_triggered'
                                )),
                severity    TEXT    NOT NULL DEFAULT 'info'
                                CHECK (severity IN ('info','warning','error','success')),
                message     TEXT    NOT NULL,
                metadata    TEXT,
                created_at  TEXT    NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_alerts (
                alert_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER,
                alert_type  TEXT    NOT NULL
                                CHECK (alert_type IN ('high_plagiarism','threshold_exceeded',
                                                       'batch_failure','anomaly_detected')),
                title       TEXT    NOT NULL,
                message     TEXT    NOT NULL,
                is_read     INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT    NOT NULL,
                FOREIGN KEY (run_id) REFERENCES batch_runs(run_id)
                    ON DELETE SET NULL
            )
            """
        )

        # Indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_batch_runs_started ON batch_runs(started_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_batch_runs_status ON batch_runs(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_batch_run_docs_run ON batch_run_documents(run_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_batch_timeline_run ON batch_timeline_events(run_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_batch_timeline_type ON batch_timeline_events(event_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_batch_alerts_read ON batch_alerts(is_read)"
        )


# ---------------------------------------------------------------------------
# Repository class
# ---------------------------------------------------------------------------

class BatchHistoryRepository(BaseRepository):
    """Data access object for batch analysis history tables."""

    def __init__(self, db_path: str | os.PathLike = _DB_PATH) -> None:
        super().__init__(db_path)

    # -- Batch runs ---------------------------------------------------------

    def create_batch_run(
        self,
        trigger_source: str = "manual",
        threshold_used: float = 0.75,
        created_by: str | None = None,
    ) -> int:
        """Insert a new batch run and return its run_id."""
        now = datetime.now().isoformat()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO batch_runs
                    (started_at, status, trigger_source, threshold_used, created_by)
                VALUES (?, 'running', ?, ?, ?)
                """,
                (now, trigger_source, threshold_used, created_by),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def complete_batch_run(
        self,
        run_id: int,
        *,
        documents_scanned: int = 0,
        documents_flagged: int = 0,
        avg_similarity: float = 0.0,
        max_similarity: float = 0.0,
        duration_ms: int | None = None,
    ) -> None:
        """Mark a batch run as completed with summary statistics."""
        now = datetime.now().isoformat()
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE batch_runs
                SET status = 'completed',
                    completed_at = ?,
                    documents_scanned = ?,
                    documents_flagged = ?,
                    avg_similarity = ?,
                    max_similarity = ?,
                    duration_ms = ?
                WHERE run_id = ?
                """,
                (
                    now,
                    documents_scanned,
                    documents_flagged,
                    avg_similarity,
                    max_similarity,
                    duration_ms,
                    run_id,
                ),
            )

    def fail_batch_run(self, run_id: int, error_message: str) -> None:
        """Mark a batch run as failed."""
        now = datetime.now().isoformat()
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE batch_runs
                SET status = 'failed',
                    completed_at = ?,
                    error_message = ?
                WHERE run_id = ?
                """,
                (now, error_message, run_id),
            )

    def cancel_batch_run(self, run_id: int) -> None:
        """Mark a batch run as cancelled."""
        now = datetime.now().isoformat()
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE batch_runs
                SET status = 'cancelled',
                    completed_at = ?
                WHERE run_id = ?
                """,
                (now, run_id),
            )

    def get_batch_run(self, run_id: int) -> dict[str, Any] | None:
        """Retrieve a single batch run by ID."""
        conn_ctx = _connect()
        with conn_ctx as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM batch_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_batch_runs(
        self,
        *,
        status: str | None = None,
        trigger_source: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List batch runs with optional filtering and pagination."""
        query = "SELECT * FROM batch_runs WHERE 1=1"
        params: list[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if trigger_source:
            query += " AND trigger_source = ?"
            params.append(trigger_source)
        if start_date:
            query += " AND started_at >= ?"
            params.append(start_date)
        if end_date:
            query += " AND started_at <= ?"
            params.append(end_date)

        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def count_batch_runs(
        self,
        *,
        status: str | None = None,
        trigger_source: str | None = None,
    ) -> int:
        """Count total batch runs matching optional filters."""
        query = "SELECT COUNT(1) FROM batch_runs WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if trigger_source:
            query += " AND trigger_source = ?"
            params.append(trigger_source)

        with _connect() as conn:
            row = conn.execute(query, params).fetchone()
            return int(row[0]) if row else 0

    def delete_batch_run(self, run_id: int) -> bool:
        """Delete a batch run and cascade to child records."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM batch_runs WHERE run_id = ?", (run_id,)
            )
            return cursor.rowcount > 0

    # -- Batch run documents ------------------------------------------------

    def add_batch_document(
        self,
        run_id: int,
        document_name: str,
        similarity_score: float = 0.0,
        severity: str = "none",
        flagged: bool = False,
        matched_docs: list[str] | None = None,
        processing_ms: int | None = None,
    ) -> int:
        """Record a document result for a batch run."""
        matched_json = json.dumps(matched_docs) if matched_docs else None
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO batch_run_documents
                    (run_id, document_name, similarity_score, severity, flagged,
                     matched_docs, processing_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    document_name,
                    similarity_score,
                    severity,
                    1 if flagged else 0,
                    matched_json,
                    processing_ms,
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_batch_documents(self, run_id: int) -> list[dict[str, Any]]:
        """Retrieve all document results for a given batch run."""
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM batch_run_documents WHERE run_id = ? ORDER BY similarity_score DESC",
                (run_id,),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d.get("matched_docs"):
                    try:
                        d["matched_docs"] = json.loads(d["matched_docs"])
                    except (json.JSONDecodeError, TypeError):
                        d["matched_docs"] = []
                else:
                    d["matched_docs"] = []
                results.append(d)
            return results

    def get_document_results_count(self, run_id: int) -> int:
        """Count documents in a specific batch run."""
        with _connect() as conn:
            row = conn.execute(
                "SELECT COUNT(1) FROM batch_run_documents WHERE run_id = ?", (run_id,)
            ).fetchone()
            return int(row[0]) if row else 0

    # -- Timeline events ----------------------------------------------------

    def add_timeline_event(
        self,
        event_type: str,
        message: str,
        *,
        run_id: int | None = None,
        severity: str = "info",
        metadata: dict | None = None,
    ) -> int:
        """Insert a timeline event for the audit trail."""
        now = datetime.now().isoformat()
        meta_json = json.dumps(metadata) if metadata else None
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO batch_timeline_events
                    (run_id, event_type, severity, message, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, event_type, severity, message, meta_json, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_timeline_events(
        self,
        *,
        run_id: int | None = None,
        event_type: str | None = None,
        severity: str | None = None,
        start_date: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Retrieve timeline events with optional filters."""
        query = "SELECT * FROM batch_timeline_events WHERE 1=1"
        params: list[Any] = []

        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d.get("metadata"):
                    try:
                        d["metadata"] = json.loads(d["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                results.append(d)
            return results

    # -- Alerts -------------------------------------------------------------

    def create_alert(
        self,
        alert_type: str,
        title: str,
        message: str,
        run_id: int | None = None,
    ) -> int:
        """Create a new alert."""
        now = datetime.now().isoformat()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO batch_alerts (run_id, alert_type, title, message, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, alert_type, title, message, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_alerts(
        self,
        *,
        is_read: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Retrieve alerts with optional read-status filter."""
        query = "SELECT * FROM batch_alerts WHERE 1=1"
        params: list[Any] = []

        if is_read is not None:
            query += " AND is_read = ?"
            params.append(1 if is_read else 0)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def mark_alert_read(self, alert_id: int) -> bool:
        """Mark a single alert as read."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE batch_alerts SET is_read = 1 WHERE alert_id = ?",
                (alert_id,),
            )
            return cursor.rowcount > 0

    def mark_all_alerts_read(self) -> int:
        """Mark all unread alerts as read. Returns the count of updated rows."""
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE batch_alerts SET is_read = 1 WHERE is_read = 0"
            )
            return cursor.rowcount

    def get_unread_alert_count(self) -> int:
        """Count unread alerts."""
        with _connect() as conn:
            row = conn.execute(
                "SELECT COUNT(1) FROM batch_alerts WHERE is_read = 0"
            ).fetchone()
            return int(row[0]) if row else 0

    # -- Analytics ----------------------------------------------------------

    def get_trend_data(self, days: int = 30) -> list[dict[str, Any]]:
        """Return daily aggregated scan statistics for the last N days."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    DATE(started_at) AS scan_date,
                    COUNT(*)         AS total_runs,
                    SUM(documents_scanned) AS total_docs_scanned,
                    SUM(documents_flagged) AS total_docs_flagged,
                    AVG(avg_similarity)   AS avg_similarity,
                    MAX(max_similarity)   AS peak_similarity,
                    AVG(duration_ms)      AS avg_duration_ms
                FROM batch_runs
                WHERE started_at >= ?
                  AND status IN ('completed', 'failed')
                GROUP BY DATE(started_at)
                ORDER BY scan_date DESC
                """,
                (since,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_summary_stats(self) -> dict[str, Any]:
        """Return high-level summary statistics across all batch runs."""
        with _connect() as conn:
            total_runs = conn.execute(
                "SELECT COUNT(1) FROM batch_runs"
            ).fetchone()[0]

            completed_runs = conn.execute(
                "SELECT COUNT(1) FROM batch_runs WHERE status = 'completed'"
            ).fetchone()[0]

            failed_runs = conn.execute(
                "SELECT COUNT(1) FROM batch_runs WHERE status = 'failed'"
            ).fetchone()[0]

            total_docs_scanned = conn.execute(
                "SELECT COALESCE(SUM(documents_scanned), 0) FROM batch_runs"
            ).fetchone()[0]

            total_docs_flagged = conn.execute(
                "SELECT COALESCE(SUM(documents_flagged), 0) FROM batch_runs"
            ).fetchone()[0]

            avg_similarity_row = conn.execute(
                "SELECT AVG(avg_similarity) FROM batch_runs WHERE status = 'completed'"
            ).fetchone()
            avg_similarity = float(avg_similarity_row[0]) if avg_similarity_row[0] else 0.0

            avg_duration_row = conn.execute(
                "SELECT AVG(duration_ms) FROM batch_runs WHERE status = 'completed'"
            ).fetchone()
            avg_duration_ms = int(avg_duration_row[0]) if avg_duration_row[0] else 0

            last_run_row = conn.execute(
                "SELECT started_at FROM batch_runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            last_run_at = last_run_row[0] if last_run_row else None

            return {
                "total_runs": total_runs,
                "completed_runs": completed_runs,
                "failed_runs": failed_runs,
                "success_rate": round(completed_runs / total_runs * 100, 1) if total_runs > 0 else 0.0,
                "total_documents_scanned": total_docs_scanned,
                "total_documents_flagged": total_docs_flagged,
                "avg_similarity": round(avg_similarity, 4),
                "avg_duration_ms": avg_duration_ms,
                "last_run_at": last_run_at,
            }

    def get_severity_distribution(self, run_id: int | None = None) -> dict[str, int]:
        """Return the distribution of severity levels across document results."""
        query = "SELECT severity, COUNT(1) AS cnt FROM batch_run_documents"
        params: list[Any] = []
        if run_id is not None:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " GROUP BY severity"

        with _connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return {row[0]: row[1] for row in rows}

    def purge_old_runs(self, days: int = 90) -> int:
        """Delete batch runs older than the specified number of days."""
        threshold = (datetime.now() - timedelta(days=days)).isoformat()
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM batch_runs WHERE started_at < ?", (threshold,)
            )
            return cursor.rowcount


# ---------------------------------------------------------------------------
# Module-level convenience instances
# ---------------------------------------------------------------------------

batch_repo = BatchHistoryRepository(_DB_PATH)
