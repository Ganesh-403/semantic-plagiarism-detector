"""
audit_log.py
------------
Persistent audit trail for all significant system actions.
Records document uploads, deletions, scan runs, incident reviews,
threshold changes, and user management events in a dedicated SQLite
table with structured metadata and full-text search support.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.app_config import CORPUS_DB_PATH
from src.db.base import BaseRepository

logger = logging.getLogger(__name__)

_DB_PATH = os.path.abspath(str(CORPUS_DB_PATH))
_connection_pool = threading.local()
_pool_lock = threading.Lock()


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuditEntry:
    """A single immutable audit log record."""
    entry_id: str
    timestamp: str
    action: str
    actor: str
    target: str
    detail: str
    metadata_json: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["metadata"] = json.loads(self.metadata_json) if self.metadata_json else {}
        return d


@dataclass(frozen=True)
class AuditFilter:
    """Filter criteria for querying the audit log."""
    action: Optional[str] = None
    actor: Optional[str] = None
    target: Optional[str] = None
    search_text: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = 100
    offset: int = 0


# ── Action type constants ─────────────────────────────────────────────────────

ACTION_DOCUMENT_UPLOAD = "document.upload"
ACTION_DOCUMENT_DELETE = "document.delete"
ACTION_DOCUMENT_RESTORE = "document.restore"
ACTION_SCAN_RUN = "scan.run"
ACTION_INCIDENT_REVIEW = "incident.review"
ACTION_INCIDENT_RESOLVE = "incident.resolve"
ACTION_INCIDENT_DISMISS = "incident.dismiss_fp"
ACTION_THRESHOLD_CHANGE = "threshold.change"
ACTION_USER_CREATE = "user.create"
ACTION_USER_DELETE = "user.delete"
ACTION_USER_LOGIN = "user.login"
ACTION_EXPORT_REPORT = "export.report"

_ALL_ACTIONS = (
    ACTION_DOCUMENT_UPLOAD, ACTION_DOCUMENT_DELETE, ACTION_DOCUMENT_RESTORE,
    ACTION_SCAN_RUN, ACTION_INCIDENT_REVIEW, ACTION_INCIDENT_RESOLVE,
    ACTION_INCIDENT_DISMISS, ACTION_THRESHOLD_CHANGE,
    ACTION_USER_CREATE, ACTION_USER_DELETE, ACTION_USER_LOGIN,
    ACTION_EXPORT_REPORT,
)


# ── Connection management ─────────────────────────────────────────────────────


def _pool() -> Dict[str, sqlite3.Connection]:
    pool = getattr(_connection_pool, "connections", None)
    if pool is None:
        pool = {}
        _connection_pool.connections = pool
    return pool


@contextmanager
def _connect():
    """Open a pooled connection to the audit log database."""
    path = os.path.abspath(_DB_PATH)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except (OSError, PermissionError):
        pass

    pool = _pool()
    conn = pool.get(path)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            conn = None

    if conn is None:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        pool[path] = conn

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def configure_db_path(db_path: str | os.PathLike) -> None:
    """Override the audit log database path (mainly for testing)."""
    global _DB_PATH
    _DB_PATH = os.path.abspath(str(db_path))


def close_connections() -> None:
    """Close all pooled connections."""
    pool = getattr(_connection_pool, "connections", {})
    for conn in pool.values():
        try:
            conn.close()
        except Exception:
            pass
    pool.clear()


# ── Schema initialisation ─────────────────────────────────────────────────────


def init_audit_db() -> None:
    """Create the audit_log table and FTS index if they don't exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                entry_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                target TEXT NOT NULL DEFAULT '',
                detail TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)
        """)
        # FTS5 for full-text search across detail and target
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS audit_log_fts USING fts5(
                action, actor, target, detail,
                content='audit_log',
                content_rowid='rowid'
            )
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS audit_log_ai AFTER INSERT ON audit_log BEGIN
                INSERT INTO audit_log_fts(rowid, action, actor, target, detail)
                VALUES (new.rowid, new.action, new.actor, new.target, new.detail);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS audit_log_ad AFTER DELETE ON audit_log BEGIN
                INSERT INTO audit_log_fts(audit_log_fts, rowid, action, actor, target, detail)
                VALUES ('delete', old.rowid, old.action, old.actor, old.target, old.detail);
            END
        """)


# ── Write operations ──────────────────────────────────────────────────────────


def record_event(
    action: str,
    actor: str,
    target: str = "",
    detail: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Record a single audit event and return its entry_id.

    Args:
        action: One of the ACTION_* constants.
        actor: Username or system identity performing the action.
        target: Filename, username, or entity acted upon.
        detail: Human-readable description.
        metadata: Optional structured payload stored as JSON.

    Returns:
        The generated UUID entry_id.
    """
    entry_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    meta_json = json.dumps(metadata or {}, default=str)

    with _connect() as conn:
        conn.execute(
            "INSERT INTO audit_log (entry_id, timestamp, action, actor, target, detail, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (entry_id, now, action, actor, target, detail, meta_json),
        )

    logger.debug("Audit event recorded: %s by %s on %s", action, actor, target)
    return entry_id


def record_document_upload(filename: str, actor: str, **extra: Any) -> str:
    """Shortcut for recording a document upload event."""
    return record_event(
        ACTION_DOCUMENT_UPLOAD, actor, target=filename,
        detail=f"Document '{filename}' uploaded and indexed.",
        metadata=extra,
    )


def record_document_delete(filename: str, actor: str, soft: bool = True, **extra: Any) -> str:
    """Shortcut for recording a document deletion event."""
    kind = "soft-deleted" if soft else "permanently deleted"
    return record_event(
        ACTION_DOCUMENT_DELETE, actor, target=filename,
        detail=f"Document '{filename}' {kind}.",
        metadata={"soft_delete": soft, **extra},
    )


def record_document_restore(filename: str, actor: str) -> str:
    """Shortcut for recording a document restore event."""
    return record_event(
        ACTION_DOCUMENT_RESTORE, actor, target=filename,
        detail=f"Document '{filename}' restored from trash.",
    )


def record_scan_run(
    actor: str,
    document_count: int,
    flagged_count: int,
    threshold: float,
    **extra: Any,
) -> str:
    """Shortcut for recording a scan execution event."""
    return record_event(
        ACTION_SCAN_RUN, actor,
        detail=f"Scan completed: {document_count} docs, {flagged_count} flagged at threshold {threshold}.",
        metadata={"document_count": document_count, "flagged_count": flagged_count,
                  "threshold": threshold, **extra},
    )


def record_incident_review(
    incident_id: str,
    actor: str,
    new_status: str,
    doc_a: str = "",
    doc_b: str = "",
) -> str:
    """Shortcut for recording an incident status change."""
    return record_event(
        ACTION_INCIDENT_REVIEW, actor, target=incident_id,
        detail=f"Incident '{incident_id}' status changed to '{new_status}'.",
        metadata={"new_status": new_status, "document_a": doc_a, "document_b": doc_b},
    )


def record_threshold_change(
    actor: str,
    old_threshold: float,
    new_threshold: float,
) -> str:
    """Shortcut for recording a threshold change."""
    return record_event(
        ACTION_THRESHOLD_CHANGE, actor,
        detail=f"Threshold changed from {old_threshold} to {new_threshold}.",
        metadata={"old_threshold": old_threshold, "new_threshold": new_threshold},
    )


def record_user_action(
    action: str,
    actor: str,
    target_user: str,
    **extra: Any,
) -> str:
    """Shortcut for user management events (create, delete, login)."""
    return record_event(
        action, actor, target=target_user,
        detail=f"User management: {action} on '{target_user}'.",
        metadata=extra,
    )


def record_export(actor: str, format: str, scope: str = "", **extra: Any) -> str:
    """Shortcut for recording an export event."""
    return record_event(
        ACTION_EXPORT_REPORT, actor, target=scope,
        detail=f"Report exported as {format}.",
        metadata={"format": format, "scope": scope, **extra},
    )


# ── Read operations ───────────────────────────────────────────────────────────


def _row_to_entry(row: tuple) -> AuditEntry:
    """Convert a database row to an AuditEntry."""
    return AuditEntry(
        entry_id=row[0],
        timestamp=row[1],
        action=row[2],
        actor=row[3],
        target=row[4],
        detail=row[5],
        metadata_json=row[6],
    )


def get_entry(entry_id: str) -> Optional[AuditEntry]:
    """Retrieve a single audit entry by ID."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT entry_id, timestamp, action, actor, target, detail, metadata_json "
            "FROM audit_log WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()
    return _row_to_entry(row) if row else None


def query_entries(filters: Optional[AuditFilter] = None) -> List[AuditEntry]:
    """Query audit entries with optional filters.

    Supports filtering by action, actor, target, date range, and
    free-text search (via FTS5). Results are ordered by timestamp desc.
    """
    if filters is None:
        filters = AuditFilter()

    conditions = []
    params: list = []

    if filters.action:
        conditions.append("action = ?")
        params.append(filters.action)
    if filters.actor:
        conditions.append("actor = ?")
        params.append(filters.actor)
    if filters.target:
        conditions.append("target LIKE ?")
        params.append(f"%{filters.target}%")
    if filters.start_date:
        conditions.append("timestamp >= ?")
        params.append(f"{filters.start_date}T00:00:00")
    if filters.end_date:
        conditions.append("timestamp <= ?")
        params.append(f"{filters.end_date}T23:59:59")

    where = " AND ".join(conditions) if conditions else "1=1"

    if filters.search_text:
        # Use FTS5 for full-text search
        sanitized = filters.search_text.strip().replace('"', '""')
        fts_query = f'"{sanitized}"'
        try:
            with _connect() as conn:
                rows = conn.execute(
                    f"SELECT a.entry_id, a.timestamp, a.action, a.actor, a.target, a.detail, a.metadata_json "
                    f"FROM audit_log a "
                    f"JOIN audit_log_fts fts ON a.rowid = fts.rowid "
                    f"WHERE audit_log_fts MATCH ? AND {where} "
                    f"ORDER BY a.timestamp DESC LIMIT ? OFFSET ?",
                    (fts_query, *params, filters.limit, filters.offset),
                ).fetchall()
                return [_row_to_entry(r) for r in rows]
        except sqlite3.OperationalError:
            # Fallback to LIKE search if FTS fails
            where += " AND (detail LIKE ? OR target LIKE ?)"
            params.extend([f"%{filters.search_text}%", f"%{filters.search_text}%"])

    with _connect() as conn:
        rows = conn.execute(
            f"SELECT entry_id, timestamp, action, actor, target, detail, metadata_json "
            f"FROM audit_log WHERE {where} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (*params, filters.limit, filters.offset),
        ).fetchall()
    return [_row_to_entry(r) for r in rows]


def count_entries(filters: Optional[AuditFilter] = None) -> int:
    """Count audit entries matching the given filters."""
    if filters is None:
        filters = AuditFilter()

    conditions = []
    params: list = []

    if filters.action:
        conditions.append("action = ?")
        params.append(filters.action)
    if filters.actor:
        conditions.append("actor = ?")
        params.append(filters.actor)
    if filters.start_date:
        conditions.append("timestamp >= ?")
        params.append(f"{filters.start_date}T00:00:00")
    if filters.end_date:
        conditions.append("timestamp <= ?")
        params.append(f"{filters.end_date}T23:59:59")

    where = " AND ".join(conditions) if conditions else "1=1"

    with _connect() as conn:
        row = conn.execute(
            f"SELECT COUNT(1) FROM audit_log WHERE {where}", params
        ).fetchone()
    return int(row[0]) if row else 0


def get_action_counts() -> Dict[str, int]:
    """Return counts grouped by action type."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT action, COUNT(1) as cnt FROM audit_log GROUP BY action ORDER BY cnt DESC"
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def get_actor_counts(limit: int = 10) -> List[Dict[str, Any]]:
    """Return top actors by event count."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT actor, COUNT(1) as cnt FROM audit_log "
            "GROUP BY actor ORDER BY cnt DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"actor": r[0], "count": r[1]} for r in rows]


def get_recent_activity(limit: int = 20) -> List[AuditEntry]:
    """Return the most recent audit entries."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT entry_id, timestamp, action, actor, target, detail, metadata_json "
            "FROM audit_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_entry(r) for r in rows]


# ── Export ────────────────────────────────────────────────────────────────────


def export_entries_csv(entries: List[AuditEntry]) -> str:
    """Export a list of audit entries to CSV string."""
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["entry_id", "timestamp", "action", "actor", "target", "detail"])
    for e in entries:
        w.writerow([e.entry_id, e.timestamp, e.action, e.actor, e.target, e.detail])
    return buf.getvalue()


def export_entries_json(entries: List[AuditEntry], indent: int = 2) -> str:
    """Export a list of audit entries to JSON string."""
    return json.dumps(
        [e.to_dict() for e in entries],
        indent=indent,
        default=str,
    )


def purge_old_entries(days: int = 90) -> int:
    """Delete audit entries older than the specified number of days.

    Args:
        days: Retention period in days.

    Returns:
        Number of deleted entries.
    """
    from datetime import timedelta

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM audit_log WHERE timestamp < ?", (cutoff,)
        )
    deleted = cursor.rowcount
    if deleted > 0:
        logger.info("Purged %d audit entries older than %d days.", deleted, days)
    return deleted
