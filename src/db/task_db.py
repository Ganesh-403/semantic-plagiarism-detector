"""
src/db/task_db.py
-----------------
SQLite-backed job state manager for the distributed task queue (Issue #3146).

Manages job lifecycle states (PENDING → PROCESSING → COMPLETED / FAILED)
and their JSON payloads. Thread-safe via SQLite WAL mode + a per-thread
connection local. The schema is created automatically on first use so the
queue works out-of-the-box without a separate migration step.
"""

from __future__ import annotations

import atexit
from enum import Enum
import json
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from src.core.app_config import DATA_DIR

logger = logging.getLogger(__name__)
# ── Constants ────────────────────────────────────────────────────

VALID_STATUSES = ("PENDING", "PROCESSING", "COMPLETED", "FAILED", "DEAD_LETTER")

DEFAULT_DB_PATH = Path(os.environ.get(
    "TASK_QUEUE_DB_PATH",
    str(DATA_DIR / "task_queue.db"),
))
_connection_pool = threading.local()
_pool_lock = threading.Lock()
_all_connections: set[sqlite3.Connection] = set()

# ── Schema ──────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS task_jobs (
    id              TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'PENDING',
    payload         TEXT NOT NULL,             -- JSON blob
    result          TEXT,                     -- JSON blob (set on COMPLETED)
    error           TEXT,                     -- error message (set on FAILED)
    retry_count     INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 3,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT,
    worker_id       TEXT
);

CREATE INDEX IF NOT EXISTS idx_task_jobs_status
    ON task_jobs (status, created_at);
"""

# ── Connection management ──────────────────────────────────────

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_db_path(db_path: Optional[Union[Path, str]] = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    return Path(os.environ.get(
        "TASK_QUEUE_DB_PATH",
        str(DATA_DIR / "task_queue.db"),
    ))

def _get_connection(db_path: Optional[Union[Path, str]] = None) -> sqlite3.Connection:
    """Return a thread-local connection, creating one if needed."""
    resolved_path = _resolve_db_path(db_path)
    conn_key = f"conn_{resolved_path.resolve()}"
    conn = getattr(_connection_pool, conn_key, None)
    if conn is not None:
        return conn

    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(resolved_path),
        check_same_thread=False,
        timeout=30.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)
    conn.commit()

    with _pool_lock:
        _all_connections.add(conn)
    setattr(_connection_pool, conn_key, conn)
    return conn


def _cleanup_all_connections() -> None:
    with _pool_lock:
        for conn in _all_connections:
            try:
                conn.close()
            except Exception:
                pass
        _all_connections.clear()


atexit.register(_cleanup_all_connections)


@contextmanager
def get_conn(db_path: Optional[Union[Path, str]] = None) -> Generator[sqlite3.Connection, None, None]:
    """Yield the thread-local connection (no auto-commit)."""
    conn = _get_connection(db_path)
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise


class JsonPayload(str):
    """A string subclass that also behaves like a dict when indexed by key."""

    def __new__(cls, val: Any):
        if isinstance(val, (dict, list)):
            s = json.dumps(val, ensure_ascii=False)
            obj = super().__new__(cls, s)
            obj._data = val
        else:
            s = str(val or "")
            obj = super().__new__(cls, s)
            try:
                obj._data = json.loads(s) if s else {}
            except Exception:
                obj._data = {}
        return obj

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str) and isinstance(self._data, dict):
            return self._data[item]
        if isinstance(item, int) and isinstance(self._data, list):
            return self._data[item]
        return super().__getitem__(item)

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(self._data, dict):
            return self._data.get(key, default)
        return default

    def keys(self) -> Any:
        if isinstance(self._data, dict):
            return self._data.keys()
        return [].keys()

    def values(self) -> Any:
        if isinstance(self._data, dict):
            return self._data.values()
        return [].values()

    def items(self) -> Any:
        if isinstance(self._data, dict):
            return self._data.items()
        return [].items()

    def __contains__(self, item: Any) -> bool:
        if isinstance(self._data, (dict, list)):
            return item in self._data
        return super().__contains__(item)


class JobRecord(dict):
    """Dict representing a job row with property/str helpers."""

    @property
    def id(self) -> str:
        return str(self.get("id", ""))

    def __str__(self) -> str:
        return self.id


def _row_to_dict(row: sqlite3.Row) -> JobRecord:
    d = JobRecord(row)
    for key in ("payload", "result"):
        v = d.get(key)
        if v is not None:
            d[key] = JsonPayload(v)
    d["attempts"] = d.get("retry_count", 0)
    d["max_attempts"] = d.get("max_retries", 3)
    d["error_message"] = d.get("error")
    return d


def create_job(
    payload: Union[str, dict[str, Any]],
    *,
    max_retries: int = 3,
    max_attempts: Optional[int] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> JobRecord:
    """Insert a new PENDING job and return its row as a dict."""
    if max_attempts is not None:
        max_retries = max_attempts
    job_id = str(uuid.uuid4())
    now = _utcnow_iso()
    if isinstance(payload, str):
        payload_json = payload
    else:
        payload_json = json.dumps(payload, ensure_ascii=False)

    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO task_jobs
               (id, status, payload, max_retries, created_at, updated_at)
               VALUES (?, 'PENDING', ?, ?, ?, ?)""",
            (job_id, payload_json, max_retries, now, now),
        )
        conn.commit()

    return get_job(job_id, db_path=db_path)  # type: ignore[return-value]


def get_job(
    job_id: Union[str, dict[str, Any]],
    *,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[JobRecord]:
    """Return one job row as a dict, or None."""
    actual_id = job_id.get("id") if isinstance(job_id, dict) else str(job_id)
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM task_jobs WHERE id = ?", (actual_id,)
        ).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def list_jobs(
    *,
    status: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Union[Path, str]] = None,
) -> list[dict[str, Any]]:
    """List jobs, optionally filtered by status, newest first."""
    with get_conn(db_path) as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM task_jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM task_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [_row_to_dict(r) for r in rows]


def claim_next_job(
    worker_id: str = "worker",
    *,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[dict[str, Any]]:
    """Atomically claim the oldest PENDING job for a worker."""
    now = _utcnow_iso()
    with get_conn(db_path) as conn:
        try:
            row = conn.execute(
                """UPDATE task_jobs
                   SET status = 'PROCESSING',
                       updated_at = ?,
                       started_at = ?,
                       worker_id = ?
                   WHERE id = (
                       SELECT id FROM task_jobs
                       WHERE status = 'PENDING'
                       ORDER BY created_at ASC
                       LIMIT 1
                   )
                   RETURNING *""",
                (now, now, worker_id),
            ).fetchone()
            if row is not None:
                conn.commit()
                return _row_to_dict(row)
            conn.commit()
        except sqlite3.OperationalError:
            conn.execute("BEGIN IMMEDIATE")
            target = conn.execute(
                """SELECT id FROM task_jobs
                   WHERE status = 'PENDING'
                   ORDER BY created_at ASC
                   LIMIT 1"""
            ).fetchone()
            if target is None:
                conn.commit()
                return None
            job_id = target["id"]
            conn.execute(
                """UPDATE task_jobs
                   SET status = 'PROCESSING',
                       updated_at = ?,
                       started_at = ?,
                       worker_id = ?
                 WHERE id = ?""",
                (now, now, worker_id, job_id),
            )
            conn.commit()
            return get_job(job_id, db_path=db_path)
    return None


def mark_completed(
    job_id: str,
    result: dict[str, Any],
    *,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Mark a job as COMPLETED and store its result dict."""
    now = _utcnow_iso()
    result_json = json.dumps(result, ensure_ascii=False)
    with get_conn(db_path) as conn:
        conn.execute(
            """UPDATE task_jobs
               SET status = 'COMPLETED',
                   result = ?,
                   updated_at = ?,
                   completed_at = ?
             WHERE id = ?""",
            (result_json, now, now, job_id),
        )
        conn.commit()
    return True


def mark_failed(
    job_id: str,
    error: str,
    *,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Mark a job as FAILED. If retries remain, re-queue it as PENDING."""
    now = _utcnow_iso()
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT retry_count, max_retries FROM task_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return False

        new_retry_count = row["retry_count"] + 1
        if new_retry_count < row["max_retries"]:
            conn.execute(
                """UPDATE task_jobs
                   SET status = 'PENDING',
                       retry_count = ?,
                       error = ?,
                       updated_at = ?,
                       worker_id = NULL
                 WHERE id = ?""",
                (new_retry_count, error, now, job_id),
            )
        else:
            conn.execute(
                """UPDATE task_jobs
                   SET status = 'DEAD_LETTER',
                       retry_count = ?,
                       error = ?,
                       updated_at = ?,
                       completed_at = ?
                 WHERE id = ?""",
                (new_retry_count, error, now, now, job_id),
            )
        conn.commit()
    return True


def mark_dead_letter(
    job_id: str,
    error: str,
    *,
    db_path: Optional[Union[Path, str]] = None,
) -> None:
    """Immediately move a job to DEAD_LETTER, bypassing retries."""
    now = _utcnow_iso()
    with get_conn(db_path) as conn:
        conn.execute(
            """UPDATE task_jobs
               SET status = 'DEAD_LETTER',
                   error = ?,
                   updated_at = ?,
                   completed_at = ?
             WHERE id = ?""",
            (error, now, now, job_id),
        )
        conn.commit()


def get_dead_letter_jobs(
    *,
    limit: int = 50,
    db_path: Optional[Union[Path, str]] = None,
) -> list[dict[str, Any]]:
    return list_jobs(status="DEAD_LETTER", limit=limit, db_path=db_path)


def reset_db(db_path: Optional[Union[Path, str]] = None) -> None:
    """Drop and recreate the task_jobs table (for tests)."""
    with get_conn(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS task_jobs")
        conn.executescript(_SCHEMA_SQL)
        conn.commit()


# ── Compatibility & Enums ──────────────────────────────────────

class JobStatus(str, Enum):
    """Enumeration of valid job states."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"


def initialize_task_db(db_path: Optional[Union[Path, str]] = None) -> None:
    """Initialize schema for the task queue database."""
    _get_connection(db_path)


def claim_job(
    db_path: Optional[Union[Path, str]] = None,
    worker_id: str = "worker",
) -> Optional[dict[str, Any]]:
    """Compatibility wrapper around claim_next_job."""
    return claim_next_job(worker_id=worker_id, db_path=db_path)


def complete_job(
    job_id: str,
    result: dict[str, Any],
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Compatibility wrapper around mark_completed."""
    return mark_completed(job_id, result, db_path=db_path)


def fail_job(
    job_id: str,
    error_message: str,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Compatibility wrapper around mark_failed."""
    return mark_failed(job_id, error_message, db_path=db_path)


