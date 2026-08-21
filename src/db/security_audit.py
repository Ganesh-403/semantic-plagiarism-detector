"""
src/db/security_audit.py
------------------------
Security audit logging, account lockout mechanism, and audit-event pagination.

Tracks security-related events, provides utilities to enforce account lockout
policies, and supports paginated retrieval of audit events.
"""

import json
import logging
import os
import sqlite3
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import metrics

    _meter = metrics.get_meter("semantic_plagiarism_detector.security_audit")
    _audit_failure_counter = _meter.create_counter(
        name="security_audit_log.failures",
        unit="1",
        description="Count of security audit log write failures",
    )
except Exception:
    _audit_failure_counter = None


def _emit_audit_log_failure_alert(
    event_type: str,
    username: Optional[str],
    error: Exception,
) -> None:
    """Emit an OpenTelemetry metric and/or fire an alert webhook when writing to security audit log fails (Issue #2729)."""
    if _audit_failure_counter is not None:
        try:
            _audit_failure_counter.add(
                1,
                {
                    "event_type": str(event_type),
                    "error_type": error.__class__.__name__,
                },
            )
        except Exception as otel_exc:
            logger.warning(
                "Failed to emit OpenTelemetry metric for security log failure: %s",
                otel_exc,
            )

    webhook_url = os.getenv("SECURITY_AUDIT_ALERT_WEBHOOK_URL") or os.getenv(
        "ALERT_WEBHOOK_URL"
    )
    if webhook_url:
        try:
            payload = json.dumps(
                {
                    "alert": "SecurityAuditLogWriteFailure",
                    "event_type": event_type,
                    "username": username,
                    "error": str(error),
                    "error_type": error.__class__.__name__,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as webhook_exc:
            logger.warning(
                "Failed to fire alert webhook for security log failure: %s",
                webhook_exc,
            )


def log_security_event(
    event_type: str,
    username: Optional[str] = None,
    details: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Log a security-related event to the audit log.

    Args:
        event_type: Type of event (e.g., 'login_failed', 'login_success').
        username: The username associated with the event.
        details: Additional context about the event.
        db_path: Optional path to the SQLite database. If None, uses default auth DB.
    """
    if db_path is None:
        from src.db.auth import get_auth_db_path

        db_path = str(get_auth_db_path())

    timestamp = datetime.utcnow().isoformat()

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    username TEXT,
                    details TEXT
                )
                """
            )

            conn.execute(
                """
                INSERT INTO security_audit_log (
                    timestamp, event_type, username, details
                )
                VALUES (?, ?, ?, ?)
                """,
                (timestamp, event_type, username, details),
            )
            conn.commit()

    except Exception as e:
        logger.error("Failed to log security event %s: %s", event_type, e)
        _emit_audit_log_failure_alert(event_type=event_type, username=username, error=e)


def count_recent_failed_logins(
    username: str,
    window_minutes: int = 15,
    db_path: Optional[str] = None,
) -> int:
    """Count failed login attempts for a user within a time window.

    Args:
        username: The username to check for failed attempts.
        window_minutes: Time window in minutes. Defaults to 15.
        db_path: Optional path to the SQLite database.

    Returns:
        Number of failed login attempts within the specified window.
        Returns 0 if the username is empty or an error occurs.
    """
    if not username or not isinstance(username, str):
        return 0

    if db_path is None:
        from src.db.auth import get_auth_db_path

        db_path = str(get_auth_db_path())

    cutoff_time = (datetime.utcnow() - timedelta(minutes=window_minutes)).isoformat()

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*)
                FROM security_audit_log
                WHERE event_type = 'login_failed'
                  AND username = ?
                  AND timestamp >= ?
                """,
                (username.strip().lower(), cutoff_time),
            )

            result = cursor.fetchone()
            return result[0] if result else 0

    except sqlite3.Error as e:
        logger.error(
            "Failed to count recent failed logins for %s: %s",
            username,
            e,
        )
        # Fail open: if we can't read the audit log, don't lock the user out.
        return 0


def get_recent_audit_events(
    limit: int = 20,
    offset: int = 0,
    event_type: str | None = None,
    username: str | None = None,
    db_path: str | None = None,
) -> list[dict]:
    """Fetch recent security audit events with pagination support.

    Retrieves audit log entries ordered by timestamp descending (most recent
    first). Supports filtering by event type and username, as well as
    pagination via limit and offset parameters.

    Args:
        limit: Maximum number of events to return. Defaults to 20.
        offset: Number of events to skip. Defaults to 0.
        event_type: Optional filter for a specific event type.
        username: Optional filter for a specific username.
        db_path: Optional path to the SQLite database.

    Returns:
        List of dictionaries representing audit events.
    """
    if limit < 1:
        limit = 20

    if offset < 0:
        offset = 0

    if db_path is None:
        from src.db.auth import get_auth_db_path

        db_path = str(get_auth_db_path())

    query = """
        SELECT id, timestamp, event_type, username, details
        FROM security_audit_log
        WHERE 1=1
    """
    params: list = []

    if event_type:
        query += " AND event_type = ?"
        params.append(event_type)

    if username:
        query += " AND username = ?"
        params.append(username.strip().lower())

    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    except sqlite3.Error as e:
        logger.error("Failed to fetch audit events: %s", e)
        return []


def get_audit_events_count(
    event_type: str | None = None,
    username: str | None = None,
    db_path: str | None = None,
) -> int:
    """Get the total count of audit events matching the filters.

    Used in conjunction with get_recent_audit_events to calculate total
    pages for paginated UIs.

    Args:
        event_type: Optional filter for a specific event type.
        username: Optional filter for a specific username.
        db_path: Optional path to the SQLite database.

    Returns:
        Total number of matching audit events.
    """
    if db_path is None:
        from src.db.auth import get_auth_db_path

        db_path = str(get_auth_db_path())

    query = "SELECT COUNT(*) FROM security_audit_log WHERE 1=1"
    params: list = []

    if event_type:
        query += " AND event_type = ?"
        params.append(event_type)

    if username:
        query += " AND username = ?"
        params.append(username.strip().lower())

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(query, tuple(params))
            result = cursor.fetchone()
            return result[0] if result else 0

    except sqlite3.Error as e:
        logger.error("Failed to count audit events: %s", e)
        return 0
