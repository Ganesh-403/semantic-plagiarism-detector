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
src/db/anomaly_alerts_db.py
---------------------------
SQLite persistence layer for anomaly detection alerts, scan history,
and alert management.

Provides:
  • Anomaly alert storage with severity, type, and confidence tracking
  • Scan history to record when anomaly detection was run
  • Alert acknowledgement and resolution workflow
  • Analytics queries for severity distribution and trends
  • Bulk operations and maintenance
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/anomaly_alerts.db")


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


@contextmanager
def get_connection(db_path: Path | None = None):
    """Context manager for SQLite connections."""
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


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


def init_anomaly_alerts_db(db_path: Path | None = None) -> None:
    """Create all tables for the anomaly alerts system."""
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS anomaly_scans (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_type       TEXT    NOT NULL DEFAULT 'full',
                status          TEXT    NOT NULL DEFAULT 'running',
                documents_scanned INTEGER NOT NULL DEFAULT 0,
                anomalies_found INTEGER NOT NULL DEFAULT 0,
                started_at      TEXT    NOT NULL,
                completed_at    TEXT,
                triggered_by    TEXT    NOT NULL DEFAULT 'system',
                config_snapshot TEXT    NOT NULL DEFAULT '{}',
                error_message   TEXT
            );

            CREATE TABLE IF NOT EXISTS anomaly_alerts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id         INTEGER,
                anomaly_type    TEXT    NOT NULL,
                severity        TEXT    NOT NULL DEFAULT 'info',
                title           TEXT    NOT NULL,
                description     TEXT    NOT NULL DEFAULT '',
                confidence      REAL    NOT NULL DEFAULT 0.0,
                affected_docs   TEXT    NOT NULL DEFAULT '[]',
                evidence        TEXT    NOT NULL DEFAULT '{}',
                is_acknowledged INTEGER NOT NULL DEFAULT 0,
                is_resolved     INTEGER NOT NULL DEFAULT 0,
                acknowledged_by TEXT,
                resolved_by     TEXT,
                notes           TEXT    NOT NULL DEFAULT '',
                detected_at     TEXT    NOT NULL,
                acknowledged_at TEXT,
                resolved_at     TEXT,
                FOREIGN KEY (scan_id) REFERENCES anomaly_scans(id)
            );

            CREATE INDEX IF NOT EXISTS idx_alerts_type
                ON anomaly_alerts(anomaly_type);

            CREATE INDEX IF NOT EXISTS idx_alerts_severity
                ON anomaly_alerts(severity);

            CREATE INDEX IF NOT EXISTS idx_alerts_scan
                ON anomaly_alerts(scan_id);

            CREATE TABLE IF NOT EXISTS anomaly_config (
                id              INTEGER PRIMARY KEY CHECK (id = 1),
                z_score_threshold     REAL    NOT NULL DEFAULT 2.5,
                cluster_min_size      INTEGER NOT NULL DEFAULT 3,
                cluster_similarity    REAL    NOT NULL DEFAULT 0.85,
                outlier_percentile    REAL    NOT NULL DEFAULT 95.0,
                collusion_threshold   REAL    NOT NULL DEFAULT 0.80,
                template_threshold    REAL    NOT NULL DEFAULT 0.75,
                enable_statistical    INTEGER NOT NULL DEFAULT 1,
                enable_cluster        INTEGER NOT NULL DEFAULT 1,
                enable_pattern        INTEGER NOT NULL DEFAULT 1,
                enable_collusion      INTEGER NOT NULL DEFAULT 1,
                updated_at            TEXT
            );

            INSERT OR IGNORE INTO anomaly_config (id) VALUES (1);
        """
        )
    logger.info("Anomaly alerts DB initialized at %s", db_path or DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# AnomalyAlertRepository
# ---------------------------------------------------------------------------


class AnomalyAlertRepository:
    """CRUD + analytics for anomaly alerts and scans."""

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path

    def _conn(self):
        return get_connection(self._db_path)

    # -- Scan management ------------------------------------------------------

    def create_scan(
        self,
        scan_type: str = "full",
        triggered_by: str = "system",
        config: dict | None = None,
    ) -> int:
        """Create a new scan record and return its ID."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO anomaly_scans
                   (scan_type, status, started_at, triggered_by, config_snapshot)
                   VALUES (?, 'running', ?, ?, ?)""",
                (scan_type, now, triggered_by, json.dumps(config or {})),
            )
            return cursor.lastrowid

    def complete_scan(
        self,
        scan_id: int,
        documents_scanned: int = 0,
        anomalies_found: int = 0,
    ) -> None:
        """Mark a scan as completed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """UPDATE anomaly_scans
                   SET status = 'completed', completed_at = ?,
                       documents_scanned = ?, anomalies_found = ?
                   WHERE id = ?""",
                (now, documents_scanned, anomalies_found, scan_id),
            )

    def fail_scan(self, scan_id: int, error_message: str) -> None:
        """Mark a scan as failed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """UPDATE anomaly_scans
                   SET status = 'failed', completed_at = ?, error_message = ?
                   WHERE id = ?""",
                (now, error_message, scan_id),
            )

    def list_scans(
        self,
        page: int = 1,
        per_page: int = 20,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List scans with optional status filter."""
        conditions: list[str] = []
        params: list[Any] = []
        if status:
            conditions.append("status = ?")
            params.append(status)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""

        with self._conn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM anomaly_scans{where}",  # nosec
                params,  # nosec
            ).fetchone()[0]

            offset = (page - 1) * per_page
            cursor = conn.execute(
                f"""SELECT * FROM anomaly_scans{where}  # nosec
                    ORDER BY started_at DESC LIMIT ? OFFSET ?""",
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

    def get_scan(self, scan_id: int) -> dict[str, Any] | None:
        """Get a scan by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM anomaly_scans WHERE id = ?", (scan_id,)
            ).fetchone()
            return dict(row) if row else None

    # -- Alert management -----------------------------------------------------

    def create_alert(
        self,
        scan_id: int | None,
        anomaly_type: str,
        severity: str,
        title: str,
        description: str = "",
        confidence: float = 0.0,
        affected_docs: list[str] | None = None,
        evidence: dict | None = None,
    ) -> int:
        """Create an anomaly alert."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO anomaly_alerts
                   (scan_id, anomaly_type, severity, title, description,
                    confidence, affected_docs, evidence, detected_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    scan_id,
                    anomaly_type,
                    severity,
                    title,
                    description,
                    confidence,
                    json.dumps(affected_docs or []),
                    json.dumps(evidence or {}),
                    now,
                ),
            )
            return cursor.lastrowid

    def get_alert(self, alert_id: int) -> dict[str, Any] | None:
        """Get an alert by ID."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM anomaly_alerts WHERE id = ?", (alert_id,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["affected_docs"] = json.loads(d.get("affected_docs", "[]"))
            d["evidence"] = json.loads(d.get("evidence", "{}"))
            return d

    def list_alerts(
        self,
        page: int = 1,
        per_page: int = 20,
        severity: str | None = None,
        anomaly_type: str | None = None,
        acknowledged: bool | None = None,
        resolved: bool | None = None,
        scan_id: int | None = None,
    ) -> dict[str, Any]:
        """List alerts with filtering and pagination."""
        conditions: list[str] = []
        params: list[Any] = []

        if severity:
            conditions.append("severity = ?")
            params.append(severity)
        if anomaly_type:
            conditions.append("anomaly_type = ?")
            params.append(anomaly_type)
        if acknowledged is not None:
            conditions.append("is_acknowledged = ?")
            params.append(int(acknowledged))
        if resolved is not None:
            conditions.append("is_resolved = ?")
            params.append(int(resolved))
        if scan_id is not None:
            conditions.append("scan_id = ?")
            params.append(scan_id)

        where = " WHERE " + " AND ".join(conditions) if conditions else ""

        with self._conn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM anomaly_alerts{where}",  # nosec
                params,  # nosec
            ).fetchone()[0]

            offset = (page - 1) * per_page
            cursor = conn.execute(
                f"""SELECT * FROM anomaly_alerts{where}  # nosec
                    ORDER BY detected_at DESC LIMIT ? OFFSET ?""",
                params + [per_page, offset],
            )
            items = []
            for row in cursor.fetchall():
                d = dict(row)
                d["affected_docs"] = json.loads(d.get("affected_docs", "[]"))
                d["evidence"] = json.loads(d.get("evidence", "{}"))
                items.append(d)

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

    def acknowledge_alert(self, alert_id: int, by: str = "system") -> bool:
        """Acknowledge an alert."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """UPDATE anomaly_alerts
                   SET is_acknowledged = 1, acknowledged_by = ?, acknowledged_at = ?
                   WHERE id = ? AND is_acknowledged = 0""",
                (by, now, alert_id),
            )
            return cursor.rowcount > 0

    def resolve_alert(self, alert_id: int, by: str = "system", notes: str = "") -> bool:
        """Resolve an alert."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """UPDATE anomaly_alerts
                   SET is_resolved = 1, resolved_by = ?, resolved_at = ?, notes = ?
                   WHERE id = ? AND is_resolved = 0""",
                (by, now, notes, alert_id),
            )
            return cursor.rowcount > 0

    def add_notes(self, alert_id: int, notes: str) -> bool:
        """Add notes to an alert."""
        with self._conn() as conn:
            cursor = conn.execute(
                "UPDATE anomaly_alerts SET notes = ? WHERE id = ?",
                (notes, alert_id),
            )
            return cursor.rowcount > 0

    def acknowledge_all(self, by: str = "system") -> int:
        """Acknowledge all unacknowledged alerts."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                """UPDATE anomaly_alerts
                   SET is_acknowledged = 1, acknowledged_by = ?, acknowledged_at = ?
                   WHERE is_acknowledged = 0""",
                (by, now),
            )
            return cursor.rowcount

    def delete_alert(self, alert_id: int) -> bool:
        """Delete an alert."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM anomaly_alerts WHERE id = ?", (alert_id,)
            )
            return cursor.rowcount > 0

    # -- Analytics ------------------------------------------------------------

    def analytics_summary(self) -> dict[str, Any]:
        """Return aggregate statistics across all alerts."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM anomaly_alerts").fetchone()[0]
            unacknowledged = conn.execute(
                "SELECT COUNT(*) FROM anomaly_alerts WHERE is_acknowledged = 0"
            ).fetchone()[0]
            unresolved = conn.execute(
                "SELECT COUNT(*) FROM anomaly_alerts WHERE is_resolved = 0"
            ).fetchone()[0]
            critical_unresolved = conn.execute(
                """SELECT COUNT(*) FROM anomaly_alerts
                   WHERE is_resolved = 0 AND severity = 'critical'"""
            ).fetchone()[0]

            avg_confidence = (
                conn.execute("SELECT AVG(confidence) FROM anomaly_alerts").fetchone()[0]
                or 0.0
            )

            total_scans = conn.execute("SELECT COUNT(*) FROM anomaly_scans").fetchone()[
                0
            ]

            completed_scans = conn.execute(
                "SELECT COUNT(*) FROM anomaly_scans WHERE status = 'completed'"
            ).fetchone()[0]

            return {
                "total_alerts": total,
                "unacknowledged": unacknowledged,
                "unresolved": unresolved,
                "critical_unresolved": critical_unresolved,
                "avg_confidence": round(avg_confidence, 4),
                "total_scans": total_scans,
                "completed_scans": completed_scans,
            }

    def severity_distribution(self) -> dict[str, int]:
        """Return alert counts per severity level."""
        with self._conn() as conn:
            cursor = conn.execute(
                """SELECT severity, COUNT(*) as cnt
                   FROM anomaly_alerts
                   GROUP BY severity
                   ORDER BY cnt DESC"""
            )
            return {row["severity"]: row["cnt"] for row in cursor.fetchall()}

    def type_distribution(self) -> dict[str, int]:
        """Return alert counts per anomaly type."""
        with self._conn() as conn:
            cursor = conn.execute(
                """SELECT anomaly_type, COUNT(*) as cnt
                   FROM anomaly_alerts
                   GROUP BY anomaly_type
                   ORDER BY cnt DESC"""
            )
            return {row["anomaly_type"]: row["cnt"] for row in cursor.fetchall()}

    def recent_alerts(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent alerts."""
        with self._conn() as conn:
            cursor = conn.execute(
                """SELECT * FROM anomaly_alerts
                   ORDER BY detected_at DESC LIMIT ?""",
                (limit,),
            )
            items = []
            for row in cursor.fetchall():
                d = dict(row)
                d["affected_docs"] = json.loads(d.get("affected_docs", "[]"))
                d["evidence"] = json.loads(d.get("evidence", "{}"))
                items.append(d)
            return items

    def high_confidence_alerts(
        self, min_confidence: float = 0.8, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return alerts above a confidence threshold."""
        with self._conn() as conn:
            cursor = conn.execute(
                """SELECT * FROM anomaly_alerts
                   WHERE confidence >= ? AND is_resolved = 0
                   ORDER BY confidence DESC LIMIT ?""",
                (min_confidence, limit),
            )
            items = []
            for row in cursor.fetchall():
                d = dict(row)
                d["affected_docs"] = json.loads(d.get("affected_docs", "[]"))
                d["evidence"] = json.loads(d.get("evidence", "{}"))
                items.append(d)
            return items

    # -- Config management ----------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        """Get the current anomaly detection configuration."""
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM anomaly_config WHERE id = 1").fetchone()
            return dict(row) if row else {}

    def update_config(self, **kwargs) -> dict[str, Any]:
        """Update anomaly detection configuration."""
        allowed = {
            "z_score_threshold",
            "cluster_min_size",
            "cluster_similarity",
            "outlier_percentile",
            "collusion_threshold",
            "template_threshold",
            "enable_statistical",
            "enable_cluster",
            "enable_pattern",
            "enable_collusion",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_config()

        now = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [now]

        with self._conn() as conn:
            conn.execute(
                f"UPDATE anomaly_config SET {set_clause} WHERE id = 1",  # nosec
                values,
            )
        return self.get_config()

    # -- Maintenance ----------------------------------------------------------

    def purge_old_scans(self, keep_count: int = 50) -> int:
        """Delete old scan records, keeping only the most recent ones."""
        with self._conn() as conn:
            cursor = conn.execute(
                """DELETE FROM anomaly_scans WHERE id NOT IN
                   (SELECT id FROM anomaly_scans ORDER BY started_at DESC LIMIT ?)""",
                (keep_count,),
            )
            return cursor.rowcount


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

anomaly_repo = AnomalyAlertRepository()
