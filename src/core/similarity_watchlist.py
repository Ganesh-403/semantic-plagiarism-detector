"""
src/core/similarity_watchlist.py
---------------------------------
Similarity watchlist system for persistent document monitoring.

Allows users to maintain watchlists of documents or document pairs that
should be tracked across scans. When new scans find matches against
watchlisted items, alerts are generated with full match details.

Integrates with the existing corpus.db scan pipeline and feeds into
the notification and recommendation systems.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np

from src.core.config import (
    DEFAULT_THRESHOLDS,
    HIGH_SEVERITY,
    LOW_SEVERITY,
    MEDIUM_SEVERITY,
    SimilarityThresholds,
    normalize_score,
    severity_from_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data types
# ---------------------------------------------------------------------------

class WatchlistType(str, Enum):
    """Type of watchlist entry."""
    DOCUMENT = "document"       # Watch a single document against corpus
    PAIR = "pair"               # Watch a specific document pair
    TAG = "tag"                 # Watch all documents with a given tag
    CLASS = "class_section"     # Watch all documents in a class section


class WatchlistStatus(str, Enum):
    """Status of a watchlist entry."""
    ACTIVE = "active"
    PAUSED = "paused"
    RESOLVED = "resolved"


class AlertTrigger(str, Enum):
    """What triggered a watchlist alert."""
    NEW_SCAN = "new_scan"
    RESCAN = "rescan"
    MANUAL_CHECK = "manual_check"


@dataclass
class WatchlistEntry:
    """A single watchlist monitoring rule."""
    entry_id: Optional[int] = None
    watchlist_type: WatchlistType = WatchlistType.DOCUMENT
    target: str = ""
    label: str = ""
    description: str = ""
    status: WatchlistStatus = WatchlistStatus.ACTIVE
    similarity_threshold: float = 0.0  # Override global threshold (0 = use default)
    created_by: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "watchlist_type": self.watchlist_type.value,
            "target": self.target,
            "label": self.label,
            "description": self.description,
            "status": self.status.value,
            "similarity_threshold": self.similarity_threshold,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class WatchlistAlert:
    """An alert generated when a scan matches a watchlist entry."""
    alert_id: Optional[int] = None
    entry_id: int = 0
    triggered_by: AlertTrigger = AlertTrigger.NEW_SCAN
    matched_document: str = ""
    similarity_score: float = 0.0
    severity: str = LOW_SEVERITY
    scan_timestamp: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "entry_id": self.entry_id,
            "triggered_by": self.triggered_by.value,
            "matched_document": self.matched_document,
            "similarity_score": round(self.similarity_score, 6),
            "severity": self.severity,
            "scan_timestamp": self.scan_timestamp,
            "details": self.details,
            "acknowledged": self.acknowledged,
            "created_at": self.created_at,
        }


@dataclass
class WatchlistCheckResult:
    """Result of checking scan results against a watchlist."""
    entry: WatchlistEntry
    matches: List[WatchlistAlert]
    checked_at: str
    scan_document_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "matches": [m.to_dict() for m in self.matches],
            "checked_at": self.checked_at,
            "scan_document_count": self.scan_document_count,
            "match_count": len(self.matches),
        }


@dataclass
class WatchlistSummary:
    """Summary statistics for a watchlist."""
    total_entries: int
    active_entries: int
    paused_entries: int
    resolved_entries: int
    total_alerts: int
    unacknowledged_alerts: int
    entries_by_type: Dict[str, int]
    alerts_by_severity: Dict[str, int]
    recent_alerts: List[WatchlistAlert]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entries": self.total_entries,
            "active_entries": self.active_entries,
            "paused_entries": self.paused_entries,
            "resolved_entries": self.resolved_entries,
            "total_alerts": self.total_alerts,
            "unacknowledged_alerts": self.unacknowledged_alerts,
            "entries_by_type": self.entries_by_type,
            "alerts_by_severity": self.alerts_by_severity,
            "recent_alerts": [a.to_dict() for a in self.recent_alerts],
        }


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

_connection_pool = threading.local()


def _get_connection(db_path: str) -> sqlite3.Connection:
    """Get or create a thread-local SQLite connection."""
    pool = getattr(_connection_pool, "connections", {})
    if db_path in pool:
        try:
            pool[db_path].execute("SELECT 1")
            return pool[db_path]
        except sqlite3.ProgrammingError:
            del pool[db_path]

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    pool[db_path] = conn
    _connection_pool.connections = pool
    return conn


@contextmanager
def _connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for a managed SQLite connection."""
    conn = _get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_watchlist_db(db_path: str) -> None:
    """Create watchlist tables if they don't exist."""
    with _connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_entries (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                watchlist_type TEXT NOT NULL,
                target TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'paused', 'resolved')),
                similarity_threshold REAL NOT NULL DEFAULT 0.0,
                created_by TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER NOT NULL,
                triggered_by TEXT NOT NULL,
                matched_document TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                severity TEXT NOT NULL,
                scan_timestamp TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '{}',
                acknowledged INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (entry_id)
                    REFERENCES watchlist_entries(entry_id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_watchlist_alerts_entry
            ON watchlist_alerts(entry_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_watchlist_alerts_ack
            ON watchlist_alerts(acknowledged)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_watchlist_entries_status
            ON watchlist_entries(status)
        """)


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class SimilarityWatchlistRepository:
    """Data access layer for watchlist entries and alerts.

    Uses SQLite for persistence with WAL mode for concurrent reads.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        init_watchlist_db(db_path)

    # -- Entry CRUD --

    def add_entry(self, entry: WatchlistEntry) -> int:
        """Insert a new watchlist entry and return its ID."""
        now = datetime.now().isoformat()
        meta_json = json.dumps(entry.metadata, ensure_ascii=False)
        with _connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO watchlist_entries
                (watchlist_type, target, label, description, status,
                 similarity_threshold, created_by, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.watchlist_type.value,
                    entry.target,
                    entry.label,
                    entry.description,
                    entry.status.value,
                    entry.similarity_threshold,
                    entry.created_by,
                    now,
                    now,
                    meta_json,
                ),
            )
            return cursor.lastrowid

    def get_entry(self, entry_id: int) -> Optional[WatchlistEntry]:
        """Fetch a single watchlist entry by ID."""
        with _connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM watchlist_entries WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_all_entries(
        self,
        status: Optional[WatchlistStatus] = None,
        watchlist_type: Optional[WatchlistType] = None,
    ) -> List[WatchlistEntry]:
        """Fetch all entries with optional filtering."""
        query = "SELECT * FROM watchlist_entries WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status.value)
        if watchlist_type:
            query += " AND watchlist_type = ?"
            params.append(watchlist_type.value)
        query += " ORDER BY created_at DESC"

        with _connection(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def update_entry_status(
        self, entry_id: int, status: WatchlistStatus
    ) -> bool:
        """Update the status of a watchlist entry."""
        now = datetime.now().isoformat()
        with _connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE watchlist_entries
                SET status = ?, updated_at = ?
                WHERE entry_id = ?
                """,
                (status.value, now, entry_id),
            )
            return cursor.rowcount > 0

    def delete_entry(self, entry_id: int) -> bool:
        """Delete a watchlist entry and its alerts."""
        with _connection(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM watchlist_entries WHERE entry_id = ?",
                (entry_id,),
            )
            return cursor.rowcount > 0

    def find_entries_for_document(
        self, document_name: str
    ) -> List[WatchlistEntry]:
        """Find all active entries that match a given document name.

        Matches against DOCUMENT-type entries by exact name, TAG-type
        entries if the document has that tag, and CLASS-type entries if
        the document belongs to that class section.
        """
        entries = self.get_all_entries(status=WatchlistStatus.ACTIVE)
        matches = []
        for entry in entries:
            if entry.watchlist_type == WatchlistType.DOCUMENT:
                if entry.target == document_name:
                    matches.append(entry)
            elif entry.watchlist_type == WatchlistType.PAIR:
                if document_name in entry.target.split("|"):
                    matches.append(entry)
        return matches

    def find_entries_for_pair(
        self, doc_a: str, doc_b: str
    ) -> List[WatchlistEntry]:
        """Find active entries matching a document pair."""
        entries = self.get_all_entries(status=WatchlistStatus.ACTIVE)
        matches = []
        for entry in entries:
            if entry.watchlist_type == WatchlistType.PAIR:
                targets = entry.target.split("|")
                if (
                    (doc_a in targets and doc_b in targets)
                    or (doc_a in targets or doc_b in targets)
                ):
                    matches.append(entry)
            elif entry.watchlist_type == WatchlistType.DOCUMENT:
                if document_name in (doc_a, doc_b):
                    matches.append(entry)
        return matches

    # -- Alert CRUD --

    def add_alert(self, alert: WatchlistAlert) -> int:
        """Insert a new watchlist alert and return its ID."""
        now = datetime.now().isoformat()
        details_json = json.dumps(alert.details, ensure_ascii=False)
        with _connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO watchlist_alerts
                (entry_id, triggered_by, matched_document, similarity_score,
                 severity, scan_timestamp, details, acknowledged, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.entry_id,
                    alert.triggered_by.value,
                    alert.matched_document,
                    alert.similarity_score,
                    alert.severity,
                    alert.scan_timestamp,
                    details_json,
                    int(alert.acknowledged),
                    now,
                ),
            )
            return cursor.lastrowid

    def get_alerts(
        self,
        entry_id: Optional[int] = None,
        acknowledged: Optional[bool] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[WatchlistAlert]:
        """Fetch alerts with optional filtering."""
        query = "SELECT * FROM watchlist_alerts WHERE 1=1"
        params: list[Any] = []
        if entry_id is not None:
            query += " AND entry_id = ?"
            params.append(entry_id)
        if acknowledged is not None:
            query += " AND acknowledged = ?"
            params.append(int(acknowledged))
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with _connection(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_alert(r) for r in rows]

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged."""
        with _connection(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE watchlist_alerts SET acknowledged = 1 WHERE alert_id = ?",
                (alert_id,),
            )
            return cursor.rowcount > 0

    def acknowledge_all_alerts(self, entry_id: Optional[int] = None) -> int:
        """Acknowledge all alerts, optionally for a specific entry."""
        query = "UPDATE watchlist_alerts SET acknowledged = 1 WHERE acknowledged = 0"
        params: list[Any] = []
        if entry_id is not None:
            query += " AND entry_id = ?"
            params.append(entry_id)
        with _connection(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount

    def get_alert_count(
        self, entry_id: Optional[int] = None, acknowledged: Optional[bool] = None
    ) -> int:
        """Count alerts matching the given filters."""
        query = "SELECT COUNT(*) FROM watchlist_alerts WHERE 1=1"
        params: list[Any] = []
        if entry_id is not None:
            query += " AND entry_id = ?"
            params.append(entry_id)
        if acknowledged is not None:
            query += " AND acknowledged = ?"
            params.append(int(acknowledged))
        with _connection(self.db_path) as conn:
            row = conn.execute(query, params).fetchone()
            return row[0] if row else 0

    # -- Summary --

    def get_summary(self) -> WatchlistSummary:
        """Compute aggregate watchlist statistics."""
        entries = self.get_all_entries()
        all_alerts = self.get_alerts(limit=10000)
        unacked = self.get_alerts(acknowledged=False, limit=10000)

        entries_by_type: Dict[str, int] = {}
        for e in entries:
            key = e.watchlist_type.value
            entries_by_type[key] = entries_by_type.get(key, 0) + 1

        alerts_by_severity: Dict[str, int] = {}
        for a in all_alerts:
            alerts_by_severity[a.severity] = (
                alerts_by_severity.get(a.severity, 0) + 1
            )

        return WatchlistSummary(
            total_entries=len(entries),
            active_entries=sum(
                1 for e in entries if e.status == WatchlistStatus.ACTIVE
            ),
            paused_entries=sum(
                1 for e in entries if e.status == WatchlistStatus.PAUSED
            ),
            resolved_entries=sum(
                1 for e in entries if e.status == WatchlistStatus.RESOLVED
            ),
            total_alerts=len(all_alerts),
            unacknowledged_alerts=len(unacked),
            entries_by_type=entries_by_type,
            alerts_by_severity=alerts_by_severity,
            recent_alerts=unacked[:10],
        )

    # -- Internal helpers --

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> WatchlistEntry:
        return WatchlistEntry(
            entry_id=row["entry_id"],
            watchlist_type=WatchlistType(row["watchlist_type"]),
            target=row["target"],
            label=row["label"],
            description=row["description"],
            status=WatchlistStatus(row["status"]),
            similarity_threshold=row["similarity_threshold"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    @staticmethod
    def _row_to_alert(row: sqlite3.Row) -> WatchlistAlert:
        return WatchlistAlert(
            alert_id=row["alert_id"],
            entry_id=row["entry_id"],
            triggered_by=AlertTrigger(row["triggered_by"]),
            matched_document=row["matched_document"],
            similarity_score=row["similarity_score"],
            severity=row["severity"],
            scan_timestamp=row["scan_timestamp"],
            details=json.loads(row["details"]) if row["details"] else {},
            acknowledged=bool(row["acknowledged"]),
            created_at=row["created_at"],
        )


# ---------------------------------------------------------------------------
# Watchlist checker
# ---------------------------------------------------------------------------

class SimilarityWatchlistChecker:
    """Checks scan results against watchlist entries and generates alerts.

    Usage::

        repo = SimilarityWatchlistRepository("data/watchlist.db")
        checker = SimilarityWatchlistChecker(repo)
        results = checker.check_scan_results(
            doc_names=["a.pdf", "b.pdf"],
            similarity_matrix=sim_matrix,
            trigger=AlertTrigger.NEW_SCAN,
        )
    """

    def __init__(
        self,
        repository: SimilarityWatchlistRepository,
        default_thresholds: SimilarityThresholds = DEFAULT_THRESHOLDS,
    ) -> None:
        self.repo = repository
        self.default_thresholds = default_thresholds

    def check_scan_results(
        self,
        doc_names: List[str],
        similarity_matrix: np.ndarray,
        trigger: AlertTrigger = AlertTrigger.NEW_SCAN,
        scan_timestamp: Optional[str] = None,
    ) -> List[WatchlistCheckResult]:
        """Check scan results against all active watchlist entries.

        Args:
            doc_names: Ordered document names from the scan.
            similarity_matrix: NxN cosine similarity matrix.
            trigger: What triggered this check.
            scan_timestamp: Override timestamp (default: now).

        Returns:
            List of WatchlistCheckResult, one per checked entry.
        """
        if not doc_names or similarity_matrix.size == 0:
            return []

        timestamp = scan_timestamp or datetime.now().isoformat()
        entries = self.repo.get_all_entries(status=WatchlistStatus.ACTIVE)
        if not entries:
            return []

        doc_index = {name: i for i, name in enumerate(doc_names)}
        results: List[WatchlistCheckResult] = []

        for entry in entries:
            matches = self._check_entry(
                entry, doc_names, doc_index, similarity_matrix, trigger, timestamp
            )
            results.append(WatchlistCheckResult(
                entry=entry,
                matches=matches,
                checked_at=timestamp,
                scan_document_count=len(doc_names),
            ))

        return results

    def _check_entry(
        self,
        entry: WatchlistEntry,
        doc_names: List[str],
        doc_index: Dict[str, int],
        similarity_matrix: np.ndarray,
        trigger: AlertTrigger,
        timestamp: str,
    ) -> List[WatchlistAlert]:
        """Check a single entry against scan results."""
        threshold = (
            entry.similarity_threshold
            if entry.similarity_threshold > 0
            else self.default_thresholds.plagiarism
        )
        matches: List[WatchlistAlert] = []

        if entry.watchlist_type == WatchlistType.DOCUMENT:
            matches.extend(self._check_document_entry(
                entry, doc_names, doc_index, similarity_matrix, threshold, trigger, timestamp
            ))
        elif entry.watchlist_type == WatchlistType.PAIR:
            matches.extend(self._check_pair_entry(
                entry, doc_names, doc_index, similarity_matrix, threshold, trigger, timestamp
            ))

        return matches

    def _check_document_entry(
        self,
        entry: WatchlistEntry,
        doc_names: List[str],
        doc_index: Dict[str, int],
        similarity_matrix: np.ndarray,
        threshold: float,
        trigger: AlertTrigger,
        timestamp: str,
    ) -> List[WatchlistAlert]:
        """Check a DOCUMENT-type entry against scan results."""
        target_idx = doc_index.get(entry.target)
        if target_idx is None:
            return []

        matches = []
        for i, name in enumerate(doc_names):
            if i == target_idx:
                continue
            sim = float(similarity_matrix[target_idx, i])
            if sim >= threshold:
                severity = severity_from_score(sim, self.default_thresholds)
                alert = WatchlistAlert(
                    entry_id=entry.entry_id,
                    triggered_by=trigger,
                    matched_document=name,
                    similarity_score=sim,
                    severity=severity,
                    scan_timestamp=timestamp,
                    details={
                        "watched_document": entry.target,
                        "similarity_score": sim,
                        "threshold": threshold,
                    },
                    created_at=timestamp,
                )
                alert_id = self.repo.add_alert(alert)
                alert.alert_id = alert_id
                matches.append(alert)

        return matches

    def _check_pair_entry(
        self,
        entry: WatchlistEntry,
        doc_names: List[str],
        doc_index: Dict[str, int],
        similarity_matrix: np.ndarray,
        threshold: float,
        trigger: AlertTrigger,
        timestamp: str,
    ) -> List[WatchlistAlert]:
        """Check a PAIR-type entry against scan results."""
        targets = entry.target.split("|")
        if len(targets) != 2:
            return []

        idx_a = doc_index.get(targets[0])
        idx_b = doc_index.get(targets[1])
        if idx_a is None or idx_b is None:
            return []

        sim = float(similarity_matrix[idx_a, idx_b])
        if sim >= threshold:
            severity = severity_from_score(sim, self.default_thresholds)
            alert = WatchlistAlert(
                entry_id=entry.entry_id,
                triggered_by=trigger,
                matched_document=f"{targets[0]} <-> {targets[1]}",
                similarity_score=sim,
                severity=severity,
                scan_timestamp=timestamp,
                details={
                    "pair": targets,
                    "similarity_score": sim,
                    "threshold": threshold,
                },
                created_at=timestamp,
            )
            alert_id = self.repo.add_alert(alert)
            alert.alert_id = alert_id
            return [alert]

        return []


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def quick_watch_document(
    db_path: str,
    document_name: str,
    label: str = "",
    threshold: float = 0.0,
    created_by: str = "system",
) -> int:
    """Add a document to the watchlist in one call.

    Args:
        db_path: Path to the watchlist database.
        document_name: Document filename to watch.
        label: Optional human-readable label.
        threshold: Custom similarity threshold (0 = use default).
        created_by: Username of the creator.

    Returns:
        The new entry ID.
    """
    repo = SimilarityWatchlistRepository(db_path)
    entry = WatchlistEntry(
        watchlist_type=WatchlistType.DOCUMENT,
        target=document_name,
        label=label or f"Watch: {document_name}",
        similarity_threshold=threshold,
        created_by=created_by,
    )
    return repo.add_entry(entry)


def quick_watch_pair(
    db_path: str,
    doc_a: str,
    doc_b: str,
    label: str = "",
    threshold: float = 0.0,
    created_by: str = "system",
) -> int:
    """Add a document pair to the watchlist in one call."""
    repo = SimilarityWatchlistRepository(db_path)
    entry = WatchlistEntry(
        watchlist_type=WatchlistType.PAIR,
        target=f"{doc_a}|{doc_b}",
        label=label or f"Watch: {doc_a} <-> {doc_b}",
        similarity_threshold=threshold,
        created_by=created_by,
    )
    return repo.add_entry(entry)
