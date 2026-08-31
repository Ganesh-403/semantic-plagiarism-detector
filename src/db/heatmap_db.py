"""
heatmap_db.py
-------------
SQLite persistence layer for similarity heatmap snapshots and
clustering results.

Stores:
  - Heatmap computation snapshots (matrix + metadata)
  - Clustering results with cluster assignments
  - Similarity hotspot alerts
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
# Connection pool
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
    """Open a pooled connection."""
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
    global _DB_PATH
    close_connections()
    _DB_PATH = os.path.abspath(os.fspath(db_path))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_heatmap_db() -> None:
    """Create the heatmap and clustering tables if they do not exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS heatmap_snapshots (
                snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                computed_at     TEXT    NOT NULL,
                document_count  INTEGER NOT NULL,
                labels_json     TEXT    NOT NULL,
                matrix_json     TEXT    NOT NULL,
                min_similarity  REAL    NOT NULL DEFAULT 0.0,
                max_similarity  REAL    NOT NULL DEFAULT 0.0,
                mean_similarity REAL    NOT NULL DEFAULT 0.0,
                computed_by     TEXT,
                notes           TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clustering_results (
                result_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id          INTEGER,
                computed_at          TEXT    NOT NULL,
                num_clusters         INTEGER NOT NULL DEFAULT 0,
                silhouette_score     REAL    NOT NULL DEFAULT 0.0,
                linkage_method       TEXT    NOT NULL DEFAULT 'single',
                distance_threshold   REAL    NOT NULL DEFAULT 0.5,
                clusters_json        TEXT    NOT NULL DEFAULT '[]',
                assignments_json     TEXT    NOT NULL DEFAULT '{}',
                FOREIGN KEY (snapshot_id)
                    REFERENCES heatmap_snapshots(snapshot_id)
                    ON DELETE SET NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS similarity_hotspots (
                hotspot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id  INTEGER,
                doc_a        TEXT    NOT NULL,
                doc_b        TEXT    NOT NULL,
                similarity   REAL    NOT NULL,
                severity     TEXT    NOT NULL DEFAULT 'warning'
                               CHECK (severity IN ('info','warning','critical')),
                created_at   TEXT    NOT NULL,
                is_resolved  INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (snapshot_id)
                    REFERENCES heatmap_snapshots(snapshot_id)
                    ON DELETE SET NULL
            )
            """
        )

        # Indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hs_snapshot ON heatmap_snapshots(computed_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cr_snapshot ON clustering_results(snapshot_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sh_similarity ON similarity_hotspots(similarity)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sh_resolved ON similarity_hotspots(is_resolved)"
        )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class HeatmapRepository(BaseRepository):
    """Data access object for heatmap and clustering tables."""

    def __init__(self, db_path: str | os.PathLike = _DB_PATH) -> None:
        super().__init__(db_path)

    # -- Heatmap Snapshots --------------------------------------------------

    def save_snapshot(
        self,
        labels: list[str],
        matrix: list[list[float]],
        min_similarity: float,
        max_similarity: float,
        mean_similarity: float,
        computed_by: str | None = None,
        notes: str | None = None,
    ) -> int:
        """Persist a heatmap snapshot and return its ID."""
        now = datetime.now().isoformat()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO heatmap_snapshots
                    (computed_at, document_count, labels_json, matrix_json,
                     min_similarity, max_similarity, mean_similarity,
                     computed_by, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    len(labels),
                    json.dumps(labels),
                    json.dumps(matrix),
                    min_similarity,
                    max_similarity,
                    mean_similarity,
                    computed_by,
                    notes,
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_snapshot(self, snapshot_id: int) -> dict[str, Any] | None:
        """Retrieve a heatmap snapshot by ID."""
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM heatmap_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
            return self._hydrate_snapshot(row) if row else None

    def list_snapshots(
        self, *, limit: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        """List heatmap snapshots ordered by most recent."""
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM heatmap_snapshots
                ORDER BY computed_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            return [self._hydrate_snapshot(r) for r in rows]

    def count_snapshots(self) -> int:
        with _connect() as conn:
            row = conn.execute("SELECT COUNT(1) FROM heatmap_snapshots").fetchone()
            return int(row[0]) if row else 0

    def delete_snapshot(self, snapshot_id: int) -> bool:
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM heatmap_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            return cursor.rowcount > 0

    # -- Clustering Results -------------------------------------------------

    def save_clustering(
        self,
        snapshot_id: int | None,
        num_clusters: int,
        silhouette_score: float,
        linkage_method: str,
        distance_threshold: float,
        clusters: list[dict],
        assignments: dict[str, int],
    ) -> int:
        now = datetime.now().isoformat()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO clustering_results
                    (snapshot_id, computed_at, num_clusters, silhouette_score,
                     linkage_method, distance_threshold, clusters_json,
                     assignments_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    now,
                    num_clusters,
                    silhouette_score,
                    linkage_method,
                    distance_threshold,
                    json.dumps(clusters),
                    json.dumps(assignments),
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_clustering(self, result_id: int) -> dict[str, Any] | None:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM clustering_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            return self._hydrate_clustering(row) if row else None

    def list_clusterings(
        self, *, limit: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM clustering_results
                ORDER BY computed_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            return [self._hydrate_clustering(r) for r in rows]

    def delete_clustering(self, result_id: int) -> bool:
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM clustering_results WHERE result_id = ?",
                (result_id,),
            )
            return cursor.rowcount > 0

    # -- Hotspots -----------------------------------------------------------

    def save_hotspot(
        self,
        snapshot_id: int | None,
        doc_a: str,
        doc_b: str,
        similarity: float,
        severity: str = "warning",
    ) -> int:
        now = datetime.now().isoformat()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO similarity_hotspots
                    (snapshot_id, doc_a, doc_b, similarity, severity, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (snapshot_id, doc_a, doc_b, similarity, severity, now),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def save_hotspots_batch(
        self,
        snapshot_id: int | None,
        hotspots: list[dict],
    ) -> int:
        """Batch-insert hotspots. Returns count inserted."""
        if not hotspots:
            return 0
        now = datetime.now().isoformat()
        rows = []
        for h in hotspots:
            sim = h.get("similarity", 0)
            severity = "critical" if sim >= 0.9 else "warning" if sim >= 0.7 else "info"
            rows.append((
                snapshot_id, h.get("doc_a", ""), h.get("doc_b", ""),
                sim, severity, now,
            ))

        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO similarity_hotspots
                    (snapshot_id, doc_a, doc_b, similarity, severity, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return len(rows)

    def get_hotspots(
        self,
        *,
        unresolved_only: bool = False,
        min_similarity: float | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM similarity_hotspots WHERE 1=1"
        params: list[Any] = []
        if unresolved_only:
            query += " AND is_resolved = 0"
        if min_similarity is not None:
            query += " AND similarity >= ?"
            params.append(min_similarity)
        query += " ORDER BY similarity DESC LIMIT ?"
        params.append(limit)

        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def resolve_hotspot(self, hotspot_id: int) -> bool:
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE similarity_hotspots SET is_resolved = 1 WHERE hotspot_id = ?",
                (hotspot_id,),
            )
            return cursor.rowcount > 0

    def get_unresolved_hotspot_count(self) -> int:
        with _connect() as conn:
            row = conn.execute(
                "SELECT COUNT(1) FROM similarity_hotspots WHERE is_resolved = 0"
            ).fetchone()
            return int(row[0]) if row else 0

    # -- Analytics ----------------------------------------------------------

    def get_hotspot_summary(self) -> dict[str, Any]:
        with _connect() as conn:
            total = conn.execute(
                "SELECT COUNT(1) FROM similarity_hotspots"
            ).fetchone()[0]
            unresolved = conn.execute(
                "SELECT COUNT(1) FROM similarity_hotspots WHERE is_resolved = 0"
            ).fetchone()[0]
            critical = conn.execute(
                "SELECT COUNT(1) FROM similarity_hotspots WHERE severity = 'critical' AND is_resolved = 0"
            ).fetchone()[0]
            avg_sim_row = conn.execute(
                "SELECT AVG(similarity) FROM similarity_hotspots"
            ).fetchone()
            avg_sim = float(avg_sim_row[0]) if avg_sim_row[0] else 0.0

        return {
            "total_hotspots": total,
            "unresolved": unresolved,
            "critical_unresolved": critical,
            "avg_similarity": round(avg_sim, 4),
        }

    def purge_old_data(self, days: int = 90) -> dict[str, int]:
        threshold = (datetime.now() - timedelta(days=days)).isoformat()
        with self.transaction() as conn:
            hs = conn.execute(
                "DELETE FROM heatmap_snapshots WHERE computed_at < ?", (threshold,)
            ).rowcount
            cr = conn.execute(
                "DELETE FROM clustering_results WHERE computed_at < ?", (threshold,)
            ).rowcount
            sh = conn.execute(
                "DELETE FROM similarity_hotspots WHERE created_at < ? AND is_resolved = 1",
                (threshold,),
            ).rowcount
        return {"snapshots_deleted": hs, "clusterings_deleted": cr, "hotspots_deleted": sh}

    # -- Hydrators ----------------------------------------------------------

    @staticmethod
    def _hydrate_snapshot(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for key in ("labels_json", "matrix_json"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    @staticmethod
    def _hydrate_clustering(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for key in ("clusters_json", "assignments_json"):
            if d.get(key):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d


# ---------------------------------------------------------------------------
# Module-level instance
# ---------------------------------------------------------------------------

heatmap_repo = HeatmapRepository(_DB_PATH)
