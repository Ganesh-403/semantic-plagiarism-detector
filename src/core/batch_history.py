"""
Batch History Tracking and Persistence Layer.

Provides persistent storage and retrieval of batch processing history
with search, filtering, and analytics capabilities.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HistoryRecord:
    """A single history record for a batch job."""

    job_id: str
    name: str
    status: str
    priority: str
    document_count: int
    flagged_count: int
    high_severity_count: int
    progress: float
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    duration_seconds: Optional[float]
    error_message: Optional[str]
    metadata: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status,
            "priority": self.priority,
            "document_count": self.document_count,
            "flagged_count": self.flagged_count,
            "high_severity_count": self.high_severity_count,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class BatchHistory:
    """
    Persistent storage for batch processing history.

    Uses SQLite for reliable, queryable storage with automatic cleanup.
    """

    def __init__(self, db_path: str = "batch_history.db"):
        """
        Initialize history storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
        logger.info(f"BatchHistory initialized with {db_path}")

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_history (
                job_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                priority TEXT DEFAULT 'normal',
                document_count INTEGER DEFAULT 0,
                flagged_count INTEGER DEFAULT 0,
                high_severity_count INTEGER DEFAULT 0,
                progress REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                duration_seconds REAL,
                error_message TEXT,
                metadata TEXT DEFAULT '{}'
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_results (
                job_id TEXT PRIMARY KEY,
                results_json TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES batch_history(job_id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON batch_history(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created ON batch_history(created_at)
        """)
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def record_job(self, job_data: Dict[str, Any]) -> bool:
        """
        Record or update a batch job in history.

        Args:
            job_data: Job data dictionary

        Returns:
            True if recorded successfully
        """
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            duration = None
            if job_data.get("started_at") and job_data.get("completed_at"):
                try:
                    start = datetime.fromisoformat(job_data["started_at"])
                    end = datetime.fromisoformat(job_data["completed_at"])
                    duration = (end - start).total_seconds()
                except (ValueError, TypeError):
                    pass

            cursor.execute(
                """
                INSERT OR REPLACE INTO batch_history
                (job_id, name, status, priority, document_count, flagged_count,
                 high_severity_count, progress, created_at, started_at,
                 completed_at, duration_seconds, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    job_data.get("job_id"),
                    job_data.get("name", "Unnamed"),
                    job_data.get("status", "pending"),
                    job_data.get("priority", "normal"),
                    job_data.get("total_documents", 0),
                    job_data.get("flagged_pairs", 0),
                    job_data.get("high_severity_count", 0),
                    job_data.get("progress", 0.0),
                    job_data.get("created_at"),
                    job_data.get("started_at"),
                    job_data.get("completed_at"),
                    duration,
                    job_data.get("error_message"),
                    json.dumps(job_data.get("metadata", {})),
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to record job: {e}")
            return False

    def store_results(self, job_id: str, results: Dict[str, Any]) -> bool:
        """Store detailed results for a job."""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO batch_results (job_id, results_json)
                VALUES (?, ?)
            """,
                (job_id, json.dumps(results, default=str)),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to store results: {e}")
            return False

    def get_job(self, job_id: str) -> Optional[HistoryRecord]:
        """Get a specific job by ID."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM batch_history WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return self._row_to_record(row)

    def get_recent_jobs(self, limit: int = 20) -> List[HistoryRecord]:
        """Get recent jobs sorted by creation date."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM batch_history ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_record(row) for row in rows]

    def search_jobs(
        self,
        query: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_documents: Optional[int] = None,
        max_documents: Optional[int] = None,
        limit: int = 50,
    ) -> List[HistoryRecord]:
        """
        Search jobs with filters.

        Args:
            query: Text search on job name
            status: Filter by status
            priority: Filter by priority
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            min_documents: Minimum document count
            max_documents: Maximum document count
            limit: Maximum results

        Returns:
            List of matching records
        """
        conditions = []
        params = []

        if query:
            conditions.append("name LIKE ?")
            params.append(f"%{query}%")
        if status:
            conditions.append("status = ?")
            params.append(status)
        if priority:
            conditions.append("priority = ?")
            params.append(priority)
        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date)
        if min_documents is not None:
            conditions.append("document_count >= ?")
            params.append(min_documents)
        if max_documents is not None:
            conditions.append("document_count <= ?")
            params.append(max_documents)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query_sql = f"SELECT * FROM batch_history WHERE {where_clause} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(query_sql, params)
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_record(row) for row in rows]

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        conn = self._get_conn()
        cursor = conn.cursor()

        stats = {}
        cursor.execute("SELECT COUNT(*) FROM batch_history")
        stats["total_jobs"] = cursor.fetchone()[0]

        cursor.execute("SELECT status, COUNT(*) FROM batch_history GROUP BY status")
        stats["by_status"] = dict(cursor.fetchall())

        cursor.execute("SELECT priority, COUNT(*) FROM batch_history GROUP BY priority")
        stats["by_priority"] = dict(cursor.fetchall())

        cursor.execute("SELECT SUM(document_count) FROM batch_history")
        stats["total_documents"] = cursor.fetchone()[0] or 0

        cursor.execute("SELECT SUM(flagged_count) FROM batch_history")
        stats["total_flagged"] = cursor.fetchone()[0] or 0

        cursor.execute(
            "SELECT AVG(duration_seconds) FROM batch_history WHERE duration_seconds IS NOT NULL"
        )
        avg_duration = cursor.fetchone()[0]
        stats["avg_duration_seconds"] = round(avg_duration, 2) if avg_duration else 0

        cursor.execute(
            "SELECT AVG(progress) FROM batch_history WHERE status = 'completed'"
        )
        avg_progress = cursor.fetchone()[0]
        stats["avg_completion_rate"] = round(avg_progress, 2) if avg_progress else 0

        conn.close()
        return stats

    def get_daily_summary(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily job summaries for the last N days."""
        conn = self._get_conn()
        cursor = conn.cursor()
        since = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute(
            """
            SELECT DATE(created_at) as day, COUNT(*) as jobs,
                   SUM(document_count) as documents,
                   SUM(flagged_count) as flagged
            FROM batch_history
            WHERE created_at >= ?
            GROUP BY DATE(created_at)
            ORDER BY day
        """,
            (since,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "date": row[0],
                "jobs": row[1],
                "documents": row[2] or 0,
                "flagged": row[3] or 0,
            }
            for row in rows
        ]

    def cleanup_old_records(self, days: int = 90) -> int:
        """Remove records older than N days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM batch_results WHERE job_id IN (SELECT job_id FROM batch_history WHERE created_at < ?)",
            (cutoff,),
        )
        cursor.execute("DELETE FROM batch_history WHERE created_at < ?", (cutoff,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        logger.info(f"Cleaned up {deleted} old records")
        return deleted

    def _row_to_record(self, row: Tuple) -> HistoryRecord:
        """Convert database row to HistoryRecord."""
        return HistoryRecord(
            job_id=row[0],
            name=row[1],
            status=row[2],
            priority=row[3],
            document_count=row[4],
            flagged_count=row[5],
            high_severity_count=row[6],
            progress=row[7],
            created_at=row[8],
            started_at=row[9],
            completed_at=row[10],
            duration_seconds=row[11],
            error_message=row[12],
            metadata=row[13],
        )
