"""
src/db/watermark_verification_db.py
-----------------------------------
SQLite database manager for persisting AI Text Watermark Verification results,
confidence intervals, and statistical test audit trails.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/watermark_verifications.db")


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Context manager for acquiring and releasing SQLite database connections."""
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


def initialize_watermark_verification_db(db_path: Optional[Path] = None) -> None:
    """Initialize the watermark verification database schema and indexes."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watermark_verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verification_id TEXT UNIQUE NOT NULL,
                document_id TEXT NOT NULL,
                total_tokens INTEGER NOT NULL,
                green_tokens INTEGER NOT NULL,
                red_tokens INTEGER NOT NULL,
                observed_green_ratio REAL NOT NULL,
                expected_green_ratio REAL NOT NULL,
                z_score REAL NOT NULL,
                p_value REAL NOT NULL,
                confidence_level REAL NOT NULL,
                ci_lower REAL NOT NULL,
                ci_upper REAL NOT NULL,
                confidence_score REAL NOT NULL,
                is_watermarked INTEGER NOT NULL,
                watermark_scheme TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_wm_verif_doc 
            ON watermark_verifications(document_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_wm_verif_created 
            ON watermark_verifications(created_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_wm_verif_uid 
            ON watermark_verifications(verification_id)
            """
        )

        # Legacy / lightweight table for compatibility
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watermark_verification_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                z_score REAL NOT NULL,
                p_value REAL NOT NULL,
                is_watermarked INTEGER NOT NULL,
                analyzed_at TEXT NOT NULL
            )
            """
        )

    logger.info(
        "Watermark verification database initialized at %s", db_path or DEFAULT_DB_PATH
    )


def save_verification_result(
    document_id: str,
    total_tokens: int,
    green_tokens: int,
    red_tokens: int,
    observed_green_ratio: float,
    expected_green_ratio: float,
    z_score: float,
    p_value: float,
    confidence_level: float,
    ci_lower: float,
    ci_upper: float,
    confidence_score: float,
    is_watermarked: bool,
    watermark_scheme: str = "Maryland-Kirchenbauer",
    metadata: Optional[dict[str, Any]] = None,
    verification_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Save an AI watermark statistical verification run to the database."""
    v_id = verification_id or f"WMV-{uuid.uuid4().hex[:12].upper()}"
    created_at = datetime.now(timezone.utc).isoformat()
    meta_json = json.dumps(metadata or {})

    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO watermark_verifications (
                    verification_id, document_id, total_tokens, green_tokens, red_tokens,
                    observed_green_ratio, expected_green_ratio, z_score, p_value,
                    confidence_level, ci_lower, ci_upper, confidence_score,
                    is_watermarked, watermark_scheme, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    v_id,
                    document_id,
                    total_tokens,
                    green_tokens,
                    red_tokens,
                    observed_green_ratio,
                    expected_green_ratio,
                    z_score,
                    p_value,
                    confidence_level,
                    ci_lower,
                    ci_upper,
                    confidence_score,
                    1 if is_watermarked else 0,
                    watermark_scheme,
                    meta_json,
                    created_at,
                ),
            )

        return get_verification_by_id(v_id, db_path=db_path)
    except sqlite3.Error as e:
        logger.error("Failed to save watermark verification %s: %s", v_id, e)
        return None


def log_watermark_verification(
    document_id: str,
    z_score: float,
    p_value: float,
    is_watermarked: bool,
    db_path: Optional[Path] = None,
) -> bool:
    """Persist a watermark verification result to watermark_verification_logs."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO watermark_verification_logs 
                (document_id, z_score, p_value, is_watermarked, analyzed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    z_score,
                    p_value,
                    1 if is_watermarked else 0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return True
    except sqlite3.Error as e:
        logger.error("Failed to log watermark verification: %s", e)
        return False


def get_verification_by_id(
    verification_id: str, db_path: Optional[Path] = None
) -> Optional[dict[str, Any]]:
    """Retrieve a watermark verification record by unique verification ID."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM watermark_verifications WHERE verification_id = ?",
                (verification_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["is_watermarked"] = bool(data["is_watermarked"])
            if data.get("metadata_json"):
                try:
                    data["metadata"] = json.loads(data["metadata_json"])
                except json.JSONDecodeError:
                    data["metadata"] = {}
            else:
                data["metadata"] = {}
            return data
    except sqlite3.Error as e:
        logger.error("Failed to retrieve verification %s: %s", verification_id, e)
        return None


def get_verifications_for_document(
    document_id: str, limit: int = 50, offset: int = 0, db_path: Optional[Path] = None
) -> list[dict[str, Any]]:
    """Retrieve all verification records associated with a document ID."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM watermark_verifications 
                WHERE document_id = ? 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
                """,
                (document_id, limit, offset),
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                item["is_watermarked"] = bool(item["is_watermarked"])
                if item.get("metadata_json"):
                    try:
                        item["metadata"] = json.loads(item["metadata_json"])
                    except json.JSONDecodeError:
                        item["metadata"] = {}
                else:
                    item["metadata"] = {}
                results.append(item)
            return results
    except sqlite3.Error as e:
        logger.error("Failed to retrieve verifications for document %s: %s", document_id, e)
        return []


def list_recent_verifications(
    limit: int = 50, offset: int = 0, db_path: Optional[Path] = None
) -> list[dict[str, Any]]:
    """List recent watermark verifications with pagination."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM watermark_verifications 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                item["is_watermarked"] = bool(item["is_watermarked"])
                if item.get("metadata_json"):
                    try:
                        item["metadata"] = json.loads(item["metadata_json"])
                    except json.JSONDecodeError:
                        item["metadata"] = {}
                else:
                    item["metadata"] = {}
                results.append(item)
            return results
    except sqlite3.Error as e:
        logger.error("Failed to list recent watermark verifications: %s", e)
        return []


def get_verification_count(db_path: Optional[Path] = None) -> int:
    """Get total count of persisted watermark verifications."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM watermark_verifications")
            return cursor.fetchone()[0]
    except sqlite3.Error as e:
        logger.error("Failed to get verification count: %s", e)
        return 0


def delete_verification(verification_id: str, db_path: Optional[Path] = None) -> bool:
    """Delete a watermark verification record."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM watermark_verifications WHERE verification_id = ?",
                (verification_id,),
            )
            return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error("Failed to delete verification %s: %s", verification_id, e)
        return False


class WatermarkVerificationDB:
    """Class wrapper for watermark verification database operations."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        initialize_watermark_verification_db(self.db_path)

    def save(self, **kwargs) -> Optional[dict[str, Any]]:
        kwargs["db_path"] = self.db_path
        return save_verification_result(**kwargs)

    def get(self, verification_id: str) -> Optional[dict[str, Any]]:
        return get_verification_by_id(verification_id, db_path=self.db_path)

    def get_by_document(
        self, document_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        return get_verifications_for_document(
            document_id, limit=limit, offset=offset, db_path=self.db_path
        )

    def list_recent(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return list_recent_verifications(limit=limit, offset=offset, db_path=self.db_path)

    def count(self) -> int:
        return get_verification_count(db_path=self.db_path)

    def delete(self, verification_id: str) -> bool:
        return delete_verification(verification_id, db_path=self.db_path)
