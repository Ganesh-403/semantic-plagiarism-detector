from __future__ import annotations

import csv
import hashlib
import io
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.core.config import (normalize_score, normalize_severity_label,
                             severity_from_score)
from src.db.migrations import migrate_corpus_database

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus.db"
VALID_REVIEW_STATUSES = {"Pending", "Resolved"}
CSV_COLUMNS = [
    "Incident ID",
    "Document A",
    "Document B",
    "Similarity Score",
    "Threshold at Time of Flag",
    "Severity Rank",
    "Review Status",
    "Date Flagged",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalise_pair(doc_a: str, doc_b: str) -> tuple[str, str]:
    return tuple(sorted((str(doc_a).strip(), str(doc_b).strip())))  # type: ignore[return-value]


def _normalise_score(value: Any) -> float:
    try:
        return normalize_score(float(value))
    except (TypeError, ValueError):
        return 0.0


def _severity_rank(flag: Mapping[str, Any]) -> str:
    raw = str(flag.get("severity", "")).strip()
    if raw:
        try:
            return normalize_severity_label(raw)
        except ValueError:
            pass

    score = _normalise_score(flag.get("similarity", 0.0))
    return severity_from_score(score)


def build_incident_id(doc_a: str, doc_b: str) -> str:
    first, second = _normalise_pair(doc_a, doc_b)
    digest = hashlib.sha256(f"{first}\0{second}".encode("utf-8")).hexdigest()
    return f"INC-{digest[:12].upper()}"


def _get_connection(db_path: str | Path) -> sqlite3.Connection:
    abs_path = os.path.abspath(str(db_path))
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        conn = sqlite3.connect(abs_path)
    except (sqlite3.OperationalError, OSError, PermissionError):
        fallback_path = os.path.join(tempfile.gettempdir(), "semantic_plagiarism_detector", "data", os.path.basename(abs_path))
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        conn = sqlite3.connect(fallback_path)

    conn.execute("PRAGMA foreign_keys = ON")
    try:
        migrate_corpus_database(conn)
    except Exception:
        pass
    return conn


def init_incident_db(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Create or upgrade the shared corpus/incident database."""
    with closing(_get_connection(db_path)) as conn:
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            migrate_corpus_database(conn)
        except sqlite3.Error as exc:
            conn.rollback()
            raise sqlite3.Error(
                f"Failed to initialize incident database: {exc}"
            ) from exc


def _validate_incident(flag: Mapping[str, Any]) -> tuple[bool, str]:
    doc_a = str(flag.get("doc_a", "")).strip()
    doc_b = str(flag.get("doc_b", "")).strip()

    if not doc_a:
        return False, "Missing document A."

    if not doc_b:
        return False, "Missing document B."

    if doc_a == doc_b:
        return False, "Document identifiers must be different."

    try:
        similarity = float(flag.get("similarity", 0.0))
    except (TypeError, ValueError):
        return False, "Similarity score must be numeric."

    if not 0.0 <= similarity <= 1.0:
        return False, "Similarity score must be between 0.0 and 1.0."

    return True, ""


def _fetch_all_incidents(conn: sqlite3.Connection) -> list[MatchResult]:
    from src.db.schemas import MatchResult
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT pi.incident_id, pi.document_a, pi.document_b, pi.similarity_score,
               pi.severity_rank, pi.review_status, pi.date_flagged, pi.last_seen,
               pi.threshold_at_time_of_flag
        FROM plagiarism_incidents pi
        LEFT JOIN documents da ON pi.document_a = da.filename
        LEFT JOIN documents db ON pi.document_b = db.filename
        WHERE (da.is_deleted IS NULL OR da.is_deleted = 0)
          AND (db.is_deleted IS NULL OR db.is_deleted = 0)
        ORDER BY pi.date_flagged DESC, pi.incident_id ASC
        """
    ).fetchall()

    return [
        MatchResult(
            incident_id=row["incident_id"],
            document_a=row["document_a"],
            document_b=row["document_b"],
            similarity_score=row["similarity_score"],
            severity_rank=row["severity_rank"],
            review_status=row["review_status"],
            date_flagged=row["date_flagged"],
            last_seen=row["last_seen"],
            threshold_at_time_of_flag=row["threshold_at_time_of_flag"],
        )
        for row in rows
    ]


def sync_flagged_incidents(
    flags: Iterable[Mapping[str, Any]],
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    now: str | None = None,
    threshold: float | None = None,
) -> list[MatchResult]:
    from src.db.schemas import MatchResult
    init_incident_db(db_path)
    timestamp = now or _utc_now_iso()

    with closing(_get_connection(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        try:
            bulk_records = []
            for flag in flags:
                doc_a = str(flag.get("doc_a", "")).strip()
                doc_b = str(flag.get("doc_b", "")).strip()

                if not doc_a or not doc_b or doc_a == doc_b:
                    continue

                first, second = _normalise_pair(doc_a, doc_b)

                bulk_records.append((
                    build_incident_id(first, second),
                    first,
                    second,
                    _normalise_score(flag.get("similarity", 0.0)),
                    _severity_rank(flag),
                    timestamp,
                    timestamp,
                    _normalise_score(flag.get("threshold_at_time_of_flag", threshold or 0.0)),
                ))

            if bulk_records:
                conn.executemany(
                    """
                    INSERT INTO plagiarism_incidents (
                        incident_id, document_a, document_b,
                        similarity_score, severity_rank,
                        review_status, date_flagged, last_seen,
                        threshold_at_time_of_flag
                    )
                    VALUES (?, ?, ?, ?, ?, 'Pending', ?, ?, ?)
                    ON CONFLICT(incident_id) DO UPDATE SET
                        similarity_score = excluded.similarity_score,
                        severity_rank = excluded.severity_rank,
                        last_seen = excluded.last_seen
                    """,
                    bulk_records
                )
            conn.commit()

            rows = conn.execute(
                """
                SELECT pi.incident_id, pi.document_a, pi.document_b,
                       pi.similarity_score, pi.severity_rank,
                       pi.review_status, pi.date_flagged, pi.last_seen,
                       pi.threshold_at_time_of_flag
                FROM plagiarism_incidents pi
                LEFT JOIN documents da ON pi.document_a = da.filename
                LEFT JOIN documents db ON pi.document_b = db.filename
                WHERE (da.is_deleted IS NULL OR da.is_deleted = 0)
                  AND (db.is_deleted IS NULL OR db.is_deleted = 0)
                ORDER BY pi.date_flagged DESC, pi.incident_id ASC
                """
            ).fetchall()

            return [
                MatchResult(
                    incident_id=row["incident_id"],
                    document_a=row["document_a"],
                    document_b=row["document_b"],
                    similarity_score=row["similarity_score"],
                    severity_rank=row["severity_rank"],
                    review_status=row["review_status"],
                    date_flagged=row["date_flagged"],
                    last_seen=row["last_seen"],
                    threshold_at_time_of_flag=row["threshold_at_time_of_flag"],
                )
                for row in rows
            ]

        except sqlite3.Error as e:
            conn.rollback()
            raise sqlite3.Error(f"Failed to synchronize incidents: {e}") from e


def get_all_incidents(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[MatchResult]:
    init_incident_db(db_path)
    with closing(_get_connection(db_path)) as conn:
        return _fetch_all_incidents(conn)


def get_all_incidents_above_threshold_for_export(
    threshold: float,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[MatchResult]:
    from src.db.schemas import MatchResult
    init_incident_db(db_path)
    with closing(_get_connection(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT pi.document_a as doc_a, pi.document_b as doc_b,
                   pi.similarity_score as similarity,
                   pi.threshold_at_time_of_flag
            FROM plagiarism_incidents pi
            LEFT JOIN documents da ON pi.document_a = da.filename
            LEFT JOIN documents db ON pi.document_b = db.filename
            WHERE pi.similarity_score >= ?
              AND (da.is_deleted IS NULL OR da.is_deleted = 0)
              AND (db.is_deleted IS NULL OR db.is_deleted = 0)
            ORDER BY pi.similarity_score DESC
            """,
            (threshold,)
        ).fetchall()
        return [
            MatchResult(
                document_a=row["doc_a"],
                document_b=row["doc_b"],
                similarity_score=row["similarity"],
                threshold_at_time_of_flag=row["threshold_at_time_of_flag"],
            )
            for row in rows
        ]


def update_review_status(
    incident_id: str,
    review_status: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> bool:
    status = str(review_status).strip().title()

    if status not in VALID_REVIEW_STATUSES:
        raise ValueError(
            f"review_status must be one of {sorted(VALID_REVIEW_STATUSES)}"
        )

    init_incident_db(db_path)

    with closing(sqlite3.connect(str(db_path))) as conn:
        try:
            cursor = conn.execute(
                "UPDATE plagiarism_incidents SET review_status = ? WHERE incident_id = ?",
                (status, str(incident_id).strip()),
            )

            conn.commit()
            return cursor.rowcount > 0

        except sqlite3.Error as e:
            conn.rollback()
            raise sqlite3.Error(f"Failed to update review status: {e}") from e


def incidents_to_csv(incidents: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for incident in incidents:
        writer.writerow(
            {
                "Incident ID": incident.get("incident_id", ""),
                "Document A": incident.get("document_a", ""),
                "Document B": incident.get("document_b", ""),
                "Similarity Score": f"{_normalise_score(incident.get('similarity_score', 0.0)):.4f}",
                "Threshold at Time of Flag": f"{_normalise_score(incident.get('threshold_at_time_of_flag', 0.0)):.4f}",
                "Severity Rank": incident.get("severity_rank", ""),
                "Review Status": incident.get("review_status", "Pending"),
                "Date Flagged": incident.get("date_flagged", ""),
            }
        )
    return buffer.getvalue().encode("utf-8-sig")


def export_current_flags_csv(
    flags: Iterable[Mapping[str, Any]],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> bytes:
    sync_flagged_incidents(flags, db_path)
    return incidents_to_csv(get_all_incidents(db_path))


def get_high_severity_trends(
    days: int = 30,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    # Get daily count of High severity incidents over the specified number of days.
    # Returns list of dicts with 'date' and 'count' keys.
    init_incident_db(db_path)
    with closing(_get_connection(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                DATE(pi.date_flagged) as date,
                COUNT(*) as count
            FROM plagiarism_incidents pi
            LEFT JOIN documents da ON pi.document_a = da.filename
            LEFT JOIN documents db ON pi.document_b = db.filename
            WHERE pi.severity_rank = 'High'
                AND pi.date_flagged >= datetime('now', '-' || ? || ' days')
                AND (da.is_deleted IS NULL OR da.is_deleted = 0)
                AND (db.is_deleted IS NULL OR db.is_deleted = 0)
            GROUP BY DATE(pi.date_flagged)
            ORDER BY date ASC
            """,
            (days,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_most_plagiarized_documents(
    limit: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    # Get the most frequently plagiarized documents based on incident count.
    # Returns list of dicts with 'document_name' and 'incident_count' keys.
    init_incident_db(db_path)
    with closing(_get_connection(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                document_name,
                COUNT(*) as incident_count
            FROM (
                SELECT pi.document_a as document_name
                FROM plagiarism_incidents pi
                LEFT JOIN documents da ON pi.document_a = da.filename
                LEFT JOIN documents db ON pi.document_b = db.filename
                WHERE (da.is_deleted IS NULL OR da.is_deleted = 0)
                  AND (db.is_deleted IS NULL OR db.is_deleted = 0)
                UNION ALL
                SELECT pi.document_b as document_name
                FROM plagiarism_incidents pi
                LEFT JOIN documents da ON pi.document_a = da.filename
                LEFT JOIN documents db ON pi.document_b = db.filename
                WHERE (da.is_deleted IS NULL OR da.is_deleted = 0)
                  AND (db.is_deleted IS NULL OR db.is_deleted = 0)
            )
            GROUP BY document_name
            ORDER BY incident_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_false_positive(
    doc_a: str, doc_b: str, db_path: str | Path = DEFAULT_DB_PATH
) -> None:
    """Inserts a dismissed pair into the false_positives table."""
    init_incident_db(db_path)
    norm_a, norm_b = _normalise_pair(doc_a, doc_b)

    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO false_positives (document_a, document_b) VALUES (?, ?)",
            (norm_a, norm_b),
        )
        conn.commit()


def get_false_positives(db_path: str | Path = DEFAULT_DB_PATH) -> set[tuple[str, str]]:
    """Returns a set of all normalized dismissed pairs for fast filtering."""
    init_incident_db(db_path)
    with closing(sqlite3.connect(str(db_path))) as conn:
        rows = conn.execute(
            "SELECT document_a, document_b FROM false_positives"
        ).fetchall()
        return set((row[0], row[1]) for row in rows)
