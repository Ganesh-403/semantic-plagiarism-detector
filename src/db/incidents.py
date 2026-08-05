from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from src.core.app_config import CORPUS_DB_PATH, FALLBACK_DATA_DIR
from src.core.config import (
    normalize_score,
    normalize_severity_label,
    severity_from_score,
)
from src.db.common import with_sqlite_retry
from src.db.migrations import migrate_corpus_database
from src.db.schemas import MatchResult

# Seed the incidents default DB path from the centralized app_config.
# ``DEFAULT_DB_PATH`` is intentionally kept as a module-level constant so
# that callers/tests importing ``src.db.incidents.DEFAULT_DB_PATH`` continue
# to work (tests/conftest.py, tests/infrastructure/test_fixtures.py,
# app/components/incident_export.py, src/utils/daily_summary_email.py).
DEFAULT_DB_PATH = CORPUS_DB_PATH
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
    digest = hashlib.sha256(f"{first}0{second}".encode("utf-8")).hexdigest()
    return f"INC-{digest[:12].upper()}"


def _get_connection(db_path: str | Path) -> sqlite3.Connection:
    abs_path = os.path.abspath(str(db_path))
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        conn = sqlite3.connect(abs_path)
    except (sqlite3.OperationalError, OSError, PermissionError):
        # Centralized temp-dir fallback (matches corpus_db.py and
        # translation_cache.py so all three modules agree on the location).
        fallback_path = str(FALLBACK_DATA_DIR / os.path.basename(abs_path))
        os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
        conn = sqlite3.connect(fallback_path)

    conn.execute("PRAGMA foreign_keys = ON")
    try:
        migrate_corpus_database(conn)
    except Exception:
        pass
    return conn


def init_incident_db(
    db_path: str | Path | None = None,
) -> None:
    """Create or upgrade the shared corpus/incident database."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
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
    pass

def _validate_pagination(limit: int, offset: int) -> tuple[int, int]:
    """Validate and normalize SQL pagination arguments."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer.")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("offset must be an integer.")
    if limit <= 0:
        raise ValueError("limit must be greater than 0.")
    if offset < 0:
        raise ValueError("offset must be greater than or equal to 0.")
    return limit, offset


def _fetch_all_incidents(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
    offset: int = 0,
) -> list[MatchResult]:
    """Fetch one deterministic page of visible incidents."""
    from src.db.schemas import MatchResult

    safe_limit, safe_offset = _validate_pagination(limit, offset)
    conn.row_factory = sqlite3.Row
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
        LIMIT ? OFFSET ?
        """,
        (safe_limit, safe_offset),
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
            threshold_at_time_of_flag=row[
                "threshold_at_time_of_flag"
            ],
        )
        for row in rows
    ]


@with_sqlite_retry
def sync_flagged_incidents(
    flags: Iterable[Mapping[str, Any]],
    db_path: str | Path | None = None,
    *,
    now: str | None = None,
    threshold: float | None = None,
) -> list[MatchResult]:
    from src.db.schemas import MatchResult

    if db_path is None:
        db_path = DEFAULT_DB_PATH
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

                bulk_records.append(
                    (
                        build_incident_id(first, second),
                        first,
                        second,
                        _normalise_score(flag.get("similarity", 0.0)),
                        _severity_rank(flag),
                        timestamp,
                        timestamp,
                        _normalise_score(
                            flag.get("threshold_at_time_of_flag", threshold or 0.0)
                        ),
                    )
                )

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
                    bulk_records,
                )
                conn.commit()
                get_recent_incidents.cache_clear()

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
    db_path: str | Path | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    """Return one page of visible plagiarism incidents.

    Incidents are ordered by newest ``date_flagged`` first, with
    ``incident_id`` as a deterministic tie-breaker.

    Args:
        db_path: SQLite corpus database path.
        limit: Maximum number of incidents to return. Defaults to 50.
        offset: Number of matching incidents to skip. Defaults to 0.

    Raises:
        TypeError: If ``limit`` or ``offset`` is not an integer.
        ValueError: If ``limit`` is not positive or ``offset`` is negative.
    """
    safe_limit, safe_offset = _validate_pagination(limit, offset)
    init_incident_db(db_path)
    with closing(_get_connection(db_path)) as conn:
        return _fetch_all_incidents(
            conn,
            limit=safe_limit,
            offset=safe_offset,
        )


def get_total_incidents_count(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Return the number of visible plagiarism incidents.

    The count uses the same soft-deletion filters as
    :func:`get_all_incidents`, so pagination metadata cannot include
    incidents linked to deleted documents.
    """
    init_incident_db(db_path)
    with closing(_get_connection(db_path)) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM plagiarism_incidents pi
            LEFT JOIN documents da ON pi.document_a = da.filename
            LEFT JOIN documents db ON pi.document_b = db.filename
            WHERE (da.is_deleted IS NULL OR da.is_deleted = 0)
              AND (db.is_deleted IS NULL OR db.is_deleted = 0)
            """
        ).fetchone()
    return int(row[0]) if row is not None else 0


def get_incident_by_id(
    incident_id: str | int,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    """Fetch a single plagiarism incident record by its incident_id primary key.

    Args:
        incident_id: Integer or string primary key of the incident.
        db_path: Path to the SQLite corpus database.

    Returns:
        Dictionary containing incident record columns, or None if not found.
    """
    init_incident_db(db_path)
    with closing(_get_connection(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT pi.incident_id, pi.document_a, pi.document_b,
                   pi.similarity_score, pi.severity_rank,
                   pi.review_status, pi.date_flagged, pi.last_seen,
                   pi.threshold_at_time_of_flag
            FROM plagiarism_incidents pi
            LEFT JOIN documents da ON pi.document_a = da.filename
            LEFT JOIN documents db ON pi.document_b = db.filename
            WHERE (pi.incident_id = ? OR pi.incident_id = ?)
              AND (da.is_deleted IS NULL OR da.is_deleted = 0)
              AND (db.is_deleted IS NULL OR db.is_deleted = 0)
            """,
            (incident_id, str(incident_id)),
        ).fetchone()

        if row is None:
            return None
        return dict(row)


def get_incidents_by_severity(
    severity_level: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    init_incident_db(db_path)
    try:
        norm_severity = normalize_severity_label(severity_level)
    except ValueError:
        return []
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT incident_id, document_a, document_b,
                   similarity_score, severity_rank,
                   review_status, date_flagged, last_seen,
                   threshold_at_time_of_flag
            FROM plagiarism_incidents
            WHERE severity_rank = ?
            ORDER BY date_flagged DESC, incident_id ASC
            """,
(norm_severity,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_incidents_by_date_range(
    start_date: str,
    end_date: str,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    """Return incidents flagged between two explicit ISO dates (inclusive).

    Args:
        start_date: ISO-formatted start date/timestamp (inclusive).
        end_date: ISO-formatted end date/timestamp (inclusive).
        db_path: Path to the SQLite corpus database.

    Returns:
        A list of incident dicts ordered by ``date_flagged`` descending.
    """
    init_incident_db(db_path)
    with closing(_get_connection(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT incident_id, document_a, document_b,
                   similarity_score, severity_rank,
                   review_status, date_flagged, last_seen,
                   threshold_at_time_of_flag
            FROM plagiarism_incidents
            WHERE date_flagged >= ? AND date_flagged <= ?
            ORDER BY date_flagged DESC
            """,
            (start_date, end_date),
        ).fetchall()
        return [dict(row) for row in rows]


def get_all_incidents_above_threshold_for_export(    threshold: float,
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
            (threshold,),
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


@with_sqlite_retry
def update_review_status(
    incident_id: str,
    review_status: str,
    db_path: str | Path | None = None,
) -> bool:
    if db_path is None:
        db_path = DEFAULT_DB_PATH
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
    """Synchronize and export every visible incident, not only one page."""
    sync_flagged_incidents(flags, db_path)
    total = get_total_incidents_count(db_path)
    if total == 0:
        return incidents_to_csv([])
    return incidents_to_csv(
        get_all_incidents(
            db_path,
            limit=total,
            offset=0,
        )
    )


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


def get_incidents_count_by_date(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    """Get the daily count of all plagiarism incidents.

    Incidents are grouped by calendar date and returned in ascending date order.
    Incidents associated with deleted documents are excluded from the count.

    Returns:
        A list of dictionaries, where each dictionary contains a 'date' string
        and an integer 'count'. For example:
        [
            {"date": "2026-07-28", "count": 5}
        ]
    """
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
            WHERE (da.is_deleted IS NULL OR da.is_deleted = 0)
              AND (db.is_deleted IS NULL OR db.is_deleted = 0)
            GROUP BY DATE(pi.date_flagged)
            ORDER BY date ASC
            """
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


@with_sqlite_retry
def add_false_positive(
    doc_a: str, doc_b: str, db_path: str | Path | None = None
) -> None:
    if db_path is None:
        db_path = DEFAULT_DB_PATH
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


# ── Paginated query support ────────────────────────────────────────────────────


@dataclass(frozen=True)
class PaginatedIncidents:
    """Server-side paginated result for the incidents/warning list."""

    items: list[MatchResult]
    total_count: int
    page: int
    page_size: int
    total_pages: int


_ALLOWED_SORT_COLUMNS = frozenset(
    {
        "incident_id",
        "document_a",
        "document_b",
        "similarity_score",
        "severity_rank",
        "review_status",
        "date_flagged",
    }
)
_ALLOWED_SORT_ORDERS = frozenset({"ASC", "DESC"})

# ── SQL fragments reused by the paginated query ───────────────────────────────

_JOIN_DOCUMENTS = """
LEFT JOIN documents da ON pi.document_a = da.filename
LEFT JOIN documents db ON pi.document_b = db.filename
"""

_WHERE_NOT_DELETED = "(da.is_deleted IS NULL OR da.is_deleted = 0) AND (db.is_deleted IS NULL OR db.is_deleted = 0)"


def query_incidents_paginated(
    *,
    page: int = 1,
    page_size: int = 50,
    sort_by: str = "date_flagged",
    sort_order: str = "DESC",
    severity_filter: Optional[str] = None,
    search_query: str = "",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> PaginatedIncidents:
    """Query plagiarism incidents with server-side pagination, sorting, and filtering.

    Filters out incidents linked to soft-deleted documents (``is_deleted = 1``),
    consistent with the rest of the incidents API.

    Args:
        page: 1-indexed page number (default 1).
        page_size: Items per page (default 50, clamped 1–200).
        sort_by: Column to sort on (whitelisted against :data:`_ALLOWED_SORT_COLUMNS`).
        sort_order: ``"ASC"`` or ``"DESC"``.
        severity_filter: Optional severity level (``"Low"``, ``"Medium"``, ``"High"``).
        search_query: Filter by document name substring (matched against both doc_a and doc_b).
        db_path: Path to the SQLite corpus database.

    Returns:
        :class:`PaginatedIncidents` with the requested page of results.
    """
    init_incident_db(db_path)

    safe_page = max(1, int(page))
    safe_page_size = max(1, min(200, int(page_size)))
    offset = (safe_page - 1) * safe_page_size

    if sort_by not in _ALLOWED_SORT_COLUMNS:
        sort_by = "date_flagged"
    sort_order = sort_order.upper()
    if sort_order not in _ALLOWED_SORT_ORDERS:
        sort_order = "DESC"

    where_clauses: list[str] = [_WHERE_NOT_DELETED]
    params: list[Any] = []

    if severity_filter:
        sev = severity_filter.strip().title()
        if sev in {"Low", "Medium", "High"}:
            where_clauses.append("pi.severity_rank = ?")
            params.append(sev)

    query = search_query.strip()
    if query:
        where_clauses.append("(pi.document_a LIKE ? OR pi.document_b LIKE ?)")
        like_param = f"%{query}%"
        params.append(like_param)
        params.append(like_param)

    where_sql = "WHERE " + " AND ".join(where_clauses)

    with closing(_get_connection(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        # Total count
        count_row = conn.execute(
            f"SELECT COUNT(*) FROM plagiarism_incidents pi {_JOIN_DOCUMENTS} {where_sql}",
            params,
        ).fetchone()
        total_count = count_row[0] if count_row else 0
        total_pages = max(1, math.ceil(total_count / safe_page_size))

        # Paginated fetch
        order_sql = f"pi.{sort_by} {sort_order}, pi.incident_id ASC"
        rows = conn.execute(
            f"""
            SELECT pi.incident_id, pi.document_a, pi.document_b,
                   pi.similarity_score, pi.severity_rank,
                   pi.review_status, pi.date_flagged, pi.last_seen,
                   pi.threshold_at_time_of_flag
            FROM plagiarism_incidents pi
            {_JOIN_DOCUMENTS}
            {where_sql}
            ORDER BY {order_sql}
            LIMIT ? OFFSET ?
            """,
            [*params, safe_page_size, offset],
        ).fetchall()

        return PaginatedIncidents(
            items=[
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
            ],
            total_count=total_count,
            page=safe_page,
            page_size=safe_page_size,
            total_pages=total_pages,
        )


@with_sqlite_retry
def purge_old_incidents(
    days_old: int = 90,
    status: str = "Resolved",
    db_path: str | Path | None = None,
) -> int:
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    """
    Purge old plagiarism incidents that match a specific review status.

    This function helps maintain database performance and compliance by
    automatically removing resolved incidents that are older than a
    configurable number of days.

    Args:
        days_old: Number of days after which incidents should be purged.
        status: The review status of incidents to purge (default: 'Resolved').
        db_path: Path to the SQLite corpus database.

    Returns:
        int: The number of deleted records.
    """
    init_incident_db(db_path)
    with closing(_get_connection(db_path)) as conn:
        cursor = conn.execute(
            """
            DELETE FROM plagiarism_incidents
            WHERE review_status = ?
            AND date_flagged < datetime('now', '-' || ? || ' days')
            """,
            (status, days_old),
        )
conn.commit()
        return cursor.rowcount


def archive_old_incidents(
    cutoff_date: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Move incidents flagged before cutoff_date into incidents_archive.

    Copies matching rows from ``plagiarism_incidents`` into
    ``incidents_archive`` and removes them from ``plagiarism_incidents``,
    both inside a single transaction so the operation is atomic
    (issue #1492).

    Args:
        cutoff_date: ISO-8601 date/datetime string. Incidents with a
            ``date_flagged`` earlier than this value are archived.
        db_path: Path to the SQLite corpus database.

    Returns:
        int: The number of incident records archived.
    """
    init_incident_db(db_path)
    with closing(_get_connection(db_path)) as conn:
        try:
            conn.execute(
                """
                INSERT INTO incidents_archive (
                    incident_id, document_a, document_b,
                    similarity_score, severity_rank,
                    review_status, date_flagged, last_seen,
                    threshold_at_time_of_flag
                )
                SELECT incident_id, document_a, document_b,
                       similarity_score, severity_rank,
                       review_status, date_flagged, last_seen,
                       threshold_at_time_of_flag
                FROM plagiarism_incidents
                WHERE date_flagged < ?
                """,
                (cutoff_date,),
            )
            cursor = conn.execute(
                "DELETE FROM plagiarism_incidents WHERE date_flagged < ?",
                (cutoff_date,),
            )
            conn.commit()
            return cursor.rowcount
        except sqlite3.Error as exc:
            conn.rollback()
            raise sqlite3.Error(f"Failed to archive incidents: {exc}") from exc


@lru_cache(maxsize=128)
def get_recent_incidents(    limit: int = 5,
    db_path: str | Path | None = None,
) -> list[MatchResult]:
    """Fetch recent visible plagiarism incidents, cached for performance."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    return get_all_incidents(db_path=db_path, limit=limit, offset=0)


@with_sqlite_retry
def log_incident(
    flag: Mapping[str, Any],
    db_path: str | Path | None = None,
    *,
    now: str | None = None,
    threshold: float | None = None,
) -> MatchResult:
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    """Log a single plagiarism incident and clear get_recent_incidents cache.

    Args:
        flag: Dict with 'doc_a', 'doc_b', 'similarity', and optionally 'severity'.
        db_path: Path to the SQLite corpus database.

    Returns:
        The created MatchResult.
    """
    results = sync_flagged_incidents([flag], db_path, now=now, threshold=threshold)
    if not results:
        raise ValueError("Failed to log incident: Invalid input.")
    get_recent_incidents.cache_clear()
    doc_a = str(flag.get("doc_a", "")).strip()
    doc_b = str(flag.get("doc_b", "")).strip()
    first, second = _normalise_pair(doc_a, doc_b)
    target_id = build_incident_id(first, second)
    for res in results:
        if res.incident_id == target_id:
            return res
    return results[0]
