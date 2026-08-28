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

import csv
import io
from datetime import datetime, timedelta, timezone

import pytest

from src.db.incidents import (
    archive_old_incidents,
    build_incident_id,
    export_current_flags_csv,
    get_all_incidents,
    get_incident_by_id,
    get_incidents_by_assignment,
    get_incidents_by_date_range,
    get_incidents_by_severity,
    get_incidents_by_user,
    get_incidents_count_by_date,
    get_recent_incidents,
    incidents_to_csv,
    log_incident,
    purge_old_incidents,
    sync_flagged_incidents,
    update_review_status,
)


@pytest.fixture(autouse=True)
def test_db(mock_db):
    # Backward compatibility for tests expecting test_db fixture returning the path
    return mock_db


def test_build_incident_id_is_deterministic():
    id1 = build_incident_id("doc1.pdf", "doc2.pdf")
    id2 = build_incident_id("doc1.pdf", "doc2.pdf")

    assert id1 == id2
    assert id1.startswith("INC-")


def test_build_incident_id_same_pair_different_order():
    id1 = build_incident_id("doc1.pdf", "doc2.pdf")
    id2 = build_incident_id("doc2.pdf", "doc1.pdf")

    assert id1 == id2


def test_build_incident_id_no_collision_between_distinct_pairs():
    """Regression test: build_incident_id() previously joined the two
    normalised filenames with a bare "0" digit before hashing
    (f"{first}0{second}"). Because filenames can themselves contain
    digits, two genuinely different document pairs could produce the
    exact same hash input string once joined, e.g.:

        normalised pair A: ("0doc20", "doc20") -> old input "0doc200doc20"
        normalised pair B: ("0doc2", "0doc20")  -> old input "0doc200doc20"

    (Note: the pair given in the issue description, ("doc10", "doc2") vs
    ("doc1", "0doc2"), does NOT actually collide once you account for
    _normalise_pair()'s alphabetical sorting of the two filenames before
    hashing — the pair above is a real, reproducing collision under the
    old "0" separator, verified directly against the sorted output.)

    The fix joins with "||" instead, which is not a valid character in
    either filename here, so the two pairs must now hash differently.
    """
    id1 = build_incident_id("0doc20", "doc20")
    id2 = build_incident_id("0doc2", "0doc20")

    assert id1 != id2


def test_build_incident_id_no_collision_for_issue_example_pair():
    """As requested in the issue: verify no collision between
    ("doc10", "doc2") and ("doc1", "0doc2")."""
    id1 = build_incident_id("doc10", "doc2")
    id2 = build_incident_id("doc1", "0doc2")

    assert id1 != id2


def test_get_incidents_by_date_range_filters_correctly(test_db):
    sync_flagged_incidents(
        [{"doc_a": "old1.pdf", "doc_b": "old2.pdf", "similarity": 0.9}],
        test_db,
        now="2026-01-01T00:00:00+00:00",
    )
    sync_flagged_incidents(
        [{"doc_a": "mid1.pdf", "doc_b": "mid2.pdf", "similarity": 0.9}],
        test_db,
        now="2026-03-15T00:00:00+00:00",
    )
    sync_flagged_incidents(
        [{"doc_a": "new1.pdf", "doc_b": "new2.pdf", "similarity": 0.9}],
        test_db,
        now="2026-06-01T00:00:00+00:00",
    )

    results = get_incidents_by_date_range(
        "2026-02-01T00:00:00+00:00", "2026-04-01T00:00:00+00:00"
    )

    assert len(results) == 1
    assert results[0]["document_a"] == "mid1.pdf"


def test_get_incidents_by_date_range_orders_descending(test_db):
    sync_flagged_incidents(
        [{"doc_a": "a1.pdf", "doc_b": "a2.pdf", "similarity": 0.9}],
        test_db,
        now="2026-01-01T00:00:00+00:00",
    )
    sync_flagged_incidents(
        [{"doc_a": "b1.pdf", "doc_b": "b2.pdf", "similarity": 0.9}],
        test_db,
        now="2026-01-05T00:00:00+00:00",
    )

    results = get_incidents_by_date_range(
        "2026-01-01T00:00:00+00:00", "2026-01-10T00:00:00+00:00"
    )

    assert [r["document_a"] for r in results] == ["b1.pdf", "a1.pdf"]


def test_sync_flagged_incidents_adds_incident(test_db):
    flags = [
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc2.pdf",
            "similarity": 0.95,
        }
    ]

    incidents = sync_flagged_incidents(flags, test_db)

    assert len(incidents) == 1
    assert incidents[0]["review_status"] == "Pending"
    assert incidents[0]["severity_rank"] == "High"


def test_sync_flagged_incidents_ignores_duplicate_pairs(test_db):
    flags = [
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc2.pdf",
            "similarity": 0.91,
        }
    ]

    sync_flagged_incidents(flags, test_db)
    sync_flagged_incidents(flags, test_db)

    incidents = get_all_incidents(test_db)

    assert len(incidents) == 1


def test_sync_flagged_incidents_bulk_upsert(test_db):
    flags = [
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc2.pdf",
            "similarity": 0.50,
            "severity": "Medium",
        }
    ]

    sync_flagged_incidents(flags, test_db)
    incidents = get_all_incidents(test_db)
    assert len(incidents) == 1
    assert incidents[0]["similarity_score"] == 0.50
    assert incidents[0]["severity_rank"] == "Medium"

    flags[0]["similarity"] = 0.99
    flags[0]["severity"] = "Critical"
    sync_flagged_incidents(flags, test_db)
    incidents = get_all_incidents(test_db)
    assert len(incidents) == 1
    assert incidents[0]["similarity_score"] == 0.99
    assert incidents[0]["severity_rank"] == "Critical"

    flags[0]["similarity"] = 0.50
    flags[0]["severity"] = "Low"
    sync_flagged_incidents(flags, test_db)
    incidents = get_all_incidents(test_db)
    assert len(incidents) == 1
    assert incidents[0]["similarity_score"] == 0.50
    assert incidents[0]["severity_rank"] == "Low"


def test_sync_flagged_incidents_handles_invalid_similarity(test_db):
    flags = [
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc2.pdf",
            "similarity": 5,
        }
    ]

    incidents = sync_flagged_incidents(flags, test_db)

    assert incidents[0]["similarity_score"] == 1.0


def test_sync_flagged_incidents_empty_input(test_db):
    incidents = sync_flagged_incidents([], test_db)

    assert incidents == []


def test_get_all_incidents_returns_all(test_db):
    flags = [
        {
            "doc_a": "a.pdf",
            "doc_b": "b.pdf",
            "similarity": 0.9,
        },
        {
            "doc_a": "c.pdf",
            "doc_b": "d.pdf",
            "similarity": 0.7,
        },
    ]

    sync_flagged_incidents(flags, test_db)

    from src.db.schemas import MatchResult

    incidents = get_all_incidents(test_db)

    assert len(incidents) == 2
    assert all(isinstance(inc, MatchResult) for inc in incidents)
    assert incidents[0].document_a == "a.pdf"
    assert incidents[0]["document_a"] == "a.pdf"


def test_update_review_status_success(test_db):
    flags = [
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc2.pdf",
            "similarity": 0.9,
        }
    ]

    incidents = sync_flagged_incidents(flags, test_db)

    incident_id = incidents[0]["incident_id"]

    result = update_review_status(
        incident_id,
        "Resolved",
        test_db,
    )

    assert result is True

    updated = get_all_incidents(test_db)

    assert updated[0]["review_status"] == "Resolved"


def test_update_review_status_invalid_status(test_db):
    with pytest.raises(ValueError, match="Invalid review status: Done. Must be one of"):
        update_review_status(
            "INC-123456",
            "Done",
            test_db,
        )


def test_update_review_status_dismissed(test_db):
    flags = [
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc2.pdf",
            "similarity": 0.9,
        }
    ]
    incidents = sync_flagged_incidents(flags, test_db)
    incident_id = incidents[0]["incident_id"]

    result = update_review_status(
        incident_id,
        "Dismissed",
        test_db,
    )
    assert result is True

    updated = get_all_incidents(test_db)
    assert updated[0]["review_status"] == "Dismissed"


def test_bulk_update_incident_status_validation(test_db):
    from src.db.incidents import bulk_update_incident_status

    flags = [
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc2.pdf",
            "similarity": 0.9,
        }
    ]
    incidents = sync_flagged_incidents(flags, test_db)
    incident_id = incidents[0]["incident_id"]

    # 1. Invalid status check
    with pytest.raises(
        ValueError, match="Invalid review status: InvalidState. Must be one of"
    ):
        bulk_update_incident_status([incident_id], "InvalidState", test_db)

    # 2. Valid status update to Resolved
    count = bulk_update_incident_status([incident_id], "Resolved", test_db)
    assert count == 1
    updated = get_all_incidents(test_db)
    assert updated[0]["review_status"] == "Resolved"

    # 3. Valid status update to Dismissed
    count = bulk_update_incident_status([incident_id], "Dismissed", test_db)
    assert count == 1
    updated = get_all_incidents(test_db)
    assert updated[0]["review_status"] == "Dismissed"


def test_update_review_status_unknown_incident(test_db):
    result = update_review_status(
        "INC-UNKNOWN",
        "Resolved",
        test_db,
    )

    assert result is False


def test_incidents_to_csv_generates_valid_csv():
    rows = [
        {
            "incident_id": "INC-ABC123",
            "document_a": "a.pdf",
            "document_b": "b.pdf",
            "similarity_score": 0.95,
            "severity_rank": "High",
            "review_status": "Pending",
            "date_flagged": "2026-01-01T00:00:00Z",
        }
    ]

    csv_bytes = incidents_to_csv(rows)

    text = csv_bytes.decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(text))

    records = list(reader)

    assert len(records) == 1
    assert records[0]["Incident ID"] == "INC-ABC123"
    assert records[0]["Severity Rank"] == "High"


def test_incidents_to_csv_empty_input():
    csv_bytes = incidents_to_csv([])

    text = csv_bytes.decode("utf-8-sig")

    assert "Incident ID" in text


def test_incidents_to_csv_escapes_special_characters():
    """Verify filenames with commas, quotes, and newlines are RFC 4180 compliant."""
    rows = [
        {
            "incident_id": "INC-001",
            "document_a": "thesis, final (v2).pdf",
            "document_b": 'report "draft".docx',
            "similarity_score": 0.88,
            "severity_rank": "High",
            "review_status": "Pending",
            "date_flagged": "2026-01-01T00:00:00Z",
        },
        {
            "incident_id": "INC-002",
            "document_a": "line\nbreak.pdf",
            "document_b": "normal.pdf",
            "similarity_score": 0.75,
            "severity_rank": "Medium",
            "review_status": "Reviewed",
            "date_flagged": "2026-01-02T00:00:00Z",
        },
    ]

    csv_bytes = incidents_to_csv(rows)
    text = csv_bytes.decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(text))
    records = list(reader)

    assert len(records) == 2
    assert records[0]["Document A"] == "thesis, final (v2).pdf"
    assert records[0]["Document B"] == 'report "draft".docx'
    assert records[1]["Document A"] == "line\nbreak.pdf"


def test_export_current_flags_csv_exports_incidents(test_db):
    flags = [
        {
            "doc_a": "doc1.pdf",
            "doc_b": "doc2.pdf",
            "similarity": 0.94,
        }
    ]

    csv_bytes = export_current_flags_csv(flags, test_db)

    text = csv_bytes.decode("utf-8-sig")

    assert "doc1.pdf" in text
    assert "doc2.pdf" in text


def test_get_incidents_by_severity(test_db):
    """Verify incidents can be filtered by severity."""
    flags = [
        {
            "doc_a": "high_doc1.pdf",
            "doc_b": "high_doc2.pdf",
            "similarity": 0.95,
        },
        {
            "doc_a": "low_doc1.pdf",
            "doc_b": "low_doc2.pdf",
            "similarity": 0.20,
        },
    ]

    sync_flagged_incidents(flags, test_db)

    results = get_incidents_by_severity("High", test_db)

    assert len(results) == 1
    assert results[0]["severity_rank"] == "High"
    assert results[0]["document_a"] == "high_doc1.pdf"


def test_get_incidents_by_severity_orders_by_timestamp_desc(test_db):
    """Verify same-severity incidents are returned newest first."""
    flags = [
        {
            "doc_a": "high_doc_older_a.pdf",
            "doc_b": "high_doc_older_b.pdf",
            "similarity": 0.91,
        },
    ]
    sync_flagged_incidents(flags, test_db, now="2024-01-01T00:00:00+00:00")

    flags_newer = [
        {
            "doc_a": "high_doc_newer_a.pdf",
            "doc_b": "high_doc_newer_b.pdf",
            "similarity": 0.92,
        },
    ]
    sync_flagged_incidents(flags_newer, test_db, now="2024-06-01T00:00:00+00:00")

    results = get_incidents_by_severity("High", test_db)

    assert len(results) == 2
    assert results[0]["document_a"] == "high_doc_newer_a.pdf"
    assert results[1]["document_a"] == "high_doc_older_a.pdf"


def test_purge_old_incidents_deletes_resolved_older_than_days(test_db):
    """Test that purge_old_incidents deletes resolved incidents older than specified days."""
    # Create a recent resolved incident
    recent_flags = [
        {"doc_a": "recent1.pdf", "doc_b": "recent2.pdf", "similarity": 0.90}
    ]
    sync_flagged_incidents(recent_flags, test_db)

    # Manually update to Resolved and set an old date (100 days ago)
    import sqlite3

    old_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    with sqlite3.connect(test_db) as conn:
        conn.execute(
            "UPDATE plagiarism_incidents SET review_status = 'Resolved', date_flagged = ? WHERE document_a = 'recent1.pdf'",
            (old_date,),
        )
        conn.commit()

    # Create a pending incident with an old date (should NOT be deleted)
    with sqlite3.connect(test_db) as conn:
        conn.execute(
            "INSERT INTO plagiarism_incidents (incident_id, document_a, document_b, similarity_score, severity_rank, review_status, date_flagged, last_seen, threshold_at_time_of_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "INC-OLDPENDING",
                "old_pending1.pdf",
                "old_pending2.pdf",
                0.85,
                "Medium",
                "Pending",
                old_date,
                old_date,
                0.59,
            ),
        )
        conn.commit()

    # Create a recent resolved incident (should NOT be deleted)
    recent_resolved_flags = [
        {"doc_a": "recent_res1.pdf", "doc_b": "recent_res2.pdf", "similarity": 0.92}
    ]
    sync_flagged_incidents(recent_resolved_flags, test_db)
    with sqlite3.connect(test_db) as conn:
        conn.execute(
            "UPDATE plagiarism_incidents SET review_status = 'Resolved' WHERE document_a = 'recent_res1.pdf'"
        )
        conn.commit()

    # Verify initial count
    assert len(get_all_incidents(test_db)) == 3

    # Purge old resolved incidents (90 days)
    deleted_count = purge_old_incidents(days_old=90, status="Resolved", db_path=test_db)

    assert deleted_count == 1

    # Verify only the old pending and recent resolved remain
    remaining = get_all_incidents(test_db)
    assert len(remaining) == 2
    remaining_docs = {inc["document_a"] for inc in remaining}
    assert "old_pending1.pdf" in remaining_docs
    assert "recent_res1.pdf" in remaining_docs
    assert "recent1.pdf" not in remaining_docs


def test_get_incident_by_id_found(test_db):
    flags = [
        {
            "doc_a": "file1.pdf",
            "doc_b": "file2.pdf",
            "similarity": 0.88,
        }
    ]
    incidents = sync_flagged_incidents(flags, test_db)
    target_id = incidents[0]["incident_id"]

    result = get_incident_by_id(target_id, test_db)

    assert result is not None
    assert isinstance(result, dict)
    assert result["incident_id"] == target_id
    assert result["document_a"] == "file1.pdf"
    assert result["document_b"] == "file2.pdf"
    assert result["similarity_score"] == 0.88
    assert result["severity_rank"] == "Medium"
    assert result["review_status"] == "Pending"


def test_get_incident_by_id_not_found(test_db):
    result = get_incident_by_id("INC-NONEXISTENT", test_db)
    assert result is None


def test_get_incident_by_id_integer(test_db):
    import sqlite3

    with sqlite3.connect(test_db) as conn:
        conn.execute(
            """
            INSERT INTO plagiarism_incidents (
                incident_id, document_a, document_b, similarity_score,
                severity_rank, review_status, date_flagged, last_seen,
                threshold_at_time_of_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1046,
                "int_doc_a.pdf",
                "int_doc_b.pdf",
                0.75,
                "Medium",
                "Pending",
                "2026-07-31T00:00:00Z",
                "2026-07-31T00:00:00Z",
                0.50,
            ),
        )
        conn.commit()

    result = get_incident_by_id(1046, test_db)
    assert result is not None
    assert isinstance(result, dict)
    assert result["document_a"] == "int_doc_a.pdf"
    assert result["document_b"] == "int_doc_b.pdf"


def test_get_incidents_count_by_date(test_db):
    """Verify daily counts of incidents are aggregated correctly."""
    import sqlite3

    # Insert mock incidents across multiple dates
    incidents_data = [
        (
            "INC-1",
            "docA.pdf",
            "docB.pdf",
            0.90,
            "High",
            "Pending",
            "2026-08-01T10:00:00Z",
            "2026-08-01T10:00:00Z",
            0.50,
        ),
        (
            "INC-2",
            "docC.pdf",
            "docD.pdf",
            0.85,
            "Medium",
            "Pending",
            "2026-08-01T15:30:00Z",
            "2026-08-01T15:30:00Z",
            0.50,
        ),
        (
            "INC-3",
            "docE.pdf",
            "docF.pdf",
            0.70,
            "Medium",
            "Pending",
            "2026-08-02T08:00:00Z",
            "2026-08-02T08:00:00Z",
            0.50,
        ),
        (
            "INC-4",
            "docG.pdf",
            "docH.pdf",
            0.95,
            "High",
            "Pending",
            "2026-08-05T12:00:00Z",
            "2026-08-05T12:00:00Z",
            0.50,
        ),
    ]

    with sqlite3.connect(test_db) as conn:
        conn.executemany(
            """
            INSERT INTO plagiarism_incidents (
                incident_id, document_a, document_b, similarity_score,
                severity_rank, review_status, date_flagged, last_seen,
                threshold_at_time_of_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            incidents_data,
        )
        conn.commit()

    results = get_incidents_count_by_date(test_db)

    # We expect:
    # 2026-08-01: 2 counts
    # 2026-08-02: 1 count
    # 2026-08-05: 1 count
    expected = [
        {"date": "2026-08-01", "count": 2},
        {"date": "2026-08-02", "count": 1},
        {"date": "2026-08-05", "count": 1},
    ]
    assert results == expected


def test_get_recent_incidents_caching_and_invalidation(test_db):
    """Verify that get_recent_incidents caches queries and log_incident/sync_flagged_incidents invalidate it."""
    from unittest.mock import patch

    # 1. Clear any prior cache state
    get_recent_incidents.cache_clear()

    # 2. Insert initial incidents via sync
    flags = [{"doc_a": "doc1.pdf", "doc_b": "doc2.pdf", "similarity": 0.85}]
    sync_flagged_incidents(flags, test_db)

    # 3. Call get_recent_incidents for the first time (should hit DB)
    incidents_1 = get_recent_incidents(limit=5, db_path=test_db)
    assert len(incidents_1) == 1

    # 4. Repeated call with same args should hit cache
    with patch("src.db.incidents._get_connection") as mock_conn:
        incidents_2 = get_recent_incidents(limit=5, db_path=test_db)
        assert len(incidents_2) == 1
        # Since cache is hit, database connection shouldn't be opened
        mock_conn.assert_not_called()

    # 5. Log a new incident using log_incident, which should invalidate the cache
    new_flag = {"doc_a": "doc3.pdf", "doc_b": "doc4.pdf", "similarity": 0.92}
    logged = log_incident(new_flag, test_db)
    assert logged.document_a == "doc3.pdf"

    # 6. Call get_recent_incidents again (should hit DB because cache was invalidated)
    incidents_3 = get_recent_incidents(limit=5, db_path=test_db)
    assert len(incidents_3) == 2


def test_get_recent_incidents_with_cutoff_time(test_db):
    """Test get_recent_incidents filters incidents by date_flagged cutoff_time in SQL."""
    get_recent_incidents.cache_clear()

    now_dt = datetime.now(timezone.utc)
    recent_date = (now_dt - timedelta(hours=2)).isoformat()
    old_date = (now_dt - timedelta(hours=30)).isoformat()

    recent_flag = [{"doc_a": "rec1.pdf", "doc_b": "rec2.pdf", "similarity": 0.85}]
    sync_flagged_incidents(recent_flag, test_db, now=recent_date)

    old_flag = [{"doc_a": "old1.pdf", "doc_b": "old2.pdf", "similarity": 0.90}]
    sync_flagged_incidents(old_flag, test_db, now=old_date)

    cutoff = (now_dt - timedelta(hours=24)).isoformat()
    filtered = get_recent_incidents(cutoff_time=cutoff, db_path=test_db)

    assert len(filtered) == 1
    assert filtered[0]["document_a"] == "rec1.pdf"


def test_archive_old_incidents_moves_rows_to_archive_table(test_db):
    """Test that archive_old_incidents copies old rows and removes them."""
    import sqlite3

    old_date = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    recent_date = datetime.now(timezone.utc).isoformat()

    old_flags = [{"doc_a": "old1.pdf", "doc_b": "old2.pdf", "similarity": 0.90}]
    sync_flagged_incidents(old_flags, test_db, now=old_date)

    recent_flags = [{"doc_a": "new1.pdf", "doc_b": "new2.pdf", "similarity": 0.80}]
    sync_flagged_incidents(recent_flags, test_db, now=recent_date)

    assert len(get_all_incidents(test_db)) == 2

    cutoff = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    archived_count = archive_old_incidents(cutoff, db_path=test_db)

    assert archived_count == 1

    remaining = get_all_incidents(test_db)
    assert len(remaining) == 1
    assert remaining[0]["document_a"] == "new1.pdf"

    with sqlite3.connect(test_db) as conn:
        row = conn.execute(
            "SELECT document_a FROM incidents_archive WHERE incident_id = ?",
            (build_incident_id("old1.pdf", "old2.pdf"),),
        ).fetchone()
        assert row is not None
        assert row[0] == "old1.pdf"


def test_get_incidents_by_assignment(test_db):
    """Verify that get_incidents_by_assignment filters incidents by assignment title."""
    import sqlite3

    # 1. Add mock documents with matching and non-matching assignment titles
    with sqlite3.connect(test_db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (filename, file_hash, upload_date, assignment_title) VALUES (?, ?, ?, ?)",
            ("doc1.pdf", "hash1", "2026-01-01T00:00:00Z", "CS 101 Homework 1"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO documents (filename, file_hash, upload_date, assignment_title) VALUES (?, ?, ?, ?)",
            ("doc2.pdf", "hash2", "2026-01-01T00:00:00Z", "CS 101 Homework 1"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO documents (filename, file_hash, upload_date, assignment_title) VALUES (?, ?, ?, ?)",
            ("doc3.pdf", "hash3", "2026-01-01T00:00:00Z", "CS 101 Homework 2"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO documents (filename, file_hash, upload_date, assignment_title) VALUES (?, ?, ?, ?)",
            ("doc4.pdf", "hash4", "2026-01-01T00:00:00Z", "CS 101 Homework 2"),
        )
        conn.commit()

    # 2. Sync mock incidents
    # Incident matching assignment "CS 101 Homework 1"
    sync_flagged_incidents(
        [{"doc_a": "doc1.pdf", "doc_b": "doc2.pdf", "similarity": 0.85}],
        test_db,
    )
    # Incident matching assignment "CS 101 Homework 2"
    sync_flagged_incidents(
        [{"doc_a": "doc3.pdf", "doc_b": "doc4.pdf", "similarity": 0.90}],
        test_db,
    )

    # 3. Query matching "CS 101 Homework 1"
    results_hw1 = get_incidents_by_assignment("CS 101 Homework 1", db_path=test_db)
    assert len(results_hw1) == 1
    assert results_hw1[0]["document_a"] == "doc1.pdf"
    assert results_hw1[0]["document_b"] == "doc2.pdf"

    # 4. Query matching "CS 101 Homework 2"
    results_hw2 = get_incidents_by_assignment("CS 101 Homework 2", db_path=test_db)
    assert len(results_hw2) == 1
    assert results_hw2[0]["document_a"] == "doc3.pdf"
    assert results_hw2[0]["document_b"] == "doc4.pdf"

    # 5. Query non-existent assignment
    results_none = get_incidents_by_assignment("CS 101 Homework 3", db_path=test_db)
    assert len(results_none) == 0


def test_get_incidents_by_assignment_direct_table(tmp_path):
    """Verify get_incidents_by_assignment queries 'incidents' table directly when it exists."""
    import sqlite3

    db_file = tmp_path / "custom_incidents.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            CREATE TABLE incidents (
                id INTEGER PRIMARY KEY,
                assignment_title TEXT,
                timestamp TEXT,
                details TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO incidents (assignment_title, timestamp, details) VALUES (?, ?, ?)",
            ("Essay 1", "2026-05-01T10:00:00Z", "Incident A"),
        )
        conn.execute(
            "INSERT INTO incidents (assignment_title, timestamp, details) VALUES (?, ?, ?)",
            ("Essay 1", "2026-05-02T10:00:00Z", "Incident B"),
        )
        conn.execute(
            "INSERT INTO incidents (assignment_title, timestamp, details) VALUES (?, ?, ?)",
            ("Essay 2", "2026-05-03T10:00:00Z", "Incident C"),
        )
        conn.commit()

    res = get_incidents_by_assignment("Essay 1", db_path=db_file)
    assert len(res) == 2
    assert res[0]["details"] == "Incident B"
    assert res[1]["details"] == "Incident A"


# ── Issue #1765: get_incidents_by_user() ─────────────────────────────────────


def test_get_incidents_by_user_returns_empty_for_empty_username(test_db):
    """Empty / whitespace-only username should short-circuit to [] without
    touching the database (prevents accidental owner='' matches)."""
    assert get_incidents_by_user("") == []
    assert get_incidents_by_user("   ") == []
    assert get_incidents_by_user(None) == []  # type: ignore[arg-type]


def test_get_incidents_by_user_returns_empty_when_no_incidents(test_db):
    """A fresh database with no incidents should return an empty list."""
    assert get_incidents_by_user("alice", db_path=test_db) == []


def test_get_incidents_by_user_filters_by_owner(test_db):
    """Canonical schema path: incidents are filtered via documents.owner."""
    import sqlite3

    # Seed documents with different owners.
    with sqlite3.connect(test_db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (filename, file_hash, upload_date, owner) "
            "VALUES (?, ?, ?, ?)",
            ("alice_doc1.pdf", "h1", "2026-01-01T00:00:00Z", "alice"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO documents (filename, file_hash, upload_date, owner) "
            "VALUES (?, ?, ?, ?)",
            ("alice_doc2.pdf", "h2", "2026-01-01T00:00:00Z", "alice"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO documents (filename, file_hash, upload_date, owner) "
            "VALUES (?, ?, ?, ?)",
            ("bob_doc1.pdf", "h3", "2026-01-01T00:00:00Z", "bob"),
        )
        conn.commit()

    # Sync two incidents: one alice-vs-alice, one alice-vs-bob, one bob-vs-bob.
    sync_flagged_incidents(
        [{"doc_a": "alice_doc1.pdf", "doc_b": "alice_doc2.pdf", "similarity": 0.9}],
        test_db,
        now="2026-01-02T00:00:00Z",
    )
    sync_flagged_incidents(
        [{"doc_a": "alice_doc1.pdf", "doc_b": "bob_doc1.pdf", "similarity": 0.8}],
        test_db,
        now="2026-01-03T00:00:00Z",
    )
    sync_flagged_incidents(
        [
            {"doc_a": "bob_doc1.pdf", "doc_b": "bob_doc1.pdf", "similarity": 0.7}
        ],  # skipped (same doc)
        test_db,
    )

    # Alice should see both incidents that touch her documents.
    alice_results = get_incidents_by_user("alice", db_path=test_db)
    assert len(alice_results) == 2
    # Newest first.
    assert alice_results[0]["document_a"] == "alice_doc1.pdf"
    assert alice_results[0]["document_b"] == "bob_doc1.pdf"
    # owner columns should be present (added by the JOIN).
    assert "owner_a" in alice_results[0]
    assert "owner_b" in alice_results[0]

    # Bob should see only the cross-incident.
    bob_results = get_incidents_by_user("bob", db_path=test_db)
    assert len(bob_results) == 1
    assert bob_results[0]["document_a"] == "alice_doc1.pdf"
    assert bob_results[0]["document_b"] == "bob_doc1.pdf"

    # Unknown user → empty.
    assert get_incidents_by_user("charlie", db_path=test_db) == []


def test_get_incidents_by_user_orders_descending_by_date_flagged(test_db):
    """Newest incidents must come first (ORDER BY date_flagged DESC)."""
    import sqlite3

    with sqlite3.connect(test_db) as conn:
        for i in range(3):
            conn.execute(
                "INSERT OR REPLACE INTO documents (filename, file_hash, upload_date, owner) "
                "VALUES (?, ?, ?, ?)",
                (f"u{i}_a.pdf", f"hash_a_{i}", "2026-01-01T00:00:00Z", "u1"),
            )
            conn.execute(
                "INSERT OR REPLACE INTO documents (filename, file_hash, upload_date, owner) "
                "VALUES (?, ?, ?, ?)",
                (f"u{i}_b.pdf", f"hash_b_{i}", "2026-01-01T00:00:00Z", "u1"),
            )
        conn.commit()

    sync_flagged_incidents(
        [{"doc_a": "u0_a.pdf", "doc_b": "u0_b.pdf", "similarity": 0.9}],
        test_db,
        now="2026-01-01T00:00:00Z",
    )
    sync_flagged_incidents(
        [{"doc_a": "u1_a.pdf", "doc_b": "u1_b.pdf", "similarity": 0.9}],
        test_db,
        now="2026-06-01T00:00:00Z",
    )
    sync_flagged_incidents(
        [{"doc_a": "u2_a.pdf", "doc_b": "u2_b.pdf", "similarity": 0.9}],
        test_db,
        now="2026-03-01T00:00:00Z",
    )

    results = get_incidents_by_user("u1", db_path=test_db)
    assert len(results) == 3
    dates = [r["date_flagged"] for r in results]
    assert dates == sorted(dates, reverse=True)
    assert results[0]["document_a"] == "u1_a.pdf"
    assert results[-1]["document_a"] == "u0_a.pdf"


def test_get_incidents_by_user_strips_whitespace_in_username(test_db):
    """Leading/trailing whitespace in the username should be stripped."""
    import sqlite3

    with sqlite3.connect(test_db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (filename, file_hash, upload_date, owner) "
            "VALUES (?, ?, ?, ?)",
            ("a.pdf", "h1", "2026-01-01T00:00:00Z", "alice"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO documents (filename, file_hash, upload_date, owner) "
            "VALUES (?, ?, ?, ?)",
            ("b.pdf", "h2", "2026-01-01T00:00:00Z", "alice"),
        )
        conn.commit()

    sync_flagged_incidents(
        [{"doc_a": "a.pdf", "doc_b": "b.pdf", "similarity": 0.9}],
        test_db,
    )

    assert len(get_incidents_by_user("  alice  ", db_path=test_db)) == 1
    assert len(get_incidents_by_user("\talice\n", db_path=test_db)) == 1


def test_get_incidents_by_user_legacy_incidents_table(tmp_path):
    """Legacy schema path: when a standalone `incidents` table with `owner`
    and `timestamp` columns exists, the function should run the exact SQL
    from the issue spec: SELECT * FROM incidents WHERE owner = ? ORDER BY
    timestamp DESC.
    """
    import sqlite3

    db_file = tmp_path / "legacy_incidents.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            CREATE TABLE incidents (
                id INTEGER PRIMARY KEY,
                owner TEXT,
                timestamp TEXT,
                document_a TEXT,
                document_b TEXT,
                similarity_score REAL,
                details TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO incidents (owner, timestamp, document_a, document_b, similarity_score, details) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("alice", "2026-05-01T10:00:00Z", "a1.pdf", "a2.pdf", 0.85, "Incident A"),
        )
        conn.execute(
            "INSERT INTO incidents (owner, timestamp, document_a, document_b, similarity_score, details) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("alice", "2026-05-02T10:00:00Z", "a3.pdf", "a4.pdf", 0.92, "Incident B"),
        )
        conn.execute(
            "INSERT INTO incidents (owner, timestamp, document_a, document_b, similarity_score, details) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("bob", "2026-05-03T10:00:00Z", "b1.pdf", "b2.pdf", 0.78, "Incident C"),
        )
        conn.commit()

    # Alice should see her two incidents, newest first.
    res = get_incidents_by_user("alice", db_path=db_file)
    assert len(res) == 2
    assert res[0]["timestamp"] == "2026-05-02T10:00:00Z"
    assert res[0]["details"] == "Incident B"
    assert res[1]["timestamp"] == "2026-05-01T10:00:00Z"
    assert res[1]["details"] == "Incident A"

    # Bob sees one.
    bob_res = get_incidents_by_user("bob", db_path=db_file)
    assert len(bob_res) == 1
    assert bob_res[0]["details"] == "Incident C"

    # Unknown user sees none.
    assert get_incidents_by_user("charlie", db_path=db_file) == []


def test_get_incidents_by_user_legacy_table_returns_all_columns(tmp_path):
    """The legacy path uses SELECT *, so every column on the legacy table
    should be present in the returned dicts."""
    import sqlite3

    db_file = tmp_path / "legacy_full.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            CREATE TABLE incidents (
                id INTEGER PRIMARY KEY,
                owner TEXT,
                timestamp TEXT,
                document_a TEXT,
                document_b TEXT,
                similarity_score REAL,
                severity_rank TEXT,
                review_status TEXT,
                details TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO incidents (owner, timestamp, document_a, document_b, "
            "similarity_score, severity_rank, review_status, details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "alice",
                "2026-05-01T10:00:00Z",
                "a.pdf",
                "b.pdf",
                0.91,
                "High",
                "Pending",
                "flagged",
            ),
        )
        conn.commit()

    res = get_incidents_by_user("alice", db_path=db_file)
    assert len(res) == 1
    row = res[0]
    # All columns from SELECT * must be present.
    for col in (
        "id",
        "owner",
        "timestamp",
        "document_a",
        "document_b",
        "similarity_score",
        "severity_rank",
        "review_status",
        "details",
    ):
        assert col in row, f"Missing column: {col}"
    assert row["details"] == "flagged"
    assert row["severity_rank"] == "High"


# ── Issue #1772: Additional comprehensive tests for get_incident_by_id ──────


def test_get_incident_by_id_returns_none_for_nonexistent_integer(test_db):
    """A nonexistent integer ID should return None, not raise."""
    result = get_incident_by_id(99999, test_db)
    assert result is None


def test_get_incident_by_id_returns_none_for_zero(test_db):
    """ID 0 (never auto-assigned) should return None."""
    result = get_incident_by_id(0, test_db)
    assert result is None


def test_get_incident_by_id_returns_none_for_negative(test_db):
    """Negative IDs should return None without raising."""
    result = get_incident_by_id(-1, test_db)
    assert result is None


def test_get_incident_by_id_returns_all_expected_columns(test_db):
    """The returned dict should contain all 9 canonical incident columns."""
    import sqlite3

    with sqlite3.connect(test_db) as conn:
        conn.execute(
            """
            INSERT INTO plagiarism_incidents (
                incident_id, document_a, document_b, similarity_score,
                severity_rank, review_status, date_flagged, last_seen,
                threshold_at_time_of_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                2001,
                "col_a.pdf",
                "col_b.pdf",
                0.92,
                "High",
                "Pending",
                "2026-08-01T10:00:00Z",
                "2026-08-01T10:00:00Z",
                0.59,
            ),
        )
        conn.commit()

    result = get_incident_by_id(2001, test_db)
    assert result is not None

    expected_keys = {
        "incident_id",
        "document_a",
        "document_b",
        "similarity_score",
        "severity_rank",
        "review_status",
        "date_flagged",
        "last_seen",
        "threshold_at_time_of_flag",
    }
    assert expected_keys == set(result.keys())
    assert result["similarity_score"] == 0.92
    assert result["severity_rank"] == "High"
    assert result["threshold_at_time_of_flag"] == 0.59


def test_get_incident_by_id_string_integer_equivalence(test_db):
    """Passing int 3001 and str '3001' should return the same record."""
    import sqlite3

    with sqlite3.connect(test_db) as conn:
        conn.execute(
            """
            INSERT INTO plagiarism_incidents (
                incident_id, document_a, document_b, similarity_score,
                severity_rank, review_status, date_flagged, last_seen,
                threshold_at_time_of_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                3001,
                "equiv_a.pdf",
                "equiv_b.pdf",
                0.81,
                "High",
                "Pending",
                "2026-08-02T00:00:00Z",
                "2026-08-02T00:00:00Z",
                0.50,
            ),
        )
        conn.commit()

    by_int = get_incident_by_id(3001, test_db)
    by_str = get_incident_by_id("3001", test_db)

    assert by_int is not None
    assert by_str is not None
    assert by_int["document_a"] == by_str["document_a"]
    assert by_int["document_b"] == by_str["document_b"]


def test_get_incident_by_id_excludes_soft_deleted_documents(test_db):
    """If either document is soft-deleted, the incident should not be returned."""
    import sqlite3

    with sqlite3.connect(test_db) as conn:
        conn.execute(
            "INSERT INTO documents (filename, file_hash, upload_date, is_deleted) "
            "VALUES (?, ?, ?, 1)",
            ("deleted_doc.pdf", "hash_del", "2026-08-03T00:00:00Z"),
        )
        conn.execute(
            "INSERT INTO documents (filename, file_hash, upload_date, is_deleted) "
            "VALUES (?, ?, ?, 0)",
            ("alive_doc.pdf", "hash_alive", "2026-08-03T00:00:00Z"),
        )
        conn.execute(
            """
            INSERT INTO plagiarism_incidents (
                incident_id, document_a, document_b, similarity_score,
                severity_rank, review_status, date_flagged, last_seen,
                threshold_at_time_of_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                4001,
                "deleted_doc.pdf",
                "alive_doc.pdf",
                0.85,
                "High",
                "Pending",
                "2026-08-03T00:00:00Z",
                "2026-08-03T00:00:00Z",
                0.50,
            ),
        )
        conn.commit()

    result = get_incident_by_id(4001, test_db)
    assert result is None


def test_get_incident_by_id_uses_default_db_path_when_none(test_db, monkeypatch):
    """When db_path is None, the function should use DEFAULT_DB_PATH."""
    import src.db.incidents as incidents_mod

    monkeypatch.setattr(incidents_mod, "DEFAULT_DB_PATH", test_db)

    import sqlite3

    with sqlite3.connect(test_db) as conn:
        conn.execute(
            """
            INSERT INTO plagiarism_incidents (
                incident_id, document_a, document_b, similarity_score,
                severity_rank, review_status, date_flagged, last_seen,
                threshold_at_time_of_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                5001,
                "default_a.pdf",
                "default_b.pdf",
                0.77,
                "Medium",
                "Pending",
                "2026-08-04T00:00:00Z",
                "2026-08-04T00:00:00Z",
                0.50,
            ),
        )
        conn.commit()

    result = get_incident_by_id(5001, None)
    assert result is not None
    assert result["document_a"] == "default_a.pdf"


def test_get_incident_by_id_returns_dict_type(test_db):
    """The return type should be dict or None, never a sqlite3.Row."""
    import sqlite3

    with sqlite3.connect(test_db) as conn:
        conn.execute(
            """
            INSERT INTO plagiarism_incidents (
                incident_id, document_a, document_b, similarity_score,
                severity_rank, review_status, date_flagged, last_seen,
                threshold_at_time_of_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                6001,
                "type_a.pdf",
                "type_b.pdf",
                0.65,
                "Medium",
                "Pending",
                "2026-08-05T00:00:00Z",
                "2026-08-05T00:00:00Z",
                0.50,
            ),
        )
        conn.commit()

    result = get_incident_by_id(6001, test_db)
    assert result is not None
    assert isinstance(result, dict)
    assert not isinstance(result, sqlite3.Row)


def test_self_plagiarism_exclusion(test_db):
    """Verify that sync_flagged_incidents skips self-plagiarism when allow_self_plagiarism_flags=False."""
    import sqlite3

    from src.db.incidents import get_all_incidents, sync_flagged_incidents

    # 1. Insert documents with matching student_name into database
    with sqlite3.connect(test_db) as conn:
        conn.execute(
            """
            INSERT INTO documents (filename, file_hash, upload_date, class_section, student_name, assignment_title)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "student_draft1.pdf",
                "hash1",
                "2026-08-01",
                "CS101",
                "Alice Smith",
                "Assignment 1",
            ),
        )
        conn.execute(
            """
            INSERT INTO documents (filename, file_hash, upload_date, class_section, student_name, assignment_title)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "student_draft2.pdf",
                "hash2",
                "2026-08-02",
                "CS101",
                "Alice Smith",
                "Assignment 1",
            ),
        )
        conn.commit()

    # 2. Call sync_flagged_incidents with allow_self_plagiarism_flags=False
    flag = {
        "doc_a": "student_draft1.pdf",
        "doc_b": "student_draft2.pdf",
        "similarity": 0.95,
        "severity": "High",
    }

    results = sync_flagged_incidents(
        [flag], db_path=test_db, allow_self_plagiarism_flags=False
    )

    # 3. Assert it did not return any match results and is not in the db
    assert len(results) == 0
    all_incidents = get_all_incidents(db_path=test_db)
    assert len(all_incidents) == 0

    # 4. Call sync_flagged_incidents with allow_self_plagiarism_flags=True
    results_allow = sync_flagged_incidents(
        [flag], db_path=test_db, allow_self_plagiarism_flags=True
    )
    # It should successfully log and return the MatchResult
    assert len(results_allow) == 1
    assert results_allow[0].document_a == "student_draft1.pdf"


def test_self_plagiarism_no_exclusion_when_names_differ(test_db):
    """Verify that sync_flagged_incidents does not skip when student names differ or are empty."""
    import sqlite3

    from src.db.incidents import sync_flagged_incidents

    # 1. Insert documents with different student names
    with sqlite3.connect(test_db) as conn:
        conn.execute(
            """
            INSERT INTO documents (filename, file_hash, upload_date, class_section, student_name, assignment_title)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "alice_draft.pdf",
                "hash3",
                "2026-08-01",
                "CS101",
                "Alice Smith",
                "Assignment 1",
            ),
        )
        conn.execute(
            """
            INSERT INTO documents (filename, file_hash, upload_date, class_section, student_name, assignment_title)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "bob_draft.pdf",
                "hash4",
                "2026-08-02",
                "CS101",
                "Bob Jones",
                "Assignment 1",
            ),
        )
        conn.commit()

    # 2. Call sync_flagged_incidents with allow_self_plagiarism_flags=False
    flag = {
        "doc_a": "alice_draft.pdf",
        "doc_b": "bob_draft.pdf",
        "similarity": 0.85,
        "severity": "Medium",
    }

    results = sync_flagged_incidents(
        [flag], db_path=test_db, allow_self_plagiarism_flags=False
    )

    # 3. Assert it was NOT skipped (since student names differ)
    assert len(results) == 1
    assert results[0].document_a == "alice_draft.pdf"
