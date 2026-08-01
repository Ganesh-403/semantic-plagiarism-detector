import csv
import io
from datetime import datetime, timedelta, timezone

import pytest

from src.db.incidents import (
    build_incident_id,
    export_current_flags_csv,
    get_all_incidents,
    get_incidents_by_severity,

    get_incident_by_id,
    incidents_to_csv,
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
    with pytest.raises(ValueError):
        update_review_status(
            "INC-123456",
            "Done",
            test_db,
        )

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
            (old_date,)
        )
        conn.commit()

    # Create a pending incident with an old date (should NOT be deleted)
    with sqlite3.connect(test_db) as conn:
        conn.execute(
            "INSERT INTO plagiarism_incidents (incident_id, document_a, document_b, similarity_score, severity_rank, review_status, date_flagged, last_seen, threshold_at_time_of_flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("INC-OLDPENDING", "old_pending1.pdf", "old_pending2.pdf", 0.85, "Medium", "Pending", old_date, old_date, 0.59)
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
            (1046, "int_doc_a.pdf", "int_doc_b.pdf", 0.75, "Medium", "Pending", "2026-07-31T00:00:00Z", "2026-07-31T00:00:00Z", 0.50),
        )
        conn.commit()

    result = get_incident_by_id(1046, test_db)
    assert result is not None
    assert isinstance(result, dict)
    assert result["document_a"] == "int_doc_a.pdf"
    assert result["document_b"] == "int_doc_b.pdf"

