import csv
import io

import pytest

from src.db.incidents import (build_incident_id, export_current_flags_csv,
                              get_all_incidents, incidents_to_csv,
                              sync_flagged_incidents, update_review_status)


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
