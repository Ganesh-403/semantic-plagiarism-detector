import csv
import io

import pytest

from src.db.incidents import (CSV_COLUMNS, build_incident_id,
                              export_current_flags_csv, get_all_incidents,
                              incidents_to_csv, sync_flagged_incidents,
                              update_review_status)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "incident-test.db"


@pytest.fixture
def flags():
    return [
        {
            "doc_a": "beta.pdf",
            "doc_b": "alpha.pdf",
            "similarity": 0.93456,
            "severity": "High",
        },
        {
            "doc_a": "alpha.pdf",
            "doc_b": "gamma.pdf",
            "similarity": 0.72,
            "severity": "Medium",
        },
    ]


def parse_csv(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def test_incident_id_is_order_independent():
    assert build_incident_id("a.pdf", "b.pdf") == build_incident_id("b.pdf", "a.pdf")


def test_sync_creates_all_warning_records(db_path, flags):
    incidents = sync_flagged_incidents(flags, db_path, now="2026-07-18T10:00:00+00:00")
    assert len(incidents) == 2
    assert {i["review_status"] for i in incidents} == {"Pending"}


def test_resync_preserves_date_and_status(db_path, flags):
    first = sync_flagged_incidents(flags, db_path, now="2026-07-18T10:00:00+00:00")
    incident_id = first[0]["incident_id"]
    assert update_review_status(incident_id, "Resolved", db_path)
    changed = [dict(x) for x in flags]
    changed[0]["similarity"] = 0.99
    sync_flagged_incidents(changed, db_path, now="2026-07-19T10:00:00+00:00")
    row = next(i for i in get_all_incidents(db_path) if i["incident_id"] == incident_id)
    assert row["review_status"] == "Resolved"
    assert row["date_flagged"] == "2026-07-18T10:00:00+00:00"


def test_invalid_status_rejected(db_path, flags):
    incident = sync_flagged_incidents(flags[:1], db_path)[0]
    with pytest.raises(ValueError):
        update_review_status(incident["incident_id"], "Ignored", db_path)


def test_csv_has_exact_required_columns(db_path, flags):
    rows = parse_csv(export_current_flags_csv(flags, db_path))
    assert list(rows[0].keys()) == CSV_COLUMNS
    assert len(rows) == 2


def test_csv_supports_unicode_filenames():
    data = incidents_to_csv(
        [
            {
                "incident_id": "INC-123",
                "document_a": "हिंदी.pdf",
                "document_b": "english.pdf",
                "similarity_score": 0.912345,
                "severity_rank": "High",
                "review_status": "Pending",
                "date_flagged": "2026-07-18T10:00:00+00:00",
            }
        ]
    )
    rows = parse_csv(data)
    assert rows[0]["Document A"] == "हिंदी.pdf"
    assert rows[0]["Similarity Score"] == "0.9123"
    assert data.startswith(b"\xef\xbb\xbf")
