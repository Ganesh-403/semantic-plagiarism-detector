import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.db.incidents import (
    get_all_incidents,
    get_total_incidents_count,
    sync_flagged_incidents,
)


@pytest.fixture()
def populated_incident_db(mock_db):
    base_time = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    for index in range(65):
        timestamp = (
            base_time + timedelta(minutes=index)
        ).isoformat()
        sync_flagged_incidents(
            [
                {
                    "doc_a": f"document-{index:03d}-a.pdf",
                    "doc_b": f"document-{index:03d}-b.pdf",
                    "similarity": 0.85,
                }
            ],
            mock_db,
            now=timestamp,
        )

    return mock_db


def test_default_page_contains_first_50_incidents(
    populated_incident_db,
):
    incidents = get_all_incidents(populated_incident_db)

    assert len(incidents) == 50
    assert incidents[0].document_a == "document-064-a.pdf"
    assert incidents[-1].document_a == "document-015-a.pdf"


def test_second_page_uses_limit_and_offset(
    populated_incident_db,
):
    incidents = get_all_incidents(
        populated_incident_db,
        limit=10,
        offset=50,
    )

    assert len(incidents) == 10
    assert incidents[0].document_a == "document-014-a.pdf"
    assert incidents[-1].document_a == "document-005-a.pdf"


def test_final_partial_page(
    populated_incident_db,
):
    incidents = get_all_incidents(
        populated_incident_db,
        limit=10,
        offset=60,
    )

    assert len(incidents) == 5
    assert incidents[0].document_a == "document-004-a.pdf"
    assert incidents[-1].document_a == "document-000-a.pdf"


def test_offset_beyond_total_returns_empty_page(
    populated_incident_db,
):
    assert get_all_incidents(
        populated_incident_db,
        limit=10,
        offset=1000,
    ) == []


def test_total_count_is_independent_of_page_size(
    populated_incident_db,
):
    assert get_total_incidents_count(
        populated_incident_db
    ) == 65
    assert len(
        get_all_incidents(
            populated_incident_db,
            limit=7,
            offset=0,
        )
    ) == 7


@pytest.mark.parametrize(
    ("limit", "offset", "exception_type"),
    [
        (0, 0, ValueError),
        (-1, 0, ValueError),
        (10, -1, ValueError),
        (1.5, 0, TypeError),
        (10, "0", TypeError),
        (True, 0, TypeError),
        (10, False, TypeError),
    ],
)
def test_invalid_pagination_arguments_are_rejected(
    mock_db,
    limit,
    offset,
    exception_type,
):
    with pytest.raises(exception_type):
        get_all_incidents(
            mock_db,
            limit=limit,
            offset=offset,
        )


def test_count_and_page_exclude_soft_deleted_documents(
    mock_db,
):
    sync_flagged_incidents(
        [
            {
                "doc_a": "active-a.pdf",
                "doc_b": "active-b.pdf",
                "similarity": 0.9,
            },
            {
                "doc_a": "deleted-a.pdf",
                "doc_b": "deleted-b.pdf",
                "similarity": 0.9,
            },
        ],
        mock_db,
    )

    with sqlite3.connect(mock_db) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO documents (
                filename,
                file_hash,
                is_deleted
            )
            VALUES (?, ?, ?)
            """,
            ("deleted-a.pdf", "deleted-a-hash", 1),
        )
        connection.commit()

    incidents = get_all_incidents(
        mock_db,
        limit=50,
        offset=0,
    )

    assert get_total_incidents_count(mock_db) == 1
    assert len(incidents) == 1
    assert incidents[0].document_a == "active-a.pdf"


def test_order_is_stable_when_timestamps_match(mock_db):
    timestamp = "2026-01-01T00:00:00+00:00"
    sync_flagged_incidents(
        [
            {
                "doc_a": "a.pdf",
                "doc_b": "b.pdf",
                "similarity": 0.9,
            },
            {
                "doc_a": "c.pdf",
                "doc_b": "d.pdf",
                "similarity": 0.9,
            },
        ],
        mock_db,
        now=timestamp,
    )

    first = get_all_incidents(
        mock_db,
        limit=1,
        offset=0,
    )
    second = get_all_incidents(
        mock_db,
        limit=1,
        offset=1,
    )

    assert first[0].incident_id < second[0].incident_id
