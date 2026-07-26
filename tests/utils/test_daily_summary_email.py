"""
test_daily_summary_email.py
---------------------------
Tests for daily summary email functionality.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.utils.daily_summary_email import (format_daily_summary,
                                           get_admin_emails,
                                           get_incidents_last_24h,
                                           send_daily_summary, send_email)


@pytest.fixture
def mock_incidents():
    """Mock incident data for testing."""
    return [
        {
            "incident_id": "INC-123",
            "document_a": "student1.pdf",
            "document_b": "student2.pdf",
            "similarity_score": 0.95,
            "severity_rank": "High",
            "review_status": "Pending",
            "date_flagged": (
                datetime.now(timezone.utc) - timedelta(hours=2)
            ).isoformat(),
            "last_seen": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        },
        {
            "incident_id": "INC-456",
            "document_a": "student3.pdf",
            "document_b": "student4.pdf",
            "similarity_score": 0.75,
            "severity_rank": "Medium",
            "review_status": "Pending",
            "date_flagged": (
                datetime.now(timezone.utc) - timedelta(hours=12)
            ).isoformat(),
            "last_seen": (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat(),
        },
        {
            "incident_id": "INC-789",
            "document_a": "student5.pdf",
            "document_b": "student6.pdf",
            "similarity_score": 0.45,
            "severity_rank": "Low",
            "review_status": "Pending",
            "date_flagged": (
                datetime.now(timezone.utc) - timedelta(hours=23)
            ).isoformat(),
            "last_seen": (datetime.now(timezone.utc) - timedelta(hours=23)).isoformat(),
        },
    ]


@pytest.fixture
def mock_old_incident():
    """Mock incident older than 24 hours."""
    return {
        "incident_id": "INC-OLD",
        "document_a": "old1.pdf",
        "document_b": "old2.pdf",
        "similarity_score": 0.90,
        "severity_rank": "High",
        "review_status": "Pending",
        "date_flagged": (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(),
        "last_seen": (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(),
    }


@patch("src.utils.daily_summary_email.get_all_incidents")
def test_get_incidents_last_24h(mock_get_all, mock_incidents, mock_old_incident):
    """Test filtering incidents from last 24 hours."""
    mock_get_all.return_value = mock_incidents + [mock_old_incident]

    recent = get_incidents_last_24h()

    assert len(recent) == 3
    assert all(inc["incident_id"] != "INC-OLD" for inc in recent)


@patch("src.utils.daily_summary_email.get_all_users")
def test_get_admin_emails(mock_get_users):
    """Test retrieving admin email addresses."""
    mock_get_users.return_value = [
        {"id": 1, "username": "admin1", "role": "admin"},
        {"id": 2, "username": "teacher1", "role": "teacher"},
        {"id": 3, "username": "admin2", "role": "admin"},
    ]

    emails = get_admin_emails()

    assert len(emails) == 2
    assert "admin1@localhost" in emails
    assert "admin2@localhost" in emails
    assert "teacher1@localhost" not in emails


@patch("src.utils.daily_summary_email.get_all_users")
@patch.dict("os.environ", {"ADMIN_EMAIL": "fallback@example.com"})
def test_get_admin_emails_fallback(mock_get_users):
    """Test fallback to environment variable when no admins exist."""
    mock_get_users.return_value = [{"id": 1, "username": "teacher1", "role": "teacher"}]

    emails = get_admin_emails()

    assert len(emails) == 1
    assert emails[0] == "fallback@example.com"


def test_format_daily_summary_with_incidents(mock_incidents):
    """Test formatting daily summary with incidents."""
    html = format_daily_summary(mock_incidents)

    assert "Daily Plagiarism Summary" in html
    assert "Total new incidents: 3" in html
    assert "<strong>High:</strong> 1" in html
    assert "<strong>Medium:</strong> 1" in html
    assert "<strong>Low:</strong> 1" in html
    assert "student1.pdf" in html
    assert "student3.pdf" in html
    assert "95.00%" in html


def test_format_daily_summary_empty():
    """Test formatting daily summary with no incidents."""
    html = format_daily_summary([])

    assert "Daily Plagiarism Summary" in html
    assert "No new plagiarism incidents detected" in html


@patch("smtplib.SMTP")
@patch.dict(
    "os.environ",
    {
        "SMTP_SERVER": "smtp.example.com",
        "SMTP_PORT": "587",
        "SMTP_USERNAME": "test@example.com",
        "SMTP_PASSWORD": "password",
        "FROM_EMAIL": "test@example.com",
    },
)
def test_send_email_success(mock_smtp):
    """Test successful email sending."""
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server

    result = send_email(["recipient@example.com"], "Test Subject", "<p>Test Body</p>")

    assert result is True
    mock_server.starttls.assert_called_once()
    mock_server.login.assert_called_once_with("test@example.com", "password")
    mock_server.send_message.assert_called_once()


@patch.dict("os.environ", {}, clear=True)
def test_send_email_missing_config():
    """Test email sending with missing SMTP configuration."""
    result = send_email(["recipient@example.com"], "Test Subject", "<p>Test Body</p>")

    assert result is False


@patch.dict(
    "os.environ",
    {
        "SMTP_SERVER": "smtp.example.com",
        "SMTP_PORT": "587",
        "SMTP_USERNAME": "test@example.com",
        "SMTP_PASSWORD": "password",
    },
)
def test_send_email_no_recipients():
    """Test email sending with no recipients."""
    result = send_email([], "Test Subject", "<p>Test Body</p>")

    assert result is False


@patch("src.utils.daily_summary_email.send_email")
@patch("src.utils.daily_summary_email.get_admin_emails")
@patch("src.utils.daily_summary_email.get_incidents_last_24h")
def test_send_daily_summary(mock_get_incidents, mock_get_emails, mock_send_email):
    """Test the complete daily summary workflow."""
    mock_get_incidents.return_value = [
        {
            "incident_id": "INC-123",
            "document_a": "test1.pdf",
            "document_b": "test2.pdf",
            "similarity_score": 0.90,
            "severity_rank": "High",
            "review_status": "Pending",
            "date_flagged": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }
    ]
    mock_get_emails.return_value = ["admin@example.com"]
    mock_send_email.return_value = True

    result = send_daily_summary()

    assert result is True
    mock_send_email.assert_called_once()
    call_args = mock_send_email.call_args
    assert "Daily Plagiarism Summary" in call_args[0][1]  # subject
