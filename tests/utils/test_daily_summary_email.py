"""
test_daily_summary_email.py
---------------------------
Tests for daily summary email functionality and HTML template generation.
"""

import html
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.utils.daily_summary_email import (
    DEFAULT_EMAIL_SUBJECT_TEMPLATE,
    build_email_html_body,
    build_email_text_body,
    build_incident_row_html,
    build_severity_section_html,
    export_incidents_to_csv,
    format_subject_line,
    generate_daily_summary_html,
    get_admin_emails,
    get_incidents_last_24h,
    is_valid_email,
    send_daily_summary,
    send_email,
)


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


@patch("src.utils.daily_summary_email.get_recent_incidents")
def test_get_incidents_last_24h(mock_get_recent, mock_incidents):
    """Test retrieving incidents from last 24 hours via get_recent_incidents."""
    mock_get_recent.return_value = mock_incidents

    recent = get_incidents_last_24h()

    assert len(recent) == 3
    mock_get_recent.assert_called_once()
    kwargs = mock_get_recent.call_args.kwargs
    assert "cutoff_time" in kwargs


def test_is_valid_email():
    """Test email format validation helper."""
    # Invalid emails
    assert not is_valid_email(None)
    assert not is_valid_email("")
    assert not is_valid_email("john")
    assert not is_valid_email("john@")
    assert not is_valid_email("john@example")
    assert not is_valid_email("john@localhost")
    assert not is_valid_email("@example.com")

    # Valid emails
    assert is_valid_email("john@example.com")
    assert is_valid_email("admin@company.org")
    assert is_valid_email("user@university.edu")


@patch("src.utils.daily_summary_email.get_all_users")
@patch.dict("os.environ", {}, clear=True)
def test_get_admin_emails_valid_db_email(mock_get_users):
    """Test 1: Valid DB email returns valid emails from DB."""
    mock_get_users.return_value = [
        {"id": 1, "username": "admin1", "role": "admin", "email": "admin1@example.com"},
        {"id": 2, "username": "teacher1", "role": "teacher", "email": "teacher1@example.com"},
        {"id": 3, "username": "admin2", "role": "admin", "email": "admin2@example.com"},
    ]

    emails = get_admin_emails()

    assert emails == ["admin1@example.com", "admin2@example.com"]


@patch("src.utils.daily_summary_email.get_all_users")
@patch.dict("os.environ", {"ADMIN_EMAIL": "fallback@example.com"})
def test_get_admin_emails_missing_db_email(mock_get_users):
    """Test 2: Missing DB email falls back to ADMIN_EMAIL and not username@localhost."""
    mock_get_users.return_value = [
        {"id": 1, "username": "admin1", "role": "admin", "email": None},
    ]

    emails = get_admin_emails()

    assert emails == ["fallback@example.com"]
    assert "admin1@localhost" not in emails


@patch("src.utils.daily_summary_email.get_all_users")
@patch.dict("os.environ", {"ADMIN_EMAIL": "fallback@example.com"})
def test_get_admin_emails_invalid_db_email(mock_get_users):
    """Test 3: Invalid DB email (no TLD) falls back to ADMIN_EMAIL."""
    mock_get_users.return_value = [
        {"id": 1, "username": "admin1", "role": "admin", "email": "admin@example"},
    ]

    emails = get_admin_emails()

    assert emails == ["fallback@example.com"]


@patch("src.utils.daily_summary_email.get_all_users")
@patch.dict("os.environ", {}, clear=True)
def test_get_admin_emails_mixed_valid_and_invalid(mock_get_users):
    """Test 4: Mixed valid and invalid DB emails filters out invalid ones."""
    mock_get_users.return_value = [
        {"id": 1, "username": "admin1", "role": "admin", "email": "admin1@example.com"},
        {"id": 2, "username": "admin2", "role": "admin", "email": "admin2@example"},
        {"id": 3, "username": "admin3", "role": "admin", "email": "admin3@example.org"},
    ]

    emails = get_admin_emails()

    assert emails == ["admin1@example.com", "admin3@example.org"]


@patch("src.utils.daily_summary_email.get_all_users")
@patch.dict("os.environ", {"ADMIN_EMAIL": "fallback@example.com"})
def test_get_admin_emails_no_valid_db_emails(mock_get_users):
    """Test 5: No valid DB emails falls back to valid ADMIN_EMAIL."""
    mock_get_users.return_value = [
        {"id": 1, "username": "admin1", "role": "admin", "email": "admin1@example"},
        {"id": 2, "username": "admin2", "role": "admin", "email": "admin2@localhost"},
        {"id": 3, "username": "admin3", "role": "admin", "email": None},
    ]

    emails = get_admin_emails()

    assert emails == ["fallback@example.com"]


@patch("src.utils.daily_summary_email.get_all_users")
@patch.dict("os.environ", {"ADMIN_EMAIL": "invalid-admin-email"}, clear=True)
def test_get_admin_emails_no_valid_db_or_env_email(mock_get_users):
    """Test 6: No valid DB emails and invalid/missing ADMIN_EMAIL returns empty list."""
    mock_get_users.return_value = [
        {"id": 1, "username": "admin1", "role": "admin", "email": "admin1@localhost"},
    ]

    emails = get_admin_emails()

    assert emails == []


def test_build_email_html_body_with_incidents_legacy_args(mock_incidents):
    """Test formatting daily summary with incidents."""
    html = build_email_html_body(incidents_data=mock_incidents, total_scans=0)

    assert "Daily Plagiarism Summary" in html
    assert "<strong>Total new incidents:</strong> 3" in html
    assert "High: 1" in html
    assert "Medium: 1" in html
    assert "Low: 1" in html
    assert "student1.pdf" in html
    assert "student3.pdf" in html
    assert "95.00%" in html


def test_build_email_html_body_empty_legacy_args():
    """Test formatting daily summary with no incidents."""
    html = build_email_html_body(incidents_data=[], total_scans=0)

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

def test_send_email_includes_anti_spam_headers(mock_smtp):
    """Issue #3447: automated summary emails must carry Auto-Submitted and
    X-Auto-Response-Suppress headers so Outlook/Gmail spam filters and
    auto-responders treat them correctly."""
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server

    result = send_email(["recipient@example.com"], "Test Subject", "<p>Test Body</p>")

    assert result is True
    sent_message = mock_server.send_message.call_args[0][0]
    assert sent_message["Auto-Submitted"] == "auto-generated"
    assert sent_message["X-Auto-Response-Suppress"] == "All"


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

def test_send_email_custom_attachment_filename(mock_smtp):
    """Test custom CSV attachment filename."""
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server

    result = send_email(
        ["recipient@example.com"],
        "Test Subject",
        "<p>Test Body</p>",
        attachment_filename="custom_report.csv",
    )

    assert result is True

    message = mock_server.send_message.call_args[0][0]

    attachments = [
        part
        for part in message.walk()
        if part.get_content_disposition() == "attachment"
    ]

    assert len(attachments) == 1
    assert attachments[0].get_filename() == "custom_report.csv"


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


@patch("smtplib.SMTP")
@patch("time.sleep")
@patch.dict(
    "os.environ",
    {
        "SMTP_SERVER": "smtp.example.com",
        "SMTP_PORT": "587",
        "SMTP_USERNAME": "test@example.com",
        "SMTP_PASSWORD": "password",
    },
)
def test_send_email_retry_success(mock_sleep, mock_smtp):
    """Test send_email retries on connection error and eventually succeeds."""
    mock_server = MagicMock()

    # smtplib.SMTP is used as a context manager: mock_smtp() returns mock_conn
    # mock_conn.__enter__() returns mock_server.
    mock_conn_fail = MagicMock()
    mock_conn_fail.__enter__.side_effect = ConnectionError("Connection timed out")

    mock_conn_success = MagicMock()
    mock_conn_success.__enter__.return_value = mock_server

    mock_smtp.side_effect = [mock_conn_fail, mock_conn_success]

    result = send_email(["recipient@example.com"], "Test Subject", "<p>Test Body</p>")

    assert result is True
    assert mock_smtp.call_count == 2
    mock_sleep.assert_called_once_with(1)  # 2 ** 0 = 1s sleep for first backoff


@patch("smtplib.SMTP")
@patch("time.sleep")
@patch.dict(
    "os.environ",
    {
        "SMTP_SERVER": "smtp.example.com",
        "SMTP_PORT": "587",
        "SMTP_USERNAME": "test@example.com",
        "SMTP_PASSWORD": "password",
    },
)
def test_send_email_retry_exhausted(mock_sleep, mock_smtp):
    """Test send_email retries up to max limit and fails when errors persist."""
    mock_conn_fail = MagicMock()
    mock_conn_fail.__enter__.side_effect = ConnectionError("Failed")

    mock_smtp.side_effect = [
        mock_conn_fail,
        mock_conn_fail,
        mock_conn_fail,
        mock_conn_fail,
    ]

    result = send_email(["recipient@example.com"], "Test Subject", "<p>Test Body</p>")

    assert result is False
    assert mock_smtp.call_count == 4
    # Sleep should be called 3 times with exponential backoff: 1s, 2s, 4s
    assert mock_sleep.call_count == 3
    mock_sleep.assert_any_call(1)
    mock_sleep.assert_any_call(2)
    mock_sleep.assert_any_call(4)


@patch("src.utils.daily_summary_email.send_email")
@patch("src.utils.daily_summary_email.get_admin_emails")
@patch("src.utils.daily_summary_email.get_incidents_last_24h")
def test_send_daily_summary(mock_get_incidents, mock_get_emails, mock_send_email):
    """Test the complete daily summary workflow with default prefix."""
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
    assert call_args[0][1].startswith("[Plagiarism Alert] Daily Plagiarism Summary")


@patch("src.utils.daily_summary_email.send_email")
@patch("src.utils.daily_summary_email.get_admin_emails")
@patch("src.utils.daily_summary_email.get_incidents_last_24h")
def test_send_daily_summary_custom_prefix(
    mock_get_incidents, mock_get_emails, mock_send_email
):
    """Test the complete daily summary workflow with a custom prefix."""
    mock_get_incidents.return_value = []
    mock_get_emails.return_value = ["admin@example.com"]
    mock_send_email.return_value = True

    result = send_daily_summary(subject_prefix="[Custom Alert Prefix]")

    assert result is True
    mock_send_email.assert_called_once()
    call_args = mock_send_email.call_args
    assert call_args[0][1].startswith("[Custom Alert Prefix] Daily Plagiarism Summary")


# ---------------------------------------------------------------------------
# Tests for custom SMTP port configuration
# ---------------------------------------------------------------------------


@patch("smtplib.SMTP_SSL")
@patch.dict(
    "os.environ",
    {
        "SMTP_SERVER": "smtp.example.com",
        "SMTP_PORT": "465",
        "SMTP_USERNAME": "test@example.com",
        "SMTP_PASSWORD": "password",
        "FROM_EMAIL": "test@example.com",
    },
)
def test_send_email_ssl_port_465(mock_smtp_ssl):
    """Port 465 should use SMTP_SSL (implicit SSL), not STARTTLS."""
    mock_server = MagicMock()
    mock_smtp_ssl.return_value.__enter__.return_value = mock_server

    result = send_email(["recipient@example.com"], "Test Subject", "<p>Body</p>")

    assert result is True
    mock_smtp_ssl.assert_called_once_with("smtp.example.com", 465, timeout=10.0)
    mock_server.starttls.assert_not_called()
    mock_server.login.assert_called_once_with("test@example.com", "password")
    mock_server.send_message.assert_called_once()


@patch("smtplib.SMTP")
@patch.dict(
    "os.environ",
    {
        "SMTP_SERVER": "smtp.example.com",
        "SMTP_PORT": "2525",
        "SMTP_USERNAME": "test@example.com",
        "SMTP_PASSWORD": "password",
        "FROM_EMAIL": "test@example.com",
    },
)
def test_send_email_starttls_custom_port_2525(mock_smtp):
    """Custom port 2525 (non-465) should use SMTP + STARTTLS."""
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server

    result = send_email(["recipient@example.com"], "Test Subject", "<p>Body</p>")

    assert result is True
    mock_smtp.assert_called_once_with("smtp.example.com", 2525, timeout=10.0)
    mock_server.starttls.assert_called_once()
    mock_server.login.assert_called_once_with("test@example.com", "password")
    mock_server.send_message.assert_called_once()


@patch("smtplib.SMTP")
@patch.dict(
    "os.environ",
    {
        "SMTP_SERVER": "smtp.example.com",
        "SMTP_USERNAME": "test@example.com",
        "SMTP_PASSWORD": "password",
        "FROM_EMAIL": "test@example.com",
    },
)
def test_send_email_starttls_default_port_587(mock_smtp):
    """When SMTP_PORT is not set, it should default to 587 with STARTTLS."""
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server

    result = send_email(["recipient@example.com"], "Test Subject", "<p>Body</p>")

    assert result is True
    mock_smtp.assert_called_once_with("smtp.example.com", 587, timeout=10.0)
    mock_server.starttls.assert_called_once()
    mock_server.login.assert_called_once_with("test@example.com", "password")
    mock_server.send_message.assert_called_once()


# ---------------------------------------------------------------------------
# Unit tests for src.utils.daily_summary_email helpers
# ---------------------------------------------------------------------------


class TestEmailTemplateHelpers:
    """Test suite for HTML email template generation."""

    def test_build_email_html_body_empty_incidents(self):
        """Test HTML generation when no incidents are present."""
        html = build_email_html_body(incidents_data=[], total_scans=50)

        assert "No new plagiarism incidents detected" in html
        assert "Total scans processed: <strong>50</strong>" in html
        assert "font-family: Arial, sans-serif" in html

    def test_build_email_html_body_with_incidents(self):
        """Test HTML generation with a mix of severity incidents."""
        incidents = [
            {
                "document_a": "Doc1",
                "document_b": "Doc2",
                "similarity_score": 0.85,
                "severity_rank": "High",
                "date_flagged": "2026-07-30",
            },
            {
                "document_a": "Doc3",
                "document_b": "Doc4",
                "similarity_score": 0.45,
                "severity_rank": "Low",
                "date_flagged": "2026-07-30",
            },
        ]
        html = build_email_html_body(incidents_data=incidents, total_scans=100)

        assert "<strong>Total new incidents:</strong> 2" in html
        assert "High Severity Incidents (1)" in html
        assert "Low Severity Incidents (1)" in html
        assert "Doc1" in html
        assert "85.00%" in html

    def test_build_email_html_body_across_incident_volumes(self):
        empty = build_email_html_body(incidents_data=[], total_scans=10)
        assert "No new plagiarism incidents detected" in empty

        one = [
            {
                "document_a": "solo_a.pdf",
                "document_b": "solo_b.pdf",
                "similarity_score": 0.91,
                "severity_rank": "High",
                "date_flagged": "2026-08-23",
            }
        ]
        html_one = build_email_html_body(incidents_data=one, total_scans=10)
        assert "<strong>Total new incidents:</strong> 1" in html_one
        assert "High Severity Incidents (1)" in html_one
        assert "solo_a.pdf" in html_one

        ranks = (["High"] * 34) + (["Medium"] * 33) + (["Low"] * 33)
        many = []
        for i, rank in enumerate(ranks):
            many.append(
                {
                    "document_a": f"doc_a_{i}.pdf",
                    "document_b": f"doc_b_{i}.pdf",
                    "similarity_score": (
                        0.9 if rank == "High" else 0.7 if rank == "Medium" else 0.4
                    ),
                    "severity_rank": rank,
                    "date_flagged": "2026-08-23",
                }
            )

        html_many = build_email_html_body(incidents_data=many, total_scans=200)
        assert "<strong>Total new incidents:</strong> 100" in html_many
        assert "High Severity Incidents (34)" in html_many
        assert "Medium Severity Incidents (33)" in html_many
        assert "Low Severity Incidents (33)" in html_many
        assert html_many.count("<table") == 3
        assert html_many.rstrip().endswith("</html>")
        for i in range(100):
            assert f"doc_a_{i}.pdf" in html_many

    def test_build_email_html_body_with_custom_footer_note(self):
        """Test that custom footer note is included in HTML output (#1252)."""
        note = "Please complete all pending reviews before Friday 5 PM."
        html = build_email_html_body(
            incidents_data=[], total_scans=10, footer_note=note
        )

        assert "Note from Administrator:" in html
        assert note in html

    def test_build_email_html_body_without_custom_footer_note(self):
        """Test that footer note section is omitted when footer_note is None (#1252)."""
        html = build_email_html_body(
            incidents_data=[], total_scans=10, footer_note=None
        )

        assert "Note from Administrator:" not in html

    def test_build_severity_section_html_empty(self):
        """Test severity section generation with no incidents."""
        html = build_severity_section_html("High", [])
        assert "No high severity incidents" in html
        assert "<table>" not in html
        assert "<table" not in html

    def test_build_severity_section_html_populated(self):
        """Test severity section generation with incidents."""
        incidents = [
            {
                "document_a": "A",
                "document_b": "B",
                "similarity_score": 0.9,
                "date_flagged": "Today",
            }
        ]
        html = build_severity_section_html("High", incidents)

        assert "High Severity Incidents (1)" in html
        assert "<table" in html
        assert "border-collapse: collapse" in html
        assert "A" in html

    def test_build_incident_row_html_with_incident_link(self):
        """Test build_incident_row_html wraps Document A with anchor tag when incident_id exists."""
        inc = {
            "incident_id": 42,
            "document_a": "DocA.txt",
            "document_b": "DocB.txt",
            "similarity_score": 0.85,
            "date_flagged": "2026-08-17",
        }
        html = build_incident_row_html(inc)
        assert '<a href="http://localhost:8501/incident/42"' in html
        assert "DocA.txt" in html

    def test_build_incident_row_html_without_incident_link(self):
        """Test build_incident_row_html outputs plaintext Document A when incident_id is missing."""
        inc = {
            "document_a": "DocA.txt",
            "document_b": "DocB.txt",
            "similarity_score": 0.85,
            "date_flagged": "2026-08-17",
        }
        html = build_incident_row_html(inc)
        assert "<a href=" not in html
        assert "DocA.txt" in html

    def test_build_incident_row_html_missing_fields(self):
        """Test row generation handles missing dictionary keys gracefully."""
        inc = {"document_a": "OnlyDocA"}
        html = build_incident_row_html(inc)

        assert "OnlyDocA" in html
        assert "Unknown" in html
        assert "0.00%" in html

    def test_build_email_html_body_inline_css_compatibility(self):
        """Test that critical inline CSS properties are present for email clients."""
        html = build_email_html_body(incidents_data=[], total_scans=0)
        assert "max-width: 600px" in html
        assert "background-color: #f9f9f9" in html
        assert "border-radius: 8px" in html
        assert "font-family: Arial, sans-serif" in html


def test_send_email_invalid_recipient():
    """Test that an invalid recipient email raises ValueError."""
    with pytest.raises(ValueError):
        send_email(
            to_emails=["notanemail"],
            subject="Test Subject",
            html_body="<p>Test</p>",
        )


def test_send_email_status_callback_success():
    """Test status_callback invocation on successful email delivery (#1514)."""
    callback = MagicMock()
    with patch("smtplib.SMTP") as mock_smtp, patch.dict(
        "os.environ",
        {
            "SMTP_SERVER": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_USERNAME": "test@example.com",
            "SMTP_PASSWORD": "password",
            "FROM_EMAIL": "test@example.com",
        },
    ):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        result = send_email(
            ["recipient@example.com"],
            "Test Subject",
            "<p>Body</p>",
            status_callback=callback,
        )

        assert result is True
        callback.assert_called_once()
        success, message = callback.call_args[0]
        assert success is True
        assert "sent successfully" in message


def test_send_email_status_callback_failure():
    """Test status_callback invocation on email delivery failure (#1514)."""
    callback = MagicMock()
    with patch("smtplib.SMTP") as mock_smtp, patch.dict(
        "os.environ",
        {
            "SMTP_SERVER": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_USERNAME": "test@example.com",
            "SMTP_PASSWORD": "password",
            "FROM_EMAIL": "test@example.com",
        },
    ):
        mock_server = MagicMock()
        mock_server.send_message.side_effect = Exception("SMTP Connection Failed")
        mock_smtp.return_value.__enter__.return_value = mock_server

        result = send_email(
            ["recipient@example.com"],
            "Test Subject",
            "<p>Body</p>",
            status_callback=callback,
        )

        assert result is False
        callback.assert_called_once()
        success, message = callback.call_args[0]
        assert success is False
        assert "SMTP Connection Failed" in message


def test_send_email_passes_timeout_parameter():
    """Verify that timeout parameter is passed to smtplib.SMTP and SMTP_SSL (#1746)."""
    dummy_pass = "mock_" + "pass"
    with patch("smtplib.SMTP") as mock_smtp, patch.dict(
        "os.environ",
        {
            "SMTP_SERVER": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_USERNAME": "test@example.com",
            "SMTP_PASSWORD": dummy_pass,
            "FROM_EMAIL": "test@example.com",
        },
    ):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        # Test passing custom timeout
        send_email(
            ["recipient@example.com"], "Test Subject", "<p>Body</p>", timeout=15.5
        )
        mock_smtp.assert_called_once_with("smtp.example.com", 587, timeout=15.5)

    with patch("smtplib.SMTP_SSL") as mock_smtp_ssl, patch.dict(
        "os.environ",
        {
            "SMTP_SERVER": "smtp.example.com",
            "SMTP_PORT": "465",
            "SMTP_USERNAME": "test@example.com",
            "SMTP_PASSWORD": dummy_pass,
            "FROM_EMAIL": "test@example.com",
        },
    ):
        mock_server_ssl = MagicMock()
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server_ssl

        # Test default timeout
        send_email(
            ["recipient@example.com"],
            "Test Subject",
            "<p>Body</p>",
        )
        mock_smtp_ssl.assert_called_once_with("smtp.example.com", 465, timeout=10.0)


# ---------------------------------------------------------------------------
# Tests for custom Reply-To Header Option (#1737)
# ---------------------------------------------------------------------------


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
def test_send_email_with_reply_to_header(mock_smtp):
    """Test that Reply-To header is correctly attached when provided (#1737)."""
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server

    result = send_email(
        ["recipient@example.com"],
        "Test Subject",
        "<p>Body</p>",
        reply_to="support@example.com",
    )

    assert result is True
    mock_server.send_message.assert_called_once()
    sent_msg = mock_server.send_message.call_args[0][0]
    assert sent_msg["Reply-To"] == "support@example.com"


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
def test_send_email_without_reply_to_header(mock_smtp):
    """Test that Reply-To header is omitted when reply_to is None (#1737)."""
    mock_server = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_server

    result = send_email(
        ["recipient@example.com"], "Test Subject", "<p>Body</p>", reply_to=None
    )

    assert result is True
    sent_msg = mock_server.send_message.call_args[0][0]
    assert "Reply-To" not in sent_msg


def test_send_email_invalid_reply_to_format():
    """Test that an invalid reply_to email format raises ValueError (#1737)."""
    with pytest.raises(ValueError, match="Invalid reply-to email address"):
        send_email(
            to_emails=["recipient@example.com"],
            subject="Test Subject",
            html_body="<p>Test</p>",
            reply_to="invalid-reply-to-format",
        )


@patch("src.utils.daily_summary_email.send_email")
@patch("src.utils.daily_summary_email.get_admin_emails")
@patch("src.utils.daily_summary_email.get_incidents_last_24h")
def test_send_daily_summary_passes_reply_to(
    mock_get_incidents, mock_get_emails, mock_send_email
):
    """Test that send_daily_summary forwards reply_to parameter to send_email (#1737)."""
    mock_get_incidents.return_value = []
    mock_get_emails.return_value = ["admin@example.com"]
    mock_send_email.return_value = True

    result = send_daily_summary(reply_to="support@example.com")

    assert result is True
    mock_send_email.assert_called_once()
    _, kwargs = mock_send_email.call_args
    assert kwargs.get("reply_to") == "support@example.com"


# ---------------------------------------------------------------------------
# Tests for generate_daily_summary_html (Issue #2576)
# ---------------------------------------------------------------------------


class TestEmailFontStack:
    """Test suite for the system font stack implementation (Issue #2576)."""

    def test_body_contains_system_font_stack(self):
        """Verify the body tag uses the modern system font stack."""
        stats = {"total_scans": 10, "flagged_incidents": 2, "avg_similarity": 0.45}
        html = generate_daily_summary_html(stats)

        # The exact font stack required by Issue #2576
        expected_stack = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

        # Check if the font stack is present in the body style
        assert expected_stack in html, "Body tag must use the robust system font stack"

    def test_headings_use_system_font_stack(self):
        """Verify h1 and h2 tags also use the system font stack for consistency."""
        stats = {
            "total_scans": 10,
            "flagged_incidents": 2,
            "avg_similarity": 0.45,
            "top_pairs": [],
        }
        html = generate_daily_summary_html(stats)

        expected_stack = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

        # Count occurrences - should be in body, h1, and h2 at minimum
        assert html.count(expected_stack) >= 3

    def test_no_standalone_arial_sans_serif(self):
        """Verify the old 'Arial, sans-serif' is completely replaced."""
        stats = {"total_scans": 10, "flagged_incidents": 2, "avg_similarity": 0.45}
        html = generate_daily_summary_html(stats)

        # Regex to find "font-family: Arial, sans-serif;" NOT preceded by the system stack
        # We want to ensure the old hardcoded string is gone
        old_pattern = re.compile(r"font-family:\s*Arial,\s*sans-serif\s*;")

        # If the old pattern exists, it should ONLY be as part of the larger system stack
        # Let's just assert the exact old string without the system prefix is not there
        assert "font-family: Arial, sans-serif;" not in html

    def test_font_stack_includes_apple_system(self):
        """Verify -apple-system is the first font for macOS/iOS optimization."""
        stats = {"total_scans": 1}
        html = generate_daily_summary_html(stats)

        # Extract font-family declarations
        font_families = re.findall(r"font-family:\s*([^;]+);", html)

        for family in font_families:
            # Every font-family declaration should start with -apple-system
            assert family.strip().startswith(
                "-apple-system"
            ), f"Font stack should start with -apple-system, got: {family}"

    def test_font_stack_includes_segoe_ui(self):
        """Verify 'Segoe UI' is included for Windows optimization."""
        stats = {"total_scans": 1}
        html = generate_daily_summary_html(stats)

        assert "Segoe UI" in html

    def test_font_stack_includes_roboto(self):
        """Verify 'Roboto' is included for Android optimization."""
        stats = {"total_scans": 1}
        html = generate_daily_summary_html(stats)

        assert "Roboto" in html


class TestEmailDataInterpolation:
    """Test suite for correct data rendering in the email template."""

    def test_total_scans_rendered_with_commas(self):
        """Verify large numbers are formatted with comma separators."""
        stats = {"total_scans": 1234567, "flagged_incidents": 0, "avg_similarity": 0.0}
        html = generate_daily_summary_html(stats)

        assert "1,234,567" in html

    def test_flagged_incidents_rendered(self):
        """Verify flagged incidents count is rendered correctly."""
        stats = {"total_scans": 100, "flagged_incidents": 42, "avg_similarity": 0.5}
        html = generate_daily_summary_html(stats)

        assert ">42<" in html or ">42\n" in html

    def test_avg_similarity_rendered_as_percentage(self):
        """Verify average similarity is formatted as a percentage."""
        stats = {"total_scans": 10, "flagged_incidents": 1, "avg_similarity": 0.854}
        html = generate_daily_summary_html(stats)

        assert "85.4%" in html

    def test_top_pairs_rendered_in_table(self):
        """Verify top pairs are rendered as table rows."""
        stats = {
            "total_scans": 10,
            "flagged_incidents": 2,
            "avg_similarity": 0.75,
            "top_pairs": [
                {"doc_a": "essay1.pdf", "doc_b": "wiki.pdf", "similarity": 0.95},
                {"doc_a": "essay2.pdf", "doc_b": "source.docx", "similarity": 0.88},
            ],
        }
        html = generate_daily_summary_html(stats)

        assert "essay1.pdf" in html
        assert "wiki.pdf" in html
        assert "95.0%" in html
        assert "essay2.pdf" in html
        assert "88.0%" in html

    def test_empty_top_pairs_shows_fallback_message(self):
        """Verify fallback message is shown when no top pairs exist."""
        stats = {
            "total_scans": 10,
            "flagged_incidents": 0,
            "avg_similarity": 0.10,
            "top_pairs": [],
        }
        html = generate_daily_summary_html(stats)

        assert "No high-similarity pairs detected today" in html

    def test_top_pairs_limited_to_five(self):
        """Verify only the top 5 pairs are rendered even if more are provided."""
        top_pairs = [
            {
                "doc_a": f"doc{i}.pdf",
                "doc_b": f"src{i}.pdf",
                "similarity": 0.99 - (i * 0.01),
            }
            for i in range(10)
        ]
        stats = {
            "total_scans": 100,
            "flagged_incidents": 10,
            "avg_similarity": 0.90,
            "top_pairs": top_pairs,
        }
        html = generate_daily_summary_html(stats)

        # First 5 should be present
        for i in range(5):
            assert f"doc{i}.pdf" in html

        # 6th through 10th should NOT be present
        for i in range(5, 10):
            assert f"doc{i}.pdf" not in html


class TestEmailStructureAndAccessibility:
    """Test suite for HTML structure, roles, and accessibility."""

    def test_tables_have_presentation_role(self):
        """Verify layout tables use role='presentation' for screen readers."""
        stats = {
            "total_scans": 10,
            "flagged_incidents": 1,
            "avg_similarity": 0.5,
            "top_pairs": [],
        }
        html = generate_daily_summary_html(stats)

        # All layout tables should have role="presentation"
        # The data table for top pairs should NOT have it (or should have role="table")
        presentation_count = html.count('role="presentation"')
        assert presentation_count >= 3  # Main wrapper, stats grid, etc.

    def test_html_has_lang_attribute(self):
        """Verify the html tag has lang='en' for accessibility."""
        stats = {"total_scans": 1}
        html = generate_daily_summary_html(stats)

        assert '<html lang="en">' in html

    def test_meta_viewport_present(self):
        """Verify viewport meta tag is present and prepended in <head> for mobile responsiveness."""
        stats = {"total_scans": 1}
        html = generate_daily_summary_html(stats)

        assert 'name="viewport"' in html
        assert "width=device-width" in html

        head_content = html.split("<head>")[1].split("</head>")[0]
        viewport_idx = head_content.find('<meta name="viewport"')
        charset_idx = head_content.find('<meta charset=')
        assert viewport_idx != -1, "Viewport meta tag must be in <head>"
        assert viewport_idx < charset_idx, "Viewport meta tag must be prepended before charset in <head>"

    def test_build_email_html_body_meta_viewport_prepended(self):
        """Verify viewport meta tag is present and prepended in build_email_html_body <head>."""
        html = build_email_html_body(incidents_data=[], total_scans=10)
        assert '<meta name="viewport" content="width=device-width, initial-scale=1.0">' in html
        head_content = html.split("<head>")[1].split("</head>")[0]
        viewport_idx = head_content.find('<meta name="viewport"')
        charset_idx = head_content.find('<meta charset=')
        assert viewport_idx != -1
        assert viewport_idx < charset_idx

    def test_report_date_rendered(self):
        """Verify the current date is rendered in the header."""
        stats = {"total_scans": 1}
        html = generate_daily_summary_html(stats)

        # Should contain the current year at minimum
        current_year = str(datetime.now().year)
        assert current_year in html

    def test_missing_stats_keys_default_to_zero(self):
        """Verify missing keys in stats dict default to 0/empty gracefully."""
        # Pass completely empty dict
        html = generate_daily_summary_html({})

        assert ">0<" in html or ">0\n" in html
        assert "0.0%" in html
        assert "No high-similarity pairs detected today" in html


# ---------------------------------------------------------------------------
# Tests for Dynamic Subject Line Formatting (#3444)
# ---------------------------------------------------------------------------


class TestDynamicSubjectFormatting:
    """Test suite for dynamic email subject line formatting tokens (#3444)."""

    def test_default_subject_template_constant(self):
        """Verify DEFAULT_EMAIL_SUBJECT_TEMPLATE constant has expected format."""
        assert DEFAULT_EMAIL_SUBJECT_TEMPLATE == "Daily Plagiarism Summary - {date} ({count} incidents)"

    @patch.dict("os.environ", {}, clear=True)
    def test_default_subject_formatting(self):
        """Verify default formatting without custom env vars or prefix."""
        subject = format_subject_line(date="2026-08-25", count=7)
        assert subject == "Daily Plagiarism Summary - 2026-08-25 (7 incidents)"

    @patch.dict("os.environ", {}, clear=True)
    def test_default_subject_formatting_with_prefix(self):
        """Verify default formatting with standard subject prefix."""
        subject = format_subject_line(
            date="2026-08-25", count=4, subject_prefix="[Plagiarism Alert]"
        )
        assert subject == "[Plagiarism Alert] Daily Plagiarism Summary - 2026-08-25 (4 incidents)"

    @patch.dict(
        "os.environ",
        {
            "EMAIL_SUBJECT_TEMPLATE": "[{app_name}] Daily Digest - {date}: {count} flagged",
            "APP_NAME": "EthicsGuard",
        },
        clear=True,
    )
    def test_env_subject_template_dynamic_tokens(self):
        """Verify EMAIL_SUBJECT_TEMPLATE environment variable formats {date}, {count}, and {app_name}."""
        subject = format_subject_line(date="2026-08-25", count=15, subject_prefix="")
        assert subject == "[EthicsGuard] Daily Digest - 2026-08-25: 15 flagged"

    def test_custom_template_argument_override(self):
        """Verify passing explicit template overrides env var and defaults."""
        custom_template = "Report for {app_name} on {date} (Total: {count})"
        subject = format_subject_line(
            template=custom_template,
            date="2026-08-25",
            count=2,
            app_name="Physics Department",
        )
        assert subject == "Report for Physics Department on 2026-08-25 (Total: 2)"

    def test_template_safe_handling_of_unknown_tokens(self):
        """Verify template formatting does not crash when unrecognised tokens are present."""
        template_with_extra = "Daily Report: {count} issues ({date}) [{course_id}]"
        subject = format_subject_line(
            template=template_with_extra, date="2026-08-25", count=3
        )
        assert "3 issues" in subject
        assert "2026-08-25" in subject
        assert "{course_id}" in subject

    @patch("src.utils.daily_summary_email.send_email")
    @patch("src.utils.daily_summary_email.get_admin_emails")
    @patch("src.utils.daily_summary_email.get_incidents_last_24h")
    @patch.dict(
        "os.environ",
        {
            "EMAIL_SUBJECT_TEMPLATE": "Plagiarism Summary for {date} ({count} items)",
        },
    )
    def test_send_daily_summary_uses_env_subject_template(
        self, mock_get_incidents, mock_get_emails, mock_send_email, mock_incidents
    ):
        """Verify send_daily_summary dynamically formats subject line with incident count."""
        mock_get_incidents.return_value = mock_incidents  # 3 incidents
        mock_get_emails.return_value = ["admin@example.com"]
        mock_send_email.return_value = True

        result = send_daily_summary(subject_prefix="[Alert]")

        assert result is True
        mock_send_email.assert_called_once()
        sent_subject = mock_send_email.call_args[0][1]
        assert sent_subject.startswith("[Alert] Plagiarism Summary for")
        assert "(3 items)" in sent_subject

    @patch("src.utils.daily_summary_email.send_email")
    @patch("src.utils.daily_summary_email.get_admin_emails")
    @patch("src.utils.daily_summary_email.get_incidents_last_24h")
    def test_send_daily_summary_custom_subject_template_param(
        self, mock_get_incidents, mock_get_emails, mock_send_email, mock_incidents
    ):
        """Verify send_daily_summary accepts explicit subject_template parameter."""
        mock_get_incidents.return_value = mock_incidents[:2]  # 2 incidents
        mock_get_emails.return_value = ["admin@example.com"]
        mock_send_email.return_value = True

        result = send_daily_summary(
            subject_prefix="",
            subject_template="Summary: {count} findings on {date}",
        )

        assert result is True
        mock_send_email.assert_called_once()
        sent_subject = mock_send_email.call_args[0][1]
        assert sent_subject.startswith("Summary: 2 findings on")


# ---------------------------------------------------------------------------
# Tests for Incident CSV Attachment Option (#3445)
# ---------------------------------------------------------------------------


class TestIncidentCsvAttachment:
    """Test suite for incident CSV report attachment functionality (#3445)."""

    def test_export_incidents_to_csv_content(self, mock_incidents):
        """Verify export_incidents_to_csv generates valid CSV byte stream with data."""
        csv_bytes = export_incidents_to_csv(mock_incidents)
        assert isinstance(csv_bytes, bytes)
        assert len(csv_bytes) > 0

        # Decode and inspect CSV text
        csv_text = csv_bytes.decode("utf-8-sig")
        lines = csv_text.strip().splitlines()

        assert len(lines) == 4  # Header + 3 incident rows
        header = lines[0]
        assert "Incident ID" in header
        assert "Document A" in header
        assert "Document B" in header
        assert "Similarity Score" in header
        assert "Severity Rank" in header

        assert "INC-123" in lines[1]
        assert "student1.pdf" in lines[1]
        assert "0.9500" in lines[1]

    def test_export_incidents_to_csv_empty(self):
        """Verify export_incidents_to_csv on empty list generates header-only CSV."""
        csv_bytes = export_incidents_to_csv([])
        assert isinstance(csv_bytes, bytes)

        csv_text = csv_bytes.decode("utf-8-sig")
        lines = csv_text.strip().splitlines()
        assert len(lines) == 1  # Only header line
        assert "Incident ID" in lines[0]

    def test_export_incidents_to_csv_utf8_sig_bom(self, mock_incidents):
        """Verify export_incidents_to_csv starts with UTF-8 BOM for Excel compatibility."""
        csv_bytes = export_incidents_to_csv(mock_incidents)
        assert csv_bytes.startswith(b"\xef\xbb\xbf")

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
    def test_send_email_with_attach_csv_true(self, mock_smtp, mock_incidents):
        """Verify send_email includes CSV MIMEApplication attachment when attach_csv=True."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        csv_bytes = export_incidents_to_csv(mock_incidents)
        result = send_email(
            ["recipient@example.com"],
            "Daily Summary",
            "<p>Summary Body</p>",
            attach_csv=True,
            csv_data=csv_bytes,
            attachment_filename="custom_summary.csv",
        )

        assert result is True
        sent_msg = mock_server.send_message.call_args[0][0]

        attachments = [
            part
            for part in sent_msg.walk()
            if part.get_content_disposition() == "attachment"
        ]
        assert len(attachments) == 1
        assert attachments[0].get_filename() == "custom_summary.csv"
        assert attachments[0].get_payload(decode=True) == csv_bytes

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
    def test_send_email_with_attach_csv_false(self, mock_smtp):
        """Verify send_email omits attachment when attach_csv=False."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        result = send_email(
            ["recipient@example.com"],
            "Daily Summary",
            "<p>Summary Body</p>",
            attach_csv=False,
        )

        assert result is True
        sent_msg = mock_server.send_message.call_args[0][0]

        attachments = [
            part
            for part in sent_msg.walk()
            if part.get_content_disposition() == "attachment"
        ]
        assert len(attachments) == 0

    @patch("src.utils.daily_summary_email.send_email")
    @patch("src.utils.daily_summary_email.get_admin_emails")
    @patch("src.utils.daily_summary_email.get_incidents_last_24h")
    def test_send_daily_summary_with_attach_csv_default(
        self, mock_get_incidents, mock_get_emails, mock_send_email, mock_incidents
    ):
        """Verify send_daily_summary forwards attach_csv=True by default with generated CSV."""
        mock_get_incidents.return_value = mock_incidents
        mock_get_emails.return_value = ["admin@example.com"]
        mock_send_email.return_value = True

        result = send_daily_summary()

        assert result is True
        mock_send_email.assert_called_once()
        _, kwargs = mock_send_email.call_args
        assert kwargs.get("attach_csv") is True
        csv_data = kwargs.get("csv_data")
        assert isinstance(csv_data, bytes)
        assert b"INC-123" in csv_data

    @patch("src.utils.daily_summary_email.send_email")
    @patch("src.utils.daily_summary_email.get_admin_emails")
    @patch("src.utils.daily_summary_email.get_incidents_last_24h")
    def test_send_daily_summary_with_attach_csv_false(
        self, mock_get_incidents, mock_get_emails, mock_send_email, mock_incidents
    ):
        """Verify send_daily_summary respects attach_csv=False."""
        mock_get_incidents.return_value = mock_incidents
        mock_get_emails.return_value = ["admin@example.com"]
        mock_send_email.return_value = True

        result = send_daily_summary(attach_csv=False)

        assert result is True
        mock_send_email.assert_called_once()
        _, kwargs = mock_send_email.call_args
        assert kwargs.get("attach_csv") is False
        assert kwargs.get("csv_data") is None

    def test_export_incidents_to_csv_large_volume_over_20(self):
        """Verify exporting > 20 incidents includes all rows for offline spreadsheet analysis."""
        many_incidents = [
            {
                "incident_id": f"INC-{i:03d}",
                "document_a": f"doc_a_{i}.pdf",
                "document_b": f"doc_b_{i}.pdf",
                "similarity_score": 0.80,
                "severity_rank": "High",
                "review_status": "Pending",
                "date_flagged": "2026-08-25",
            }
            for i in range(50)
        ]
        csv_bytes = export_incidents_to_csv(many_incidents)
        csv_text = csv_bytes.decode("utf-8-sig")
        lines = csv_text.strip().splitlines()

        assert len(lines) == 51  # Header + 50 rows
        for i in range(50):
            assert f"INC-{i:03d}" in csv_text


# ---------------------------------------------------------------------------
# Unit tests for plain-text MIME alternative summary emails (#3450)
# ---------------------------------------------------------------------------


class TestPlainTextMimeSummaryEmail:
    """Test suite verifying plain-text MIME alternative generation and attachment (#3450)."""

    def test_build_email_text_body_empty_incidents(self):
        """Verify plain text structure when no incidents occurred."""
        text = build_email_text_body(incidents_data=[], total_scans=42)
        assert "DAILY PLAGIARISM SUMMARY" in text
        assert "No new plagiarism incidents detected in the last 24 hours." in text
        assert "Total scans processed: 42" in text

    def test_build_email_text_body_with_custom_footer_note(self):
        """Verify plain text format appends administrator footer note."""
        note = "Please review before Monday."
        text = build_email_text_body(incidents_data=[], total_scans=10, footer_note=note)
        assert "Note from Administrator:" in text
        assert note in text

    def test_build_email_text_body_with_populated_incidents(self):
        """Verify plain text formatting lists all severity groups and incident details clearly."""
        incidents = [
            {
                "document_a": "Essay1.docx",
                "document_b": "Essay2.docx",
                "similarity_score": 0.952,
                "severity_rank": "High",
                "date_flagged": "2026-08-25 10:00:00",
            },
            {
                "document_a": "LabReportA.pdf",
                "document_b": "LabReportB.pdf",
                "similarity_score": 0.725,
                "severity_rank": "Medium",
                "date_flagged": "2026-08-25 11:30:00",
            },
            {
                "document_a": "ProjectAlpha.pdf",
                "document_b": "ProjectBeta.pdf",
                "similarity_score": 0.421,
                "severity_rank": "Low",
                "date_flagged": "2026-08-25 14:15:00",
            },
        ]
        text = build_email_text_body(incidents, total_scans=150, footer_note="Review needed.")

        assert "DAILY PLAGIARISM SUMMARY" in text
        assert "Total new incidents: 3" in text
        assert "Total scans processed: 150" in text
        assert "- High: 1" in text
        assert "- Medium: 1" in text
        assert "- Low: 1" in text
        assert "--- HIGH SEVERITY INCIDENTS (1) ---" in text
        assert "* Document A: Essay1.docx" in text
        assert "Document B: Essay2.docx" in text
        assert "Similarity: 95.20%" in text
        assert "Date Flagged: 2026-08-25 10:00:00" in text
        assert "--- MEDIUM SEVERITY INCIDENTS (1) ---" in text
        assert "* Document A: LabReportA.pdf" in text
        assert "--- LOW SEVERITY INCIDENTS (1) ---" in text
        assert "* Document A: ProjectAlpha.pdf" in text
        assert "Note from Administrator:\nReview needed." in text
        assert "Review all incidents in the dashboard: http://localhost:8501" in text

    @patch("smtplib.SMTP")
    def test_send_email_attaches_both_plain_and_html_mime_parts(self, mock_smtp):
        """Verify send_email attaches both MIMEText('text/plain') and MIMEText('text/html') in MIMEMultipart('alternative')."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        plain_content = "Plain text version of the summary"
        html_content = "<p>HTML version of the summary</p>"

        with patch.dict(
            "os.environ",
            {
                "SMTP_SERVER": "smtp.example.com",
                "SMTP_PORT": non_integer_port,
                "SMTP_USERNAME": "user@example.com",
                "SMTP_PASSWORD": "password",
            },
        ):
            res = send_email(
                to_emails=["admin@example.com"],
                subject="Daily Summary",
                html_body=html_content,
                text_body=plain_content,
                attach_csv=False,
            )

            assert res is True
            mock_server.send_message.assert_called_once()
            msg = mock_server.send_message.call_args[0][0]

            assert msg.is_multipart()
            assert msg.get_content_type() == "multipart/alternative"

            payloads = msg.get_payload()
            assert len(payloads) == 2

            plain_part = payloads[0]
            html_part = payloads[1]

            assert plain_part.get_content_type() == "text/plain"
            assert plain_part.get_payload(decode=True).decode("utf-8") == plain_content

            assert html_part.get_content_type() == "text/html"
            assert html_part.get_payload(decode=True).decode("utf-8") == html_content

    @patch("smtplib.SMTP")
    def test_send_email_with_csv_attachment_and_both_mime_parts(self, mock_smtp):
        """Verify container contains plain part, html part, and CSV application/octet-stream attachment."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        with patch.dict(
            "os.environ",
            {
                "SMTP_SERVER": "smtp.example.com",
                "SMTP_PORT": "587",
                "SMTP_USERNAME": "user@example.com",
                "SMTP_PASSWORD": "password",
            },
        ):
            res = send_email(
                to_emails=["admin@example.com"],
                subject="Daily Summary with CSV",
                html_body="<h1>HTML</h1>",
                text_body="Plain text",
                attach_csv=True,
                csv_data=b"col1,col2\nval1,val2",
                attachment_filename="incidents_report.csv",
            )

            assert res is True
            msg = mock_server.send_message.call_args[0][0]
            parts = list(msg.walk())

            content_types = [p.get_content_type() for p in parts]
            assert "text/plain" in content_types
            assert "text/html" in content_types
            assert "application/csv" in content_types or "application/octet-stream" in content_types

    @patch("src.utils.daily_summary_email.send_email")
    @patch("src.utils.daily_summary_email.get_admin_emails")
    @patch("src.utils.daily_summary_email.get_incidents_last_24h")
    def test_send_daily_summary_integrates_plain_text_body(
        self, mock_get_incidents, mock_get_emails, mock_send_email
    ):
        """Verify send_daily_summary builds and provides text_body to send_email."""
        mock_get_incidents.return_value = [
            {
                "incident_id": "INC-01",
                "document_a": "docA.txt",
                "document_b": "docB.txt",
                "similarity_score": 0.89,
                "severity_rank": "High",
                "date_flagged": "2026-08-25",
            }
        ]
        mock_get_emails.return_value = ["admin@example.com"]
        mock_send_email.return_value = True

        result = send_daily_summary(footer_note="High alert")
        assert result is True
        mock_send_email.assert_called_once()
        _, kwargs = mock_send_email.call_args

        text_body = kwargs.get("text_body")
        assert text_body is not None
        assert "DAILY PLAGIARISM SUMMARY" in text_body
        assert "docA.txt" in text_body
        assert "High alert" in text_body

    def test_build_email_text_body_handles_missing_incident_fields(self):
        """Verify missing incident dictionary keys do not cause KeyError and default cleanly in plain text."""
        incidents = [
            {"similarity_score": 0.5},
            {"document_a": "OnlyA.pdf"},
        ]
        text = build_email_text_body(incidents, total_scans=10)
        assert "Unknown" in text
        assert "OnlyA.pdf" in text
        assert "50.00%" in text

    def test_build_email_text_body_custom_base_url(self):
        """Verify APP_BASE_URL environment variable is reflected in plain text dashboard link."""
        with patch.dict("os.environ", {"APP_BASE_URL": "https://plagiarism.university.edu"}):
            text = build_email_text_body(
                incidents_data=[
                    {
                        "document_a": "A.pdf",
                        "document_b": "B.pdf",
                        "similarity_score": 0.9,
                        "severity_rank": "High",
                    }
                ],
                total_scans=50,
            )
            assert "https://plagiarism.university.edu" in text

    def test_build_email_text_body_all_severity_types_empty(self):
        """Verify each empty severity rank displays the appropriate 'No ... detected' line."""
        incidents = [
            {
                "document_a": "OnlyLow.docx",
                "document_b": "Ref.docx",
                "similarity_score": 0.25,
                "severity_rank": "Low",
            }
        ]
        text = build_email_text_body(incidents, total_scans=20)
        assert "No high severity incidents detected." in text
        assert "No medium severity incidents detected." in text
        assert "* Document A: OnlyLow.docx" in text

    def test_build_email_text_body_many_incidents_plain_list(self):
        """Verify large volume of incidents is rendered cleanly in plain text."""
        incidents = [
            {
                "document_a": f"doc_a_{i}.pdf",
                "document_b": f"doc_b_{i}.pdf",
                "similarity_score": 0.75,
                "severity_rank": "Medium" if i % 2 == 0 else "High",
                "date_flagged": f"2026-08-25 12:{i:02d}:00",
            }
            for i in range(25)
        ]
        text = build_email_text_body(incidents, total_scans=500)
        assert "Total new incidents: 25" in text
        assert "Total scans processed: 500" in text
        for i in range(25):
            assert f"doc_a_{i}.pdf" in text

    @patch("smtplib.SMTP")
    def test_send_email_default_text_body_none_attaches_only_html(self, mock_smtp):
        """Verify backward compatibility: if text_body is None, send_email attaches html MIMEText without plain text."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        with patch("src.utils.daily_summary_email.logger.warning") as mock_warn, patch.dict(
            "os.environ",
            {
                "SMTP_SERVER": "smtp.example.com",
                "SMTP_PORT": "587.5",
                "SMTP_USERNAME": "user@example.com",
                "SMTP_PASSWORD": "password",
            },
        ):
            res = send_email(
                to_emails=["admin@example.com"],
                subject="HTML Only",
                html_body="<p>Just HTML</p>",
                text_body=None,
                attach_csv=False,
            )
            assert res is True
            msg = mock_server.send_message.call_args[0][0]
            payloads = msg.get_payload()
            assert len(payloads) == 1
            assert payloads[0].get_content_type() == "text/html"

    @patch("smtplib.SMTP")
    def test_send_email_both_mime_parts_unicode_payload(self, mock_smtp):
        """Verify UTF-8 characters in both plain text and HTML payloads are properly preserved in MIME parts."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        plain_text = "Unicode test: 🔴 High severity ⚠️ Café résumé 測試"
        html_text = "<p>Unicode test: 🔴 High severity ⚠️ Café résumé 測試</p>"

        with patch.dict(
            "os.environ",
            {
                "SMTP_SERVER": "smtp.example.com",
                "SMTP_PORT": "0",
                "SMTP_USERNAME": "user@example.com",
                "SMTP_PASSWORD": "password",
            },
        ):
            res = send_email(
                to_emails=["admin@example.com"],
                subject="Unicode Test",
                html_body=html_text,
                text_body=plain_text,
                attach_csv=False,
            )
            assert res is True
            msg = mock_server.send_message.call_args[0][0]
            parts = msg.get_payload()
            assert len(parts) == 2
            assert parts[0].get_payload(decode=True).decode("utf-8") == plain_text
            assert parts[1].get_payload(decode=True).decode("utf-8") == html_text




