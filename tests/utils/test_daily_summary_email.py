"""
test_daily_summary_email.py
---------------------------
Tests for daily summary email functionality and HTML template generation.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch



import pytest

from src.utils.daily_summary_email import (
    build_email_html_body,
    build_incident_row_html,
    build_severity_section_html,
    format_daily_summary,
    get_admin_emails,
    get_incidents_last_24h,
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
    assert "<strong>Total new incidents:</strong> 3" in html
    assert "High: 1" in html
    assert "Medium: 1" in html
    assert "Low: 1" in html
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
def test_send_daily_summary_custom_prefix(mock_get_incidents, mock_get_emails, mock_send_email):
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
    mock_smtp_ssl.assert_called_once_with("smtp.example.com", 465)
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
    mock_smtp.assert_called_once_with("smtp.example.com", 2525)
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
    mock_smtp.assert_called_once_with("smtp.example.com", 587)
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

    def test_build_email_html_body_with_custom_footer_note(self):
        """Test that custom footer note is included in HTML output (#1252)."""
        note = "Please complete all pending reviews before Friday 5 PM."
        html = build_email_html_body(incidents_data=[], total_scans=10, footer_note=note)

        assert "Note from Administrator:" in html
        assert note in html

    def test_build_email_html_body_without_custom_footer_note(self):
        """Test that footer note section is omitted when footer_note is None (#1252)."""
        html = build_email_html_body(incidents_data=[], total_scans=10, footer_note=None)

        assert "Note from Administrator:" not in html

    def test_build_severity_section_html_empty(self):
        """Test severity section generation with no incidents."""
        html = build_severity_section_html("Medium", [])
        assert "No medium severity incidents detected" in html
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

    def test_build_incident_row_html_missing_fields(self):
        """Test row generation handles missing dictionary keys gracefully."""
        inc = {"document_a": "OnlyDocA"}
        html = build_incident_row_html(inc)

        assert "OnlyDocA" in html
        assert "Unknown" in html
        assert "0.00%" in html

    def test_format_daily_summary_backward_compatibility(self):
        """Test that the legacy wrapper still functions correctly."""
        incidents = [{"severity_rank": "Low"}]
        html = format_daily_summary(incidents)
        assert "Daily Plagiarism Summary" in html
        assert "<strong>Total scans processed:</strong> 0" in html

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

        # Test passing custom timeout
        send_email(
            ["recipient@example.com"],
            "Test Subject",
            "<p>Body</p>",
            timeout=15.5
        )
        mock_smtp.assert_called_once_with("smtp.example.com", 587, timeout=15.5)

    with patch("smtplib.SMTP_SSL") as mock_smtp_ssl, patch.dict(
        "os.environ",
        {
            "SMTP_SERVER": "smtp.example.com",
            "SMTP_PORT": "465",
            "SMTP_USERNAME": "test@example.com",
            "SMTP_PASSWORD": "password",
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