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

"""
tests/utils/test_smtp_auth_logging_issue_3448.py
------------------------------------------------
Tests for actionable error logging on SMTP authentication failures (Issue #3448).

Verifies that send_email():
* catches smtplib.SMTPAuthenticationError specifically,
* does NOT retry (auth failures are permanent),
* logs an actionable message advising verification of SMTP_USER/SMTP_USERNAME,
  SMTP_PASSWORD, or generating an App Password when 2FA is enabled,
* reports the failure through the optional status_callback.
"""

import smtplib
from unittest.mock import MagicMock, patch

import pytest

from src.utils.daily_summary_email import send_email

SMTP_ENV = {
    "SMTP_SERVER": "smtp.example.com",
    "SMTP_PORT": "587",
    "SMTP_USERNAME": "test@example.com",
    "SMTP_PASSWORD": "password",
    "FROM_EMAIL": "test@example.com",
}


@pytest.fixture
def auth_error():
    """Build a realistic SMTPAuthenticationError like Gmail's 535 reply."""
    return smtplib.SMTPAuthenticationError(
        535, b"535-5.7.8 Username and Password not accepted"
    )


@patch("time.sleep")
@patch("smtplib.SMTP")
@patch.dict("os.environ", SMTP_ENV, clear=True)
def test_auth_error_fails_immediately_with_actionable_message(
    mock_smtp, mock_sleep, auth_error
):
    """STARTTLS path: auth error must not retry and must log guidance."""
    callback = MagicMock()
    mock_server = MagicMock()
    mock_server.login.side_effect = auth_error
    mock_smtp.return_value.__enter__.return_value = mock_server

    with patch("src.utils.daily_summary_email.logger.error") as log_error:
        result = send_email(
            ["recipient@example.com"],
            "Test Subject",
            "<p>Body</p>",
            status_callback=callback,
        )

    assert result is False

    # Authentication failures are permanent: no retries, no backoff sleeps.
    assert mock_smtp.call_count == 1
    mock_sleep.assert_not_called()

    # The logged error is the actionable message from Issue #3448.
    assert log_error.call_count == 1
    logged = log_error.call_args[0][0]
    assert "SMTP authentication failed" in logged
    assert "test@example.com" in logged
    assert "535" in logged
    assert "Username and Password not accepted" in logged
    assert "SMTP_USER/SMTP_USERNAME" in logged
    assert "SMTP_PASSWORD" in logged
    assert "two-factor authentication" in logged.lower()
    assert "App Password" in logged

    callback.assert_called_once()
    success, message = callback.call_args[0]
    assert success is False
    assert "App Password" in message


@patch("time.sleep")
@patch("smtplib.SMTP_SSL")
@patch.dict("os.environ", {**SMTP_ENV, "SMTP_PORT": "465"}, clear=True)
def test_auth_error_handled_on_ssl_port_465(mock_smtp_ssl, mock_sleep, auth_error):
    """SMTP_SSL (port 465) path gets the same actionable treatment."""
    callback = MagicMock()
    mock_server = MagicMock()
    mock_server.login.side_effect = auth_error
    mock_smtp_ssl.return_value.__enter__.return_value = mock_server

    with patch("src.utils.daily_summary_email.logger.error") as log_error:
        result = send_email(
            ["recipient@example.com"],
            "Test Subject",
            "<p>Body</p>",
            status_callback=callback,
        )

    assert result is False
    assert mock_smtp_ssl.call_count == 1
    mock_sleep.assert_not_called()

    logged = log_error.call_args[0][0]
    assert "SMTP authentication failed" in logged
    assert "App Password" in logged


@patch("time.sleep")
@patch("smtplib.SMTP")
@patch.dict("os.environ", SMTP_ENV, clear=True)
def test_generic_smtp_exception_still_retries(mock_smtp, mock_sleep):
    """Non-auth SMTP errors keep the existing retry/backoff behavior."""
    mock_conn = MagicMock()
    mock_conn.__enter__.side_effect = smtplib.SMTPException("transient outage")
    mock_smtp.return_value = mock_conn

    result = send_email(["recipient@example.com"], "Test Subject", "<p>Body</p>")

    assert result is False
    assert mock_smtp.call_count == 4  # Initial attempt + 3 retries.
    assert mock_sleep.call_count == 3
