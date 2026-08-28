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

"""Unit tests for Security Audit Logs Screen in app/streamlit_app.py (#1506)."""

import pandas as pd
import pytest

from src.db.auth import auth_repo, configure_db_path, init_db


@pytest.fixture
def temp_audit_db(tmp_path):
    """Set up an isolated SQLite database for security audit log tests."""
    db_file = tmp_path / "test_audit_ui.db"
    configure_db_path(db_file)
    init_db()

    # Seed test events
    auth_repo.log_security_event("login", "admin", "Admin login successful")
    auth_repo.log_security_event(
        "file_upload", "teacher1", "Uploaded document assignment.pdf"
    )
    auth_repo.log_security_event(
        "password_change", "teacher1", "Password updated successfully"
    )
    auth_repo.log_security_event("token_revocation", "system", "Revoked bearer token")

    yield db_file


def test_audit_logs_query_and_count(temp_audit_db):
    """Verify get_security_audit_logs and count return seeded records correctly."""
    count = auth_repo.get_security_audit_log_count()
    assert count >= 4

    logs = auth_repo.get_security_audit_logs(limit=10)
    assert len(logs) >= 4

    event_types = auth_repo.get_distinct_audit_event_types()
    assert "login" in event_types
    assert "file_upload" in event_types
    assert "password_change" in event_types
    assert "token_revocation" in event_types


def test_audit_logs_username_filter(temp_audit_db):
    """Verify filtering logs by username."""
    teacher_logs = auth_repo.get_security_audit_logs(username="teacher1")
    assert len(teacher_logs) == 2
    for log in teacher_logs:
        assert log["username"] == "teacher1"


def test_audit_logs_event_type_filter(temp_audit_db):
    """Verify filtering logs by event_type."""
    login_logs = auth_repo.get_security_audit_logs(event_type="login")
    assert len(login_logs) == 1
    assert login_logs[0]["event_type"] == "login"
    assert login_logs[0]["username"] == "admin"


def test_audit_logs_csv_export_format(temp_audit_db):
    """Verify converting audit logs to pandas DataFrame and CSV bytes."""
    logs = auth_repo.get_security_audit_logs()
    df = pd.DataFrame(logs)
    csv_str = df.to_csv(index=False)

    assert "event_type" in csv_str
    assert "username" in csv_str
    assert "admin" in csv_str
    assert "teacher1" in csv_str
