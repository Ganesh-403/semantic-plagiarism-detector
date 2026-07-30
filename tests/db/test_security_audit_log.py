"""Tests for security audit log — password change event recording (Issue #620)."""

from __future__ import annotations

import uuid

import pytest

from src.db.auth import (
    add_user,
    init_db,
    log_security_event,
    update_password,
)
from src.db.auth import _connect


@pytest.fixture(autouse=True)
def setup_test_db(mock_db):
    """Use the shared mock_db fixture to isolate DB operations."""
    init_db()
    yield


def test_log_security_event_inserts_row():
    """log_security_event should write a row into security_audit_log."""
    username = f"user_{uuid.uuid4().hex[:8]}"
    add_user(username, "Password1!")
    log_security_event(event_type="password_change", username=username)
    with _connect() as conn:
        row = conn.execute(
            "SELECT event_type, username FROM security_audit_log WHERE username = ?",
            (username,),
        ).fetchone()
    assert row is not None
    assert row[0] == "password_change"
    assert row[1] == username


def test_log_security_event_stores_timestamp():
    """log_security_event should store a non-empty ISO 8601 UTC timestamp."""
    username = f"user_{uuid.uuid4().hex[:8]}"
    add_user(username, "Password1!")
    log_security_event(event_type="password_change", username=username)
    with _connect() as conn:
        row = conn.execute(
            "SELECT timestamp FROM security_audit_log WHERE username = ?",
            (username,),
        ).fetchone()
    assert row is not None
    timestamp = row[0]
    assert len(timestamp) == 20
    assert timestamp.endswith("Z")
    assert "T" in timestamp


def test_log_security_event_stores_optional_details():
    """log_security_event should persist the details field when provided."""
    username = f"user_{uuid.uuid4().hex[:8]}"
    add_user(username, "Password1!")
    log_security_event(
        event_type="password_change",
        username=username,
        details="Password updated successfully.",
    )
    with _connect() as conn:
        row = conn.execute(
            "SELECT details FROM security_audit_log WHERE username = ?",
            (username,),
        ).fetchone()
    assert row is not None
    assert row[0] == "Password updated successfully."


def test_update_password_creates_audit_log_entry():
    """Calling update_password should create at least one audit log entry."""
    username = f"user_{uuid.uuid4().hex[:8]}"
    add_user(username, "OldPassword1!")
    update_password(username, "NewPassword2@")
    with _connect() as conn:
        rows = conn.execute(
            "SELECT event_type, username FROM security_audit_log WHERE username = ?",
            (username,),
        ).fetchall()
    assert len(rows) >= 1
    assert any(r[0] == "password_change" and r[1] == username for r in rows)


def test_update_password_audit_log_entry_has_timestamp():
    """Audit log entry created by update_password must have a valid timestamp."""
    username = f"user_{uuid.uuid4().hex[:8]}"
    add_user(username, "OldPassword1!")
    update_password(username, "NewPassword2@")
    with _connect() as conn:
        row = conn.execute(
            "SELECT timestamp FROM security_audit_log "
            "WHERE username = ? AND event_type = 'password_change'",
            (username,),
        ).fetchone()
    assert row is not None
    assert row[0].endswith("Z")


def test_update_password_logs_multiple_changes():
    """Each call to update_password should produce a separate audit log entry."""
    username = f"user_{uuid.uuid4().hex[:8]}"
    add_user(username, "FirstPass1!")
    update_password(username, "SecondPass2@")
    update_password(username, "ThirdPass3#")
    with _connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM security_audit_log "
            "WHERE username = ? AND event_type = 'password_change'",
            (username,),
        ).fetchone()[0]
    assert count >= 2
