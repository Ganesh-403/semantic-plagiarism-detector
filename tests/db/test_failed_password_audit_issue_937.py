import json
import sqlite3
import uuid

import pytest

from src.db.auth import (
    _connect,
    add_user,
    init_db,
    update_password,
    verify_user,
)


@pytest.fixture(autouse=True)
def setup_database(mock_db):
    init_db()
    yield


def unique_username():
    return f"audit_{uuid.uuid4().hex[:10]}"


def failed_events(username):
    with _connect() as connection:
        return connection.execute(
            """
            SELECT event_type, username, details
            FROM security_audit_log
            WHERE username = ?
              AND event_type = 'password_change_failed'
            ORDER BY id
            """,
            (username,),
        ).fetchall()


def test_incorrect_old_password_creates_failure_event():
    username = unique_username()
    add_user(username, "OldPassword1!")

    with pytest.raises(
        ValueError,
        match="Current password is incorrect",
    ):
        update_password(
            username,
            "NewPassword2@",
            old_password="WrongPassword9!",
        )

    events = failed_events(username)
    assert len(events) == 1
    assert json.loads(events[0][2]) == {
        "reason": "incorrect_old_password"
    }
    assert verify_user(username, "OldPassword1!") is True
    assert verify_user(username, "NewPassword2@") is False


@pytest.mark.parametrize(
    "weak_password",
    [
        "short",
        "lowercase1!",
        "NoNumber!",
        "NoSpecial1",
    ],
)
def test_complexity_failure_is_audited(weak_password):
    username = unique_username()
    add_user(username, "OldPassword1!")

    with pytest.raises(ValueError):
        update_password(
            username,
            weak_password,
            old_password="OldPassword1!",
        )

    events = failed_events(username)
    assert len(events) == 1
    assert json.loads(events[0][2]) == {
        "reason": "complexity_failed"
    }
    assert verify_user(username, "OldPassword1!") is True


def test_unknown_user_failure_is_audited():
    username = unique_username()

    with pytest.raises(ValueError, match="User not found"):
        update_password(
            username,
            "NewPassword2@",
        )

    events = failed_events(username)
    assert len(events) == 1
    assert json.loads(events[0][2]) == {
        "reason": "user_not_found"
    }


def test_unauthorized_foreign_change_is_audited():
    username = unique_username()
    actor = unique_username()
    add_user(username, "OldPassword1!")
    add_user(actor, "ActorPassword1!", role="teacher")

    with pytest.raises(PermissionError):
        update_password(
            username,
            "NewPassword2@",
            current_user=actor,
        )

    events = failed_events(username)
    assert len(events) == 1
    assert json.loads(events[0][2]) == {
        "reason": "unauthorized_actor"
    }


def test_success_does_not_create_failure_event():
    username = unique_username()
    add_user(username, "OldPassword1!")

    update_password(
        username,
        "NewPassword2@",
        old_password="OldPassword1!",
    )

    assert failed_events(username) == []
    assert verify_user(username, "NewPassword2@") is True


def test_audit_details_never_contain_passwords():
    username = unique_username()
    old_password = "OldPassword1!"
    wrong_password = "WrongPassword9!"
    new_password = "NewPassword2@"
    add_user(username, old_password)

    with pytest.raises(ValueError):
        update_password(
            username,
            new_password,
            old_password=wrong_password,
        )

    details = failed_events(username)[0][2]
    assert old_password not in details
    assert wrong_password not in details
    assert new_password not in details


def test_database_failure_is_audited(
    monkeypatch,
    username=None,
):
    username = unique_username()
    add_user(username, "OldPassword1!")

    class FailingConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("database unavailable")

    original_connect = __import__(
        "src.db.auth",
        fromlist=["_connect"],
    )._connect
    calls = {"count": 0}

    def selective_connect():
        calls["count"] += 1
        if calls["count"] == 1:
            return FailingConnection()
        return original_connect()

    monkeypatch.setattr(
        "src.db.auth._connect",
        selective_connect,
    )

    with pytest.raises(sqlite3.Error):
        update_password(
            username,
            "NewPassword2@",
        )

    events = failed_events(username)
    assert len(events) == 1
    assert json.loads(events[0][2]) == {
        "reason": "database_error"
    }
