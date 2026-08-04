import sqlite3
import uuid
from datetime import datetime, timezone

import pytest

from src.db.auth import (
    _connect,
    add_user,
    init_db,
    update_password,
    verify_user,
)
from src.db.migrations.auth import (
    AUTH_SCHEMA_VERSION,
    migration_001_create_users,
    migration_010_add_password_changed_at,
)
from src.db.migrations.common import (
    column_exists,
    get_user_version,
)


@pytest.fixture(autouse=True)
def initialize_auth_database(mock_db):
    init_db()
    yield


def unique_username() -> str:
    return f"password_age_{uuid.uuid4().hex[:10]}"


def read_password_changed_at(username: str):
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT password_changed_at
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
    assert row is not None
    return row[0]


def test_auth_schema_version_is_incremented():
    assert AUTH_SCHEMA_VERSION == 10


def test_password_changed_at_column_exists_after_init():
    with _connect() as connection:
        assert column_exists(
            connection,
            "users",
            "password_changed_at",
        )
        assert get_user_version(connection) == 10


def test_migration_adds_nullable_text_column(tmp_path):
    database = tmp_path / "users.db"

    with sqlite3.connect(database) as connection:
        migration_001_create_users(connection)
        migration_010_add_password_changed_at(connection)

        columns = {
            row[1]: row
            for row in connection.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
        }

    column = columns["password_changed_at"]
    assert column[2].upper() == "TEXT"
    assert column[3] == 0
    assert column[4] is None


def test_migration_is_idempotent(tmp_path):
    database = tmp_path / "users.db"

    with sqlite3.connect(database) as connection:
        migration_001_create_users(connection)
        migration_010_add_password_changed_at(connection)
        migration_010_add_password_changed_at(connection)

        matching_columns = [
            row
            for row in connection.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
            if row[1] == "password_changed_at"
        ]

    assert len(matching_columns) == 1


def test_new_user_starts_without_password_change_timestamp():
    username = unique_username()
    add_user(username, "OriginalPassword1!")

    assert read_password_changed_at(username) is None


def test_successful_password_change_updates_timestamp():
    username = unique_username()
    add_user(username, "OriginalPassword1!")
    before = datetime.now(timezone.utc)

    update_password(
        username,
        "UpdatedPassword2@",
    )

    raw_timestamp = read_password_changed_at(username)
    assert raw_timestamp is not None

    changed_at = datetime.fromisoformat(raw_timestamp)
    after = datetime.now(timezone.utc)

    assert changed_at.tzinfo is not None
    assert before <= changed_at <= after
    assert verify_user(
        username,
        "UpdatedPassword2@",
    ) is True


def test_second_password_change_advances_timestamp(
    monkeypatch,
):
    username = unique_username()
    add_user(username, "OriginalPassword1!")

    class FirstDateTime:
        @classmethod
        def now(cls, tz):
            return datetime(
                2026,
                8,
                1,
                10,
                0,
                tzinfo=timezone.utc,
            )

    class SecondDateTime:
        @classmethod
        def now(cls, tz):
            return datetime(
                2026,
                8,
                2,
                10,
                0,
                tzinfo=timezone.utc,
            )

    monkeypatch.setattr(
        "src.db.auth.datetime",
        FirstDateTime,
    )
    update_password(
        username,
        "UpdatedPassword2@",
    )
    first = read_password_changed_at(username)

    monkeypatch.setattr(
        "src.db.auth.datetime",
        SecondDateTime,
    )
    update_password(
        username,
        "UpdatedPassword3#",
    )
    second = read_password_changed_at(username)

    assert datetime.fromisoformat(second) > (
        datetime.fromisoformat(first)
    )


def test_unknown_user_does_not_create_timestamp():
    username = unique_username()

    with pytest.raises(ValueError, match="User not found"):
        update_password(
            username,
            "UpdatedPassword2@",
        )

    with _connect() as connection:
        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()[0]

    assert count == 0


def test_authorization_failure_does_not_change_timestamp():
    target = unique_username()
    actor = unique_username()
    add_user(target, "TargetPassword1!")
    add_user(actor, "ActorPassword1!", role="teacher")

    assert read_password_changed_at(target) is None

    with pytest.raises(PermissionError):
        update_password(
            target,
            "UpdatedPassword2@",
            current_user=actor,
        )

    assert read_password_changed_at(target) is None
    assert verify_user(
        target,
        "TargetPassword1!",
    ) is True
