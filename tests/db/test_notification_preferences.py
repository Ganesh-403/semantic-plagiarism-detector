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

import json
import sqlite3
from contextlib import contextmanager

import pytest

from src.db import auth


@contextmanager
def temporary_connection(database_path):
    connection = sqlite3.connect(database_path)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture()
def notification_database(tmp_path, monkeypatch):
    database_path = tmp_path / "users.db"

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                preferences TEXT DEFAULT '{}'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO users (username, preferences)
            VALUES (?, ?)
            """,
            ("teacher", "{}"),
        )
        connection.commit()

    monkeypatch.setattr(
        auth,
        "_connect",
        lambda: sqlite3.connect(database_path),
    )
    return database_path


def read_preferences(database_path):
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT preferences
            FROM users
            WHERE username = ?
            """,
            ("teacher",),
        ).fetchone()

    assert row is not None
    return json.loads(row[0])


def test_missing_preferences_use_enabled_defaults(
    notification_database,
):
    assert auth.get_notification_preferences("teacher") == {
        "email_notifications": True,
        "webhook_notifications": True,
    }


def test_update_persists_both_notification_flags(
    notification_database,
):
    result = auth.update_notification_preferences(
        "teacher",
        email_notifications=False,
        webhook_notifications=True,
    )

    assert result == {
        "email_notifications": False,
        "webhook_notifications": True,
    }
    assert read_preferences(notification_database) == result


def test_update_preserves_unrelated_preferences(
    notification_database,
):
    with sqlite3.connect(notification_database) as connection:
        connection.execute(
            """
            UPDATE users
            SET preferences = ?
            WHERE username = ?
            """,
            (
                json.dumps(
                    {
                        "theme": "Dark",
                        "threshold": 0.84,
                    }
                ),
                "teacher",
            ),
        )
        connection.commit()

    auth.update_notification_preferences(
        "teacher",
        email_notifications=True,
        webhook_notifications=False,
    )

    assert read_preferences(notification_database) == {
        "theme": "Dark",
        "threshold": 0.84,
        "email_notifications": True,
        "webhook_notifications": False,
    }


def test_getter_returns_persisted_choices(
    notification_database,
):
    auth.update_notification_preferences(
        "teacher",
        email_notifications=False,
        webhook_notifications=False,
    )

    assert auth.get_notification_preferences("teacher") == {
        "email_notifications": False,
        "webhook_notifications": False,
    }


@pytest.mark.parametrize(
    ("email_value", "webhook_value"),
    [
        (1, True),
        ("yes", True),
        (True, 0),
        (True, None),
    ],
)
def test_non_boolean_values_are_rejected(
    notification_database,
    email_value,
    webhook_value,
):
    with pytest.raises(
        TypeError,
        match="must be a boolean",
    ):
        auth.update_notification_preferences(
            "teacher",
            email_notifications=email_value,
            webhook_notifications=webhook_value,
        )


def test_invalid_stored_values_fall_back_per_key(
    notification_database,
):
    with sqlite3.connect(notification_database) as connection:
        connection.execute(
            """
            UPDATE users
            SET preferences = ?
            WHERE username = ?
            """,
            (
                json.dumps(
                    {
                        "email_notifications": "false",
                        "webhook_notifications": False,
                    }
                ),
                "teacher",
            ),
        )
        connection.commit()

    assert auth.get_notification_preferences("teacher") == {
        "email_notifications": True,
        "webhook_notifications": False,
    }


def test_username_is_normalised_before_update(
    notification_database,
):
    auth.update_notification_preferences(
        "  TEACHER ",
        email_notifications=False,
        webhook_notifications=True,
    )

    assert auth.get_notification_preferences("teacher")["email_notifications"] is False
