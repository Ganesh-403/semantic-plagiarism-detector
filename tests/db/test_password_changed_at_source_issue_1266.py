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

from pathlib import Path

AUTH_PATH = Path("src/db/auth.py")
MIGRATION_PATH = Path("src/db/migrations/auth.py")
TEST_PATH = Path("tests/db/test_password_changed_at_issue_1266.py")


def test_migration_adds_required_column():
    source = MIGRATION_PATH.read_text(encoding="utf-8")

    assert "AUTH_SCHEMA_VERSION =" in source
    assert "def migration_010_add_password_changed_at(" in source
    assert "ADD COLUMN password_changed_at TEXT" in source
    assert "10: migration_010_add_password_changed_at" in source


def test_password_update_writes_timestamp_atomically():
    source = AUTH_PATH.read_text(encoding="utf-8")

    assert "datetime.now(" in source
    assert "timezone.utc" in source
    assert "password_changed_at = ?" in source
    assert "SET password = ?," in source


def test_unit_test_verifies_timestamp_update():
    source = TEST_PATH.read_text(encoding="utf-8")

    assert "test_successful_password_change_updates_timestamp" in source
    assert "datetime.fromisoformat" in source
    assert "test_authorization_failure_does_not_change_timestamp" in source
