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
test_cli_downgrade.py
---------------------
Tests for CLI database downgrade command (`python src/cli.py db downgrade`)
and `migration_history` tracking.
"""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cli import main, run_db_downgrade
from src.db.migrations import (
    AUTH_SCHEMA_VERSION,
    CORPUS_SCHEMA_VERSION,
    get_latest_applied_migration,
    get_user_version,
    migrate_auth_database,
    migrate_corpus_database,
    rollback_migration,
)


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def test_migration_history_table_created_and_populated_on_run_migrations(tmp_path):
    """Verify run_migrations records each version in migration_history."""
    db_file = tmp_path / "corpus_history.db"
    with connect(db_file) as conn:
        migrate_corpus_database(conn)

        # Inspect migration_history table
        rows = conn.execute(
            "SELECT version, applied_at, description FROM migration_history ORDER BY version"
        ).fetchall()

        assert len(rows) == CORPUS_SCHEMA_VERSION
        versions = [r[0] for r in rows]
        assert versions == list(range(1, CORPUS_SCHEMA_VERSION + 1))


def test_get_latest_applied_migration_reads_history_and_user_version(tmp_path):
    """Verify get_latest_applied_migration retrieves top version from history."""
    db_file = tmp_path / "auth_history.db"
    with connect(db_file) as conn:
        migrate_auth_database(conn)
        latest = get_latest_applied_migration(conn)
        assert latest == AUTH_SCHEMA_VERSION


def test_rollback_migration_deletes_from_migration_history(tmp_path):
    """Verify rolling back a migration removes it from migration_history table."""
    db_file = tmp_path / "rollback_history.db"
    with connect(db_file) as conn:
        migrate_corpus_database(conn)

        latest_before = get_latest_applied_migration(conn)
        assert latest_before == CORPUS_SCHEMA_VERSION

        # Roll back 1 version
        from src.db.migrations import CORPUS_DOWN_MIGRATIONS

        rollback_migration(
            conn,
            target_version=CORPUS_SCHEMA_VERSION - 1,
            down_migrations=CORPUS_DOWN_MIGRATIONS,
        )

        latest_after = get_latest_applied_migration(conn)
        assert latest_after == CORPUS_SCHEMA_VERSION - 1
        assert get_user_version(conn) == CORPUS_SCHEMA_VERSION - 1


def test_run_db_downgrade_reverts_recent_migration(tmp_path):
    """Test run_db_downgrade function reverts most recent applied migration."""
    db_file = tmp_path / "cli_corpus.db"
    with connect(db_file) as conn:
        migrate_corpus_database(conn)

    with connect(db_file) as conn:
        initial_version = get_user_version(conn)
    assert initial_version == CORPUS_SCHEMA_VERSION

    # Run downgrade
    exit_code = run_db_downgrade(db_path=db_file, db_type="corpus", steps=1)
    assert exit_code == 0

    with connect(db_file) as conn:
        new_version = get_user_version(conn)
    assert new_version == CORPUS_SCHEMA_VERSION - 1


def test_run_db_downgrade_auth_database(tmp_path):
    """Test run_db_downgrade function on auth database."""
    db_file = tmp_path / "cli_auth.db"
    with connect(db_file) as conn:
        migrate_auth_database(conn)

    with connect(db_file) as conn:
        initial_version = get_user_version(conn)
    assert initial_version == AUTH_SCHEMA_VERSION

    # Run downgrade
    exit_code = run_db_downgrade(db_path=db_file, db_type="auth", steps=1)
    assert exit_code == 0

    with connect(db_file) as conn:
        new_version = get_user_version(conn)
    assert new_version == AUTH_SCHEMA_VERSION - 1


def test_run_db_downgrade_multiple_steps(tmp_path):
    """Test run_db_downgrade with multiple steps."""
    db_file = tmp_path / "cli_multi.db"
    with connect(db_file) as conn:
        migrate_corpus_database(conn)

    exit_code = run_db_downgrade(db_path=db_file, db_type="corpus", steps=3)
    assert exit_code == 0

    with connect(db_file) as conn:
        new_version = get_user_version(conn)
    assert new_version == CORPUS_SCHEMA_VERSION - 3


def test_run_db_downgrade_nonexistent_database(tmp_path, capsys):
    """Test run_db_downgrade on a non-existent database file."""
    db_file = tmp_path / "nonexistent.db"
    exit_code = run_db_downgrade(db_path=db_file, db_type="corpus", steps=1)
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_run_db_downgrade_already_at_version_zero(tmp_path, capsys):
    """Test run_db_downgrade on an empty database already at version 0."""
    db_file = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db_file))
    conn.close()

    exit_code = run_db_downgrade(db_path=db_file, db_type="corpus", steps=1)
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "already at version 0" in captured.out


def test_cli_db_downgrade_command(tmp_path):
    """Test invoking `python src/cli.py db downgrade` via main()."""
    db_file = tmp_path / "main_cli.db"
    with connect(db_file) as conn:
        migrate_corpus_database(conn)

    test_args = [
        "src/cli.py",
        "db",
        "downgrade",
        "--database",
        str(db_file),
        "--db-type",
        "corpus",
    ]

    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    with connect(db_file) as conn:
        new_ver = get_user_version(conn)
    assert new_ver == CORPUS_SCHEMA_VERSION - 1
