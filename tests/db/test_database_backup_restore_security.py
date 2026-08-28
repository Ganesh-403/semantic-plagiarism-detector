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

import os
import sqlite3
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from src.db.database_backup import (
    BackupRestoreSecurityError,
    restore,
    restore_database_backup,
)


def create_database(path: Path, value: str) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE records (value TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO records (value) VALUES (?)",
            (value,),
        )
        connection.commit()


def read_value(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        row = connection.execute("SELECT value FROM records").fetchone()
    assert row is not None
    return row[0]


def test_restore_authorized_backup(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    source = backup_dir / "valid.db"
    destination = tmp_path / "corpus.db"
    create_database(source, "restored")
    create_database(destination, "old")

    result = restore(
        "valid.db",
        backup_dir=backup_dir,
        destination=destination,
    )

    assert result == destination.resolve()
    assert read_value(destination) == "restored"


def test_restore_alias_uses_same_secure_path(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    source = backup_dir / "valid.db"
    destination = tmp_path / "corpus.db"
    create_database(source, "restored")

    result = restore_database_backup(
        source,
        backup_dir=backup_dir,
        destination=destination,
    )

    assert result == destination.resolve()
    assert read_value(destination) == "restored"


def test_rejects_parent_directory_traversal(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    outside = tmp_path / "outside.db"
    create_database(outside, "unsafe")

    with pytest.raises(
        BackupRestoreSecurityError,
        match="inside the designated backup directory",
    ):
        restore(
            "../outside.db",
            backup_dir=backup_dir,
            destination=tmp_path / "corpus.db",
        )


def test_rejects_absolute_path_outside_backup_dir(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    outside = tmp_path / "outside.db"
    create_database(outside, "unsafe")

    with pytest.raises(BackupRestoreSecurityError):
        restore(
            outside,
            backup_dir=backup_dir,
            destination=tmp_path / "corpus.db",
        )


def test_rejects_symlink_escape(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    outside = tmp_path / "outside.db"
    create_database(outside, "unsafe")
    link = backup_dir / "linked.db"

    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Creating symlinks is unavailable on this platform.")

    with pytest.raises(BackupRestoreSecurityError):
        restore(
            link,
            backup_dir=backup_dir,
            destination=tmp_path / "corpus.db",
        )


def test_rejects_world_writable_backup(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    source = backup_dir / "unsafe.db"
    create_database(source, "unsafe")

    current_mode = source.stat().st_mode
    os.chmod(source, current_mode | stat.S_IWOTH)

    with patch(
        "src.db.database_backup.os.stat",
        wraps=os.stat,
    ) as mocked_stat:
        with pytest.raises(
            BackupRestoreSecurityError,
            match="world-writable",
        ):
            restore(
                source,
                backup_dir=backup_dir,
                destination=tmp_path / "corpus.db",
            )

    mocked_stat.assert_called()


def test_rejects_directory_source(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    directory_source = backup_dir / "not-a-db"
    directory_source.mkdir()

    with pytest.raises(
        BackupRestoreSecurityError,
        match="regular file",
    ):
        restore(
            directory_source,
            backup_dir=backup_dir,
            destination=tmp_path / "corpus.db",
        )


def test_rejects_invalid_sqlite_before_destination_change(
    tmp_path,
):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    source = backup_dir / "invalid.db"
    source.write_bytes(b"not a sqlite database")
    destination = tmp_path / "corpus.db"
    create_database(destination, "original")
    before = destination.read_bytes()

    with pytest.raises(
        sqlite3.DatabaseError,
        match="not a valid SQLite",
    ):
        restore(
            source,
            backup_dir=backup_dir,
            destination=destination,
        )

    assert destination.read_bytes() == before


def test_atomic_replace_failure_preserves_destination(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    source = backup_dir / "valid.db"
    destination = tmp_path / "corpus.db"
    create_database(source, "new")
    create_database(destination, "original")

    with patch(
        "src.db.database_backup.os.replace",
        side_effect=OSError("replace failed"),
    ):
        with pytest.raises(OSError, match="replace failed"):
            restore(
                source,
                backup_dir=backup_dir,
                destination=destination,
            )

    assert read_value(destination) == "original"
    assert not list(destination.parent.glob(f".{destination.name}.restore-*.tmp"))


def test_source_and_destination_must_differ(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    source = backup_dir / "same.db"
    create_database(source, "value")

    with pytest.raises(
        BackupRestoreSecurityError,
        match="must differ",
    ):
        restore(
            source,
            backup_dir=backup_dir,
            destination=source,
        )
