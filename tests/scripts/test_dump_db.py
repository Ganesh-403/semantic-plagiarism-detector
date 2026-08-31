"""
tests/scripts/test_dump_db.py
-----------------------------
Tests for the database dump utility (scripts/dump_db.py).

Verifies:
- The script creates valid .sql dump files.
- The dump contains CREATE TABLE and INSERT statements.
- A dumped database can be restored to a new SQLite file.
- Non-existent databases are skipped gracefully.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the dump_db.py script
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dump_db.py"


@pytest.fixture
def temp_sqlite_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database with a small table."""
    db_path = tmp_path / "test_source.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO sample (name) VALUES ('alice')")
    conn.execute("INSERT INTO sample (name) VALUES ('bob')")
    conn.commit()
    conn.close()
    return db_path


class TestDumpDatabase:
    """Tests for the dump_database function."""

    def test_dump_creates_sql_file(self, tmp_path: Path):
        """Verify that dump_database creates a .sql file."""
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        from dump_db import dump_database

        src = tmp_path / "src.db"
        conn = sqlite3.connect(str(src))
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        conn.close()

        out = tmp_path / "out.sql"
        result = dump_database(src, out)

        assert result is True
        assert out.exists()
        content = out.read_text()
        assert "CREATE TABLE" in content
        assert "INSERT INTO" in content

    def test_dump_skips_nonexistent_db(self, tmp_path: Path):
        """Verify that a missing DB is skipped gracefully."""
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        from dump_db import dump_database

        src = tmp_path / "nonexistent.db"
        out = tmp_path / "out.sql"
        result = dump_database(src, out)

        assert result is False
        assert not out.exists()

    def test_dump_can_be_restored(self, tmp_path: Path):
        """Verify that a dump can be replayed into a new database."""
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        from dump_db import dump_database

        src = tmp_path / "src.db"
        conn = sqlite3.connect(str(src))
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO users (name) VALUES ('test_user')")
        conn.commit()
        conn.close()

        dump_path = tmp_path / "dump.sql"
        dump_database(src, dump_path)

        restored = tmp_path / "restored.db"
        restore_conn = sqlite3.connect(str(restored))
        dump_sql = dump_path.read_text()
        restore_conn.executescript(dump_sql)
        restore_conn.commit()

        row = restore_conn.execute(
            "SELECT name FROM users WHERE id = 1"
        ).fetchone()
        restore_conn.close()

        assert row is not None
        assert row[0] == "test_user"

    def test_dump_contains_header(self, tmp_path: Path):
        """Verify the dump file has a header comment."""
        sys.path.insert(0, str(SCRIPT_PATH.parent))
        from dump_db import dump_database

        src = tmp_path / "src.db"
        conn = sqlite3.connect(str(src))
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
        conn.close()

        out = tmp_path / "out.sql"
        dump_database(src, out)

        content = out.read_text()
        assert content.startswith("-- SQLite dump")
        assert "Generated:" in content


class TestDumpDbCLI:
    """Integration tests for the script's CLI interface."""

    def test_script_runs_without_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Run the script as a subprocess and verify it exits 0."""
        corpus_db = tmp_path / "corpus.db"
        auth_db = tmp_path / "users.db"

        for db_path in (corpus_db, auth_db):
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE dummy (id INTEGER)")
            conn.execute("INSERT INTO dummy VALUES (1)")
            conn.commit()
            conn.close()

        monkeypatch.setattr(
            "src.core.app_config.CORPUS_DB_PATH", corpus_db
        )
        monkeypatch.setattr(
            "src.core.app_config.AUTH_DB_PATH", auth_db
        )

        output_dir = tmp_path / "dumps"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_PATH.parents[1]),
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_dir.exists()

        sql_files = list(output_dir.glob("*.sql"))
        assert len(sql_files) >= 2

    def test_script_handles_missing_dbs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Verify the script exits with error when no DBs exist."""
        monkeypatch.setattr(
            "src.core.app_config.CORPUS_DB_PATH",
            tmp_path / "nonexistent_corpus.db",
        )
        monkeypatch.setattr(
            "src.core.app_config.AUTH_DB_PATH",
            tmp_path / "nonexistent_auth.db",
        )

        output_dir = tmp_path / "dumps"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPT_PATH.parents[1]),
        )

        assert result.returncode == 1
        assert "No databases were dumped" in result.stdout
