#!/usr/bin/env python3
"""
scripts/dump_db.py
------------------
Administrative script to export all SQLite database tables to SQL dump
files (.sql) for disaster recovery.

Uses SQLite's built-in ``connection.iterdump()`` which generates a
textual representation of the database that can be replayed via
``sqlite3 < dump.sql`` or ``.read dump.sql`` from the sqlite3 CLI.

Usage:
    # Dump both corpus.db and users.db to ./backups/
    python scripts/dump_db.py

    # Specify a custom output directory
    python scripts/dump_db.py --output-dir /tmp/dumps

    # Dump only the corpus database
    python scripts/dump_db.py --db corpus

    # Dump only the auth database
    python scripts/dump_db.py --db auth

Acceptance Criteria (Issue #4062):
- Create scripts/dump_db.py utilizing connection.iterdump().
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Ensure the project root is on sys.path so ``src.*`` imports resolve
# when running this script directly (e.g. ``python scripts/dump_db.py``).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.app_config import AUTH_DB_PATH, CORPUS_DB_PATH  # noqa: E402


def dump_database(db_path: Path, output_path: Path) -> bool:
    """Dump a single SQLite database to a .sql file.

    Args:
        db_path: Path to the source .db file.
        output_path: Path to write the .sql dump.

    Returns:
        True if the dump was created, False if the source DB does not exist.
    """
    if not db_path.exists():
        print(f"  [SKIP] Database not found: {db_path}")
        return False

    # Connect in read-only mode to avoid locking issues with a live DB.
    source_uri = f"{db_path.as_uri()}?mode=ro"
    conn = sqlite3.connect(source_uri, uri=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # Write a header comment
            f.write(f"-- SQLite dump for {db_path.name}\n")
            f.write(f"-- Generated: {datetime.now().isoformat()}\n")
            f.write(f"-- Source: {db_path}\n")
            f.write("--\n\n")

            # iterdump() yields SQL statements as strings
            for line in conn.iterdump():
                f.write(line + "\n")

        size_kb = output_path.stat().st_size / 1024
        print(
            f"  [OK] {db_path.name} → {output_path.name} "
            f"({size_kb:.1f} KB)"
        )
        return True

    except Exception as exc:
        print(f"  [ERROR] Failed to dump {db_path.name}: {exc}")
        return False
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SQLite databases to .sql dump files for "
        "disaster recovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backups",
        help="Directory to write .sql dump files (default: backups/)",
    )
    parser.add_argument(
        "--db",
        choices=["corpus", "auth", "all"],
        default="all",
        help="Which database to dump (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    databases: list[tuple[str, Path]] = []
    if args.db in ("corpus", "all"):
        databases.append(("corpus", CORPUS_DB_PATH))
    if args.db in ("auth", "all"):
        databases.append(("auth", AUTH_DB_PATH))

    print(f"Database dump — {timestamp}")
    print(f"Output directory: {output_dir.resolve()}")
    print()

    dumped = 0
    for name, db_path in databases:
        output_file = output_dir / f"{name}_dump_{timestamp}.sql"
        if dump_database(db_path, output_file):
            dumped += 1

    print()
    if dumped == 0:
        print("No databases were dumped. Check that the .db files exist.")
        sys.exit(1)
    else:
        print(
            f"Successfully dumped {dumped} database(s) to "
            f"{output_dir.resolve()}"
        )


if __name__ == "__main__":
    main()
