"""scripts/repair_db.py
----------------------
Admin helper for corpus.db and users.db health checks and light repair.

Usage:
  python scripts/repair_db.py --check
  python scripts/repair_db.py --vacuum
  python scripts/repair_db.py --reindex
  python scripts/repair_db.py --check --vacuum --reindex
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keep in sync with src.core.app_config (CORPUS_DB_PATH / AUTH_DB_PATH).
DEFAULT_CORPUS_DB = REPO_ROOT / "data" / "corpus.db"
DEFAULT_USERS_DB = REPO_ROOT / "users.db"


def _format_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.2f} MiB"


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def cleanup_orphaned_chunks(conn: sqlite3.Connection) -> int:
    """Remove chunks whose filename is not present in documents."""
    if not (_table_exists(conn, "chunks") and _table_exists(conn, "documents")):
        return 0

    deleted = conn.execute(
        """
        DELETE FROM chunks
        WHERE filename NOT IN (SELECT filename FROM documents)
        """
    ).rowcount
    conn.commit()
    return max(deleted, 0)


def run_check(path: Path, label: str) -> bool:
    print(f"\n=== CHECK: {label} ({path}) ===")
    if not path.exists():
        print("  status: MISSING (file not found)")
        return False

    print(f"  size:   {_format_size(path)}")
    ok = True
    with _connect(path) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"  integrity_check: {integrity}")
        if integrity != "ok":
            ok = False

        if label == "corpus.db":
            removed = cleanup_orphaned_chunks(conn)
            print(f"  orphaned chunks removed: {removed}")

        tables = _list_tables(conn)
        print(f"  tables ({len(tables)}): {', '.join(tables) if tables else '(none)'}")
        for table in tables:
            try:
                count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                print(f"    - {table}: {count} rows")
            except sqlite3.Error as exc:
                print(f"    - {table}: error ({exc})")
                ok = False

    print(f"  result: {'OK' if ok else 'ISSUES FOUND'}")
    return ok


def run_vacuum(path: Path, label: str) -> bool:
    print(f"\n=== VACUUM: {label} ({path}) ===")
    if not path.exists():
        print("  status: MISSING (skipped)")
        return False

    before = _format_size(path)
    with _connect(path) as conn:
        conn.execute("VACUUM")
    after = _format_size(path)
    print(f"  size before: {before}")
    print(f"  size after:  {after}")
    print("  result: OK")
    return True


def run_reindex(path: Path, label: str) -> bool:
    print(f"\n=== REINDEX: {label} ({path}) ===")
    if not path.exists():
        print("  status: MISSING (skipped)")
        return False

    with _connect(path) as conn:
        conn.execute("REINDEX")
    print("  result: OK")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose and repair corpus.db / users.db SQLite databases."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run integrity_check, print table stats, and clean orphaned chunks.",
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM to reclaim unused space.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild all indexes with REINDEX.",
    )
    parser.add_argument(
        "--corpus-db",
        type=Path,
        default=DEFAULT_CORPUS_DB,
        help=f"Path to corpus.db (default: {DEFAULT_CORPUS_DB})",
    )
    parser.add_argument(
        "--users-db",
        type=Path,
        default=DEFAULT_USERS_DB,
        help=f"Path to users.db (default: {DEFAULT_USERS_DB})",
    )
    args = parser.parse_args()

    if not (args.check or args.vacuum or args.reindex):
        parser.error("Specify at least one of --check, --vacuum, --reindex")

    targets = [
        (args.corpus_db, "corpus.db"),
        (args.users_db, "users.db"),
    ]

    print("Database repair utility")
    print(f"  corpus: {args.corpus_db}")
    print(f"  users:  {args.users_db}")

    all_ok = True
    for path, label in targets:
        if args.check and not run_check(path, label):
            all_ok = False
        if args.vacuum and not run_vacuum(path, label):
            all_ok = False
        if args.reindex and not run_reindex(path, label):
            all_ok = False

    print("\nDone.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
