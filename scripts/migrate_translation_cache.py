"""scripts/migrate_translation_cache.py
---------------------------------------
One-off migration for the legacy SQLite translation cache.

Usage:
  python scripts/migrate_translation_cache.py [--legacy-db PATH] [--cache-db PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.db.translation_cache import migrate_legacy_cache

logger = logging.getLogger("translation_cache_migration")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy_translation_cache rows into translation_cache."
    )
    parser.add_argument(
        "--legacy-db",
        type=Path,
        default=None,
        help="Legacy SQLite database path (defaults to the configured corpus DB).",
    )
    parser.add_argument(
        "--cache-db",
        type=Path,
        default=None,
        help="Modern translation-cache database path (defaults to data/translation_cache.db).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    stats = migrate_legacy_cache(
        legacy_db_path=args.legacy_db,
        cache_db_path=args.cache_db,
    )

    logger.info("Migration summary:")
    logger.info("  Scanned:  %d", stats["scanned"])
    logger.info("  Migrated: %d", stats["migrated"])
    logger.info("  Skipped:  %d", stats["skipped"])
    logger.info("  Errors:   %d", stats["errors"])

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
