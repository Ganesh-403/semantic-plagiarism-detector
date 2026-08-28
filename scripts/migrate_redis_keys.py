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
scripts/migrate_redis_keys.py
------------------------------
One-off migration script to scan for legacy cache keys and rename them
to the modern, standardized spd:v1:* namespace format.

Addresses Issue #2803.

Supported legacy prefixes:
  - login_attempts:<id>       -> spd:v1:login_attempts:<id>
  - upload_count:<user>       -> spd:v1:uploads:<user>
  - similarity:<id>           -> spd:v1:analysis:<id>
  - analysis:<id>             -> spd:v1:analysis:<id>
  - doc:<id>                  -> spd:v1:analysis:doc:<id>
  - faiss_index               -> spd:v1:faiss:index:corpus_index
  - faiss_index:<id>          -> spd:v1:faiss:index:<id>
  - session:<id>:<key>        -> spd:v1:session:<id>:<key>

Usage:
  python scripts/migrate_redis_keys.py [--dry-run] [--host HOST] [--port PORT] [--db DB] [--url URL]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional

from src.utils.redis_cache import CacheNamespace

logger = logging.getLogger("redis_migration")


def map_legacy_key(key: str) -> str | None:
    """Map a legacy Redis key to its corresponding spd:v1:* format.

    Args:
        key: The Redis key name as a string.

    Returns:
        The new key string in spd:v1:* format, or None if the key does
        not match a legacy pattern or is already in the new format.
    """
    if key.startswith("spd:v1:"):
        return None

    if key.startswith("login_attempts:"):
        identifier = key[len("login_attempts:") :]
        return CacheNamespace.LOGIN_ATTEMPTS.build_key(identifier)

    if key.startswith("upload_count:"):
        username = key[len("upload_count:") :]
        return CacheNamespace.UPLOADS.build_key(username)

    if key.startswith("similarity:"):
        identifier = key[len("similarity:") :]
        return CacheNamespace.ANALYSIS.build_key(identifier)

    if key.startswith("analysis:"):
        identifier = key[len("analysis:") :]
        return CacheNamespace.ANALYSIS.build_key(identifier)

    if key.startswith("doc:"):
        identifier = key[len("doc:") :]
        return CacheNamespace.ANALYSIS.build_key("doc", identifier)

    if key == "faiss_index":
        return CacheNamespace.FAISS.build_key("index", "corpus_index")

    if key.startswith("faiss_index:"):
        identifier = key[len("faiss_index:") :]
        return CacheNamespace.FAISS.build_key("index", identifier)

    if key.startswith("session:"):
        parts = key[len("session:") :].split(":", 1)
        if len(parts) == 2:
            return CacheNamespace.SESSION.build_key(parts[0], parts[1])
        return CacheNamespace.SESSION.build_key(parts[0])

    return None


def get_redis_client(
    url: str | None = None,
    host: str | None = None,
    port: int | None = None,
    db: int | None = None,
    password: str | None = None,
) -> Any:
    """Create and return a connected Redis client instance."""
    try:
        import redis
    except ImportError:
        logger.error("The 'redis' package is required to run the migration script.")
        raise

    redis_url = url or os.getenv("REDIS_URL")
    if redis_url:
        client = redis.from_url(redis_url, decode_responses=False)
    else:
        client = redis.Redis(
            host=host or os.getenv("REDIS_HOST", "localhost"),
            port=port or int(os.getenv("REDIS_PORT", "6379")),
            db=db if db is not None else int(os.getenv("REDIS_DB", "0")),
            password=password or os.getenv("REDIS_PASSWORD", None),
            decode_responses=False,
        )
    client.ping()
    return client


def migrate_redis_keys(client: Any, dry_run: bool = False) -> dict[str, Any]:
    """Scans all keys in the Redis database and migrates legacy keys to spd:v1:* namespace.

    Args:
        client: Active Redis client instance.
        dry_run: If True, logs planned key renames without performing modifications.

    Returns:
        Dictionary containing summary statistics and list of renamed keys.
    """
    stats: dict[str, Any] = {
        "scanned": 0,
        "migrated": 0,
        "skipped": 0,
        "errors": 0,
        "renames": [],
    }

    if hasattr(client, "scan_iter"):
        key_iter = client.scan_iter(count=1000)
    else:
        key_iter = client.keys("*")

    for raw_key in key_iter:
        stats["scanned"] += 1
        key_str = (
            raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
        )
        new_key_str = map_legacy_key(key_str)

        if not new_key_str or new_key_str == key_str:
            stats["skipped"] += 1
            continue

        new_raw_key = (
            new_key_str.encode("utf-8") if isinstance(raw_key, bytes) else new_key_str
        )

        if dry_run:
            logger.info(f"[DRY-RUN] Would rename: {key_str} -> {new_key_str}")
            stats["migrated"] += 1
            stats["renames"].append((key_str, new_key_str))
        else:
            try:
                client.rename(raw_key, new_raw_key)
                logger.info(f"[MIGRATED] {key_str} -> {new_key_str}")
                stats["migrated"] += 1
                stats["renames"].append((key_str, new_key_str))
            except Exception as e:
                logger.error(
                    f"[ERROR] Failed to rename {key_str} to {new_key_str}: {e}"
                )
                stats["errors"] += 1

    return stats


def main() -> int:
    """CLI entrypoint for the Redis migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate legacy Redis cache keys to the new spd:v1:* namespace format."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without modifying any keys in Redis.",
    )
    parser.add_argument("--url", type=str, default=None, help="Redis connection URL.")
    parser.add_argument(
        "--host", type=str, default=None, help="Redis host (default: localhost)."
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Redis port (default: 6379)."
    )
    parser.add_argument(
        "--db", type=int, default=None, help="Redis database number (default: 0)."
    )
    parser.add_argument("--password", type=str, default=None, help="Redis password.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    try:
        client = get_redis_client(
            url=args.url,
            host=args.host,
            port=args.port,
            db=args.db,
            password=args.password,
        )
    except Exception as e:
        logger.error(f"Could not connect to Redis: {e}")
        return 1

    mode_str = "DRY-RUN" if args.dry_run else "LIVE MIGRATION"
    logger.info(f"Starting Redis key migration ({mode_str})...")

    stats = migrate_redis_keys(client, dry_run=args.dry_run)

    logger.info("=" * 50)
    logger.info(f"Migration completed ({mode_str}):")
    logger.info(f"  Total scanned: {stats['scanned']}")
    logger.info(f"  Migrated:      {stats['migrated']}")
    logger.info(f"  Skipped:       {stats['skipped']}")
    logger.info(f"  Errors:        {stats['errors']}")
    logger.info("=" * 50)

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
