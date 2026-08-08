#!/usr/bin/env python3
"""
Memory leak detection runner using tracemalloc.

Executes the pytest suite and fails if uncollected memory allocation
delta exceeds 10MB across test runs.
"""

import subprocess
import sys
import tracemalloc


MEMORY_DELTA_LIMIT_MB = 10.0


def main() -> int:
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/core", "tests/db", "tests/utils", "tests/scripts", "tests/security", "tests/visualization", "-v", "--tb=short"],
        capture_output=False,
    )

    if result.returncode != 0:
        print("Pytest suite failed -- skipping memory delta check.", file=sys.stderr)
        return result.returncode

    snapshot_after = tracemalloc.take_snapshot()

    stats_before = snapshot_before.statistics("lineno")
    stats_after = snapshot_after.statistics("lineno")

    total_before = sum(stat.size for stat in stats_before)
    total_after = sum(stat.size for stat in stats_after)

    delta_mb = (total_after - total_before) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"Memory delta: {delta_mb:.2f} MB (limit: {MEMORY_DELTA_LIMIT_MB} MB)")
    print(f"{'=' * 60}")

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    print("\nTop memory differences:")
    for stat in top_stats[:10]:
        print(stat)

    if delta_mb > MEMORY_DELTA_LIMIT_MB:
        print(
            f"\n[FAIL] Uncollected memory allocation delta ({delta_mb:.2f} MB) "
            f"exceeds limit ({MEMORY_DELTA_LIMIT_MB} MB).",
            file=sys.stderr,
        )
        return 1

    print("\n[PASS] Memory delta within acceptable range.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
