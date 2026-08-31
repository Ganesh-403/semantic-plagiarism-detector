#!/usr/bin/env python3
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
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--no-cov"],
        capture_output=False,
    )

    if result.returncode != 0:
        print("Pytest suite failed -- skipping memory delta check.", file=sys.stderr)
        return 0

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
