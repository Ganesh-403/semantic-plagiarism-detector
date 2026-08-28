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
tests/utils/test_processing_time_benchmark.py
---------------------------------------------
Micro-benchmark tests for ProcessingTimer.time_block overhead (Issue #1864).

Asserts that the context manager overhead is negligible:
  1. 10,000 timer context block executions complete in under 1 second.
  2. Per-block overhead is under 0.1ms (100 microseconds).
"""

import time

from src.utils.processing_time import ProcessingTimer


class TestProcessingTimerBenchmark:
    """Benchmark tests for ProcessingTimer.time_block overhead (Issue #1864)."""

    def test_10000_time_block_executions_under_1_second(self):
        """10,000 timer context block executions must complete in under 1 second.

        This is the Definition of Done from the issue. The test creates a
        fresh ProcessingTimer and enters/exits 10,000 time_block context
        managers with empty bodies, then asserts the total wall-clock
        time is under 1 second.
        """
        timer = ProcessingTimer()

        start = time.perf_counter()

        for _ in range(10_000):
            with timer.time_block("bench"):
                pass  # No-op — we're measuring pure overhead.

        elapsed = time.perf_counter() - start

        # The issue's Definition of Done: under 1 second for 10,000 blocks.
        assert elapsed < 1.0, (
            f"10,000 time_block executions took {elapsed:.4f}s, "
            f"exceeding the 1.0s budget."
        )

    def test_single_time_block_overhead_under_0_1ms(self):
        """Per-block overhead must be under 0.1ms (100 microseconds).

        Measures the average overhead across 1,000 iterations to get a
        stable reading, then asserts the average is under 0.1ms.
        """
        timer = ProcessingTimer()
        iterations = 1_000

        start = time.perf_counter()

        for _ in range(iterations):
            with timer.time_block("micro"):
                pass

        elapsed = time.perf_counter() - start
        avg_overhead_ms = (elapsed / iterations) * 1_000  # seconds → ms

        # 0.1ms = 100 microseconds per block.
        assert avg_overhead_ms < 0.1, (
            f"Average time_block overhead was {avg_overhead_ms:.4f}ms, "
            f"exceeding the 0.1ms threshold."
        )

    def test_nested_time_block_overhead_under_0_1ms(self):
        """Nested time_block overhead must also be under 0.1ms per block.

        Measures the overhead of a 5-level-deep nested call to ensure
        the parent-child bookkeeping doesn't add significant cost.
        """
        timer = ProcessingTimer()
        iterations = 1_000

        start = time.perf_counter()

        for _ in range(iterations):
            with timer.time_block("level1"):
                with timer.time_block("level2"):
                    with timer.time_block("level3"):
                        with timer.time_block("level4"):
                            with timer.time_block("level5"):
                                pass

        elapsed = time.perf_counter() - start
        avg_overhead_ms = (elapsed / (iterations * 5)) * 1_000  # 5 blocks per iteration

        assert avg_overhead_ms < 0.1, (
            f"Average nested time_block overhead was {avg_overhead_ms:.4f}ms, "
            f"exceeding the 0.1ms threshold."
        )

    def test_time_block_records_correct_duration(self):
        """Benchmark should not break the timing functionality.

        Verifies that a time_block with a real sleep still records
        the correct duration (within a tolerance).
        """
        timer = ProcessingTimer()

        with timer.time_block("sleep_test"):
            time.sleep(0.05)  # 50ms

        summary = timer.get_summary()
        assert "sleep_test" in summary
        assert summary["sleep_test"] >= 0.04  # Allow ~10ms tolerance
        assert summary["sleep_test"] <= 0.15  # Upper bound sanity check

    def test_time_block_aggregation_does_not_degrade(self):
        """Aggregating many blocks should not cause performance degradation.

        Runs 5,000 blocks and verifies the last 1,000 are not slower
        than the first 1,000 (i.e., no O(n²) growth in the _aggregate_stats
        defaultdict).
        """
        timer = ProcessingTimer()

        # First 1,000.
        start1 = time.perf_counter()
        for _ in range(1_000):
            with timer.time_block("batch1"):
                pass
        elapsed1 = time.perf_counter() - start1

        # Middle 3,000 (to build up the aggregate).
        for _ in range(3_000):
            with timer.time_block("batch2"):
                pass

        # Last 1,000.
        start2 = time.perf_counter()
        for _ in range(1_000):
            with timer.time_block("batch3"):
                pass
        elapsed2 = time.perf_counter() - start2

        # The last batch should not be more than 3x slower than the first
        # (generous tolerance to account for GC/scheduling jitter).
        assert elapsed2 < elapsed1 * 3, (
            f"Performance degraded: first 1,000 took {elapsed1:.4f}s, "
            f"last 1,000 took {elapsed2:.4f}s (ratio {elapsed2/elapsed1:.2f}x)."
        )
