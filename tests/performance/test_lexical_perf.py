"""
test_lexical_perf.py
--------------------
Benchmarking utility comparing execution speed (operations per second) of lexical similarity algorithms:
- Jaccard similarity
- Sørensen-Dice coefficient
- Word N-gram overlap
- Character N-gram similarity

Target: 10,000 iterations across representative sample text pairs.
(Issue #4024)
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Any

from src.core.lexical_similarity import (
    compute_char_ngram_similarity,
    dice_coefficient,
    jaccard_similarity,
    n_gram_overlap,
)

logger = logging.getLogger(__name__)

# Sample texts representing typical document paragraphs for plagiarism detection
SAMPLE_TEXT_A = (
    "The quick brown fox jumps over the lazy dog. In computer science, lexical similarity "
    "measures evaluate text overlap using set theory, tokenization, and string matching algorithms."
)
SAMPLE_TEXT_B = (
    "A fast brown fox leaps over a lazy dog. In computer software engineering, lexical similarity "
    "functions quantify string overlap using set theory, tokenization, and n-gram matching."
)

DEFAULT_ITERATIONS = 10_000


def benchmark_algorithm(
    name: str,
    func: Callable[[str, str], float],
    text_a: str,
    text_b: str,
    iterations: int = DEFAULT_ITERATIONS,
) -> Dict[str, Any]:
    """Run a performance benchmark for a similarity algorithm across specified iterations.

    Returns dictionary containing:
    - name: Algorithm name
    - iterations: Total iteration count
    - total_time_sec: Total duration in seconds
    - ops_per_sec: Executed operations per second
    - avg_latency_ms: Average latency per operation in milliseconds
    """
    # Warmup run
    _ = func(text_a, text_b)

    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = func(text_a, text_b)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    ops_per_sec = iterations / total_time if total_time > 0 else 0.0
    avg_latency_ms = (total_time / iterations) * 1000.0 if iterations > 0 else 0.0

    return {
        "name": name,
        "iterations": iterations,
        "total_time_sec": round(total_time, 4),
        "ops_per_sec": round(ops_per_sec, 2),
        "avg_latency_ms": round(avg_latency_ms, 4),
    }


def run_lexical_benchmark(iterations: int = DEFAULT_ITERATIONS) -> Dict[str, Dict[str, Any]]:
    """Execute performance benchmark for all 4 lexical similarity algorithms."""
    algorithms: Dict[str, Callable[[str, str], float]] = {
        "Jaccard": lambda a, b: jaccard_similarity(a, b),
        "Dice": lambda a, b: dice_coefficient(a, b),
        "N-gram": lambda a, b: n_gram_overlap(a, b, n=3),
        "Character N-gram": lambda a, b: compute_char_ngram_similarity(a, b, n=5),
    }

    results: Dict[str, Dict[str, Any]] = {}
    header_line = "=" * 78
    sep_line = "-" * 78

    print(f"\n{header_line}")
    print(f" LEXICAL SIMILARITY BENCHMARK ({iterations:,} iterations)")
    print(header_line)
    print(f"{'Algorithm':<20} | {'Ops / Sec':<15} | {'Total Time (s)':<15} | {'Avg Latency (ms)':<15}")
    print(sep_line)

    for name, func in algorithms.items():
        res = benchmark_algorithm(name, func, SAMPLE_TEXT_A, SAMPLE_TEXT_B, iterations=iterations)
        results[name] = res
        print(
            f"{res['name']:<20} | {res['ops_per_sec']:<15,.2f} | "
            f"{res['total_time_sec']:<15.4f} | {res['avg_latency_ms']:<15.4f}"
        )

    print(f"{header_line}\n")
    return results


def test_lexical_similarity_benchmark_ops_per_sec():
    """Pytest benchmark test comparing execution speed of Jaccard, Dice, N-gram, and Character N-gram across 10,000 iterations."""
    results = run_lexical_benchmark(iterations=10_000)

    # Acceptance Criteria Verification: All 4 algorithms must produce operations per second
    required_algos = ["Jaccard", "Dice", "N-gram", "Character N-gram"]
    for algo in required_algos:
        assert algo in results, f"Algorithm '{algo}' missing from benchmark results"
        ops = results[algo]["ops_per_sec"]
        assert ops > 0, f"Operations per second for '{algo}' must be > 0, got {ops}"


if __name__ == "__main__":
    run_lexical_benchmark(iterations=10_000)
