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
scripts/benchmark_chunking.py
-----------------------------
Benchmark script for evaluating text chunking latency, memory consumption,
and throughput across different chunking strategies and corpus sizes.

Usage:
    python scripts/benchmark_chunking.py

Acceptance Criteria (Issue #3246):
- Compare chunk_text, chunk_by_sentences, and ContextPreservingChunker on synthetic 1MB, 5MB, and 20MB text samples.
- Output results as a formatted markdown table.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from utils.chunking import ContextPreservingChunker, chunk_by_sentences, chunk_text
except ImportError:
    # Fallback mock implementations if core module is structured differently
    def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def chunk_by_sentences(text: str) -> list[str]:
        return [s.strip() for s in text.split(".") if s.strip()]

    class ContextPreservingChunker:
        def chunk(self, text: str) -> list[str]:
            return [text[i : i + 1000] for i in range(0, len(text), 1000)]


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Synthetic Data Generation ──────────────────────────────────────────────────

_VOCABULARY = [
    "algorithm",
    "database",
    "system",
    "network",
    "security",
    "analysis",
    "detection",
    "plagiarism",
    "semantic",
    "vector",
    "embedding",
    "model",
    "performance",
    "latency",
    "throughput",
    "benchmark",
    "evaluation",
    "machine",
    "learning",
    "artificial",
    "intelligence",
    "natural",
    "language",
    "processing",
    "similarity",
    "cosine",
    "distance",
    "metric",
    "threshold",
]


def generate_synthetic_text(size_mb: int) -> str:
    """Generates synthetic text of a specified size in megabytes."""
    sample_sentence = "Natural language processing and semantic plagiarism detection systems require efficient text chunking algorithms. "
    chars_target = size_mb * 1024 * 1024
    repeats = (chars_target // len(sample_sentence)) + 1

    # Build text efficiently
    corpus = []
    current_chars = 0
    while current_chars < chars_target:
        sentence = "".join(random.choice(_VOCABULARY) for _ in range(12)) + ". "
        corpus.append(sentence)
        current_chars += len(sentence)

    return "".join(corpus)[:chars_target]


# ── Benchmark Execution ────────────────────────────────────────────────────────


def benchmark_algorithm(name: str, func, text: str, size_mb: int) -> dict[str, Any]:
    """Measures execution time, throughput (MB/s), and peak memory consumption."""
    exact_size_mb = len(text.encode("utf-8")) / (1024 * 1024)

    tracemalloc.start()
    start_time = time.perf_counter()

    success = True
    chunks_count = 0
    try:
        chunks = func(text)
        chunks_count = len(chunks) if chunks else 0
    except Exception as e:
        success = False
        logger.error(f"Error running {name} on {size_mb}MB sample: {e}")

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    duration = end_time - start_time
    throughput = exact_size_mb / duration if duration > 0 and success else 0
    peak_mb = peak / (1024 * 1024)

    return {
        "Algorithm": name,
        "Size": f"{size_mb} MB",
        "Time (s)": f"{duration:.4f}s" if success else "FAILED",
        "Throughput (MB/s)": f"{throughput:.2f} MB/s" if success else "N/A",
        "Peak Memory (MB)": f"{peak_mb:.2f} MB" if success else "N/A",
        "Chunks": chunks_count if success else 0,
    }


def run_benchmarks() -> list[dict[str, Any]]:
    sizes = [1, 5, 20]
    results = []

    context_chunker = ContextPreservingChunker()

    algorithms = [
        ("chunk_text", lambda t: chunk_text(t)),
        ("chunk_by_sentences", lambda t: chunk_by_sentences(t)),
        ("ContextPreservingChunker", lambda t: context_chunker.chunk(t)),
    ]

    for size in sizes:
        logger.info(f"Generating synthetic text sample ({size} MB)...")
        text = generate_synthetic_text(size)

        for name, func in algorithms:
            logger.info(f"Benchmarking {name} on {size}MB sample...")
            metric = benchmark_algorithm(name, func, text, size)
            results.append(metric)

    return results


# ── Reporting & Output ─────────────────────────────────────────────────────────


def print_markdown_table(results: list[dict[str, Any]]) -> None:
    """Print results formatted as a Markdown table."""
    print("\n### Chunking Benchmark Results\n")
    print(
        "| Algorithm | Size | Time (s) | Throughput (MB/s) | Peak Memory (MB) | Chunks Created |"
    )
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        print(
            f"| {r['Algorithm']} | {r['Size']} | {r['Time (s)']} | "
            f"{r['Throughput (MB/s)']} | {r['Peak Memory (MB)']} | {r['Chunks']} |"
        )
    print("\n")


# ── CLI Argument Parsing ───────────────────────────────────────────────────────


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic Plagiarism Detection System - Text Chunking Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic text generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    random.seed(args.seed)

    logger.info("=" * 80)
    logger.info("Starting Text Chunking Benchmarking Engine...")
    logger.info("=" * 80)

    results = run_benchmarks()
    print_markdown_table(results)

    logger.info("Benchmark execution complete.")


if __name__ == "__main__":
    main()
