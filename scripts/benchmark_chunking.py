#!/usr/bin/env python3
from __future__ import annotations

"""
scripts/benchmark_chunking.py
-----------------------------
Benchmark script for evaluating text chunking latency and throughput.

Generates a synthetic corpus of 10,000 sentences and measures the
execution time of the chunk_documents() function across varying chunk
sizes (250, 500, 1000 characters).

Usage:
    python scripts/benchmark_chunking.py

Acceptance Criteria (Issue #1803):
- Measure chunking throughput (sentences per second) across varying chunk sizes.
- Print formatted timing results summary table.
"""

import argparse
import logging
import random
import string
import sys
import time
from pathlib import Path
from typing import List, Dict

# Add project root to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.text_chunking import chunk_documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Synthetic Data Generation ──────────────────────────────────────────────────

# Vocabulary for generating realistic sentence lengths
_VOCABULARY = [
    "algorithm", "database", "system", "network", "security", "analysis",
    "detection", "plagiarism", "semantic", "vector", "embedding", "model",
    "performance", "latency", "throughput", "benchmark", "evaluation",
    "machine", "learning", "artificial", "intelligence", "natural", "language",
    "processing", "similarity", "cosine", "distance", "metric", "threshold",
]


def generate_synthetic_sentence(min_words: int = 8, max_words: int = 20) -> str:
    """Generate a single synthetic sentence with random word count."""
    num_words = random.randint(min_words, max_words)
    words = [random.choice(_VOCABULARY) for _ in range(num_words)]
    words[0] = words[0].capitalize()
    return " ".join(words) + random.choice([".", "!", "?"])


def generate_synthetic_corpus(num_sentences: int = 10000) -> List[str]:
    """Generate a list of synthetic sentences for benchmarking."""
    logger.info(f"Generating synthetic corpus with {num_sentences} sentences...")
    return [generate_synthetic_sentence() for _ in range(num_sentences)]


# ── Benchmark Execution ────────────────────────────────────────────────────────


def benchmark_chunking(
    sentences: List[str],
    chunk_sizes: List[int],
    overlap: int = 50,
) -> Dict[int, Dict[str, float]]:
    """
    Measure chunking latency and throughput across different chunk sizes.

    Args:
        sentences: List of text sentences to chunk.
        chunk_sizes: List of target chunk sizes (in characters) to test.
        overlap: Number of overlapping characters between chunks.

    Returns:
        Dictionary mapping chunk_size to metrics (time_ms, chunks_created, sentences_per_sec).
    """
    results = {}
    total_chars = sum(len(s) for s in sentences)
    
    logger.info(f"Total corpus size: {total_chars:,} characters across {len(sentences)} sentences.")
    logger.info("Starting chunking benchmarks...")

    for size in chunk_sizes:
        logger.info(f"Testing chunk_size={size}, overlap={overlap}...")
        
        # Join sentences into a single document for chunking
        # (chunk_documents expects a list of document strings)
        full_text = " ".join(sentences)
        
        start_time = time.perf_counter()
        chunks = chunk_documents([full_text], chunk_size=size, chunk_overlap=overlap)
        end_time = time.perf_counter()
        
        elapsed_sec = end_time - start_time
        elapsed_ms = elapsed_sec * 1000
        
        # Calculate throughput
        sentences_per_sec = len(sentences) / elapsed_sec if elapsed_sec > 0 else 0
        chars_per_sec = total_chars / elapsed_sec if elapsed_sec > 0 else 0
        
        results[size] = {
            "time_ms": elapsed_ms,
            "chunks_created": len(chunks) if chunks else 0,
            "sentences_per_sec": sentences_per_sec,
            "chars_per_sec": chars_per_sec,
        }
        
        logger.info(
            f"  -> Completed in {elapsed_ms:.2f} ms. "
            f"Created {len(chunks)} chunks. "
            f"Throughput: {sentences_per_sec:.0f} sentences/sec."
        )

    return results


# ── Reporting & Output ─────────────────────────────────────────────────────────


def print_results_table(results: Dict[int, Dict[str, float]]) -> None:
    """Print a formatted ASCII table of benchmark results."""
    print("\n" + "=" * 80)
    print("  Text Chunking Performance Benchmark Results")
    print("=" * 80)
    print(f"{'Chunk Size':<15} | {'Time (ms)':<15} | {'Chunks':<10} | {'Sentences/sec':<20} | {'Chars/sec':<15}")
    print("-" * 80)
    
    for size, metrics in sorted(results.items()):
        print(
            f"{size:<15} | "
            f"{metrics['time_ms']:>12.2f} | "
            f"{metrics['chunks_created']:>8} | "
            f"{metrics['sentences_per_sec']:>15,.0f} | "
            f"{metrics['chars_per_sec']:>12,.0f}"
        )
        
    print("=" * 80 + "\n")


# ── CLI Argument Parsing ───────────────────────────────────────────────────────


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Semantic Plagiarism Detection System - Text Chunking Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=10000,
        help="Number of synthetic sentences to generate for the benchmark.",
    )
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[250, 500, 1000],
        help="List of chunk sizes (in characters) to test.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Number of overlapping characters between chunks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible corpus generation.",
    )
    
    return parser.parse_args()


# ── Main Execution ─────────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point for the chunking benchmark script."""
    args = parse_arguments()
    
    random.seed(args.seed)
    
    logger.info("=" * 80)
    logger.info("Text Chunking Performance Benchmark")
    logger.info("=" * 80)
    
    sentences = generate_synthetic_corpus(num_sentences=args.num_sentences)
    
    results = benchmark_chunking(
        sentences=sentences,
        chunk_sizes=args.chunk_sizes,
        overlap=args.overlap,
    )
    
    print_results_table(results)
    
    logger.info("Benchmark execution complete.")


if __name__ == "__main__":
    main()
