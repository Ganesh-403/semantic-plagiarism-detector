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
scripts/benchmark_similarity.py
--------------------------------
Benchmark comparison script for similarity algorithms in the Semantic
Plagiarism Detection System.

Measures latency per 1,000 comparisons across:
  - Jaccard similarity (set-theoretic lexical)
  - TF-IDF similarity (term-frequency inverse-document-frequency lexical)
  - Manhattan similarity (L1 distance-based vector)
  - Cosine similarity (angular cosine dot-product vector)

Acceptance Criteria (Issue #3499):
  - Provide scripts/benchmark_similarity.py
  - Benchmark on synthetic 500-word documents
  - Measure latency per 1,000 comparisons
  - Print a formatted summary table

Usage:
  python scripts/benchmark_similarity.py
  python scripts/benchmark_similarity.py --num-comparisons 1000 --word-count 500
  python scripts/benchmark_similarity.py --json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Core algorithm imports with safe fallbacks
try:
    from src.core.lexical_similarity import (
        calculate_lexical_similarity,
        jaccard_similarity,
    )
except ImportError:

    def jaccard_similarity(text_a: str, text_b: str, **kwargs) -> float:
        set_a = set(text_a.lower().split())
        set_b = set(text_b.lower().split())
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def calculate_lexical_similarity(text_a: str, text_b: str, **kwargs) -> float:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        try:
            vec = TfidfVectorizer().fit_transform([text_a, text_b])
            return float(cosine_similarity(vec[0:1], vec[1:2])[0][0])
        except Exception:
            return 0.0


try:
    from src.core.similarity import manhattan_similarity
except ImportError:

    def manhattan_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        a = np.asarray(vec_a, dtype=np.float64)
        b = np.asarray(vec_b, dtype=np.float64)
        dist = float(np.sum(np.abs(a - b)))
        return float(np.clip(1.0 / (1.0 + dist), 0.0, 1.0))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Synthetic Vocabulary & Data Generation ──────────────────────────────────────

_SYNTHETIC_VOCABULARY: list[str] = [
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
    "document",
    "chunk",
    "text",
    "paragraph",
    "sentence",
    "word",
    "token",
    "optimization",
    "retrieval",
    "indexing",
    "clustering",
    "classification",
    "corpus",
    "pipeline",
    "architecture",
    "distributed",
    "scalability",
    "inference",
    "transformer",
    "attention",
    "representation",
    "dimension",
    "precision",
    "recall",
    "accuracy",
    "manhattan",
    "jaccard",
    "tfidf",
    "lexical",
    "hybrid",
    "scoring",
    "verification",
    "analytics",
    "engine",
    "framework",
    "compiler",
    "runtime",
    "parallel",
    "concurrency",
    "matrix",
    "normalized",
    "tensor",
    "gradient",
    "computation",
    "regression",
    "anomaly",
]


def generate_synthetic_sentence(
    min_words: int = 8,
    max_words: int = 16,
    rng: Optional[random.Random] = None,
) -> str:
    """Generate a single synthetic sentence from domain vocabulary."""
    r = rng if rng is not None else random
    num_words = r.randint(min_words, max_words)
    words = [r.choice(_SYNTHETIC_VOCABULARY) for _ in range(num_words)]
    words[0] = words[0].capitalize()
    punct = r.choice([".", ".", ".", "!", "?"])
    return " ".join(words) + punct


def generate_synthetic_document(
    num_words: int = 500,
    rng: Optional[random.Random] = None,
) -> str:
    """Generate a synthetic document with approximately/exactly num_words.

    Args:
        num_words: Target word count (default 500).
        rng: Optional Random instance for reproducibility.

    Returns:
        A formatted string with paragraphs comprising approximately num_words.
    """
    r = rng if rng is not None else random
    sentences: list[str] = []
    current_word_count = 0

    while current_word_count < num_words:
        remaining = num_words - current_word_count
        min_w = min(8, remaining)
        max_w = min(16, remaining) if remaining >= 16 else remaining
        if max_w < 1:
            break
        num_w = r.randint(min_w, max_w) if max_w > min_w else min_w
        words = [r.choice(_SYNTHETIC_VOCABULARY) for _ in range(num_w)]
        words[0] = words[0].capitalize()
        sentences.append(" ".join(words) + ".")
        current_word_count += num_w

    # Group sentences into paragraphs of 3-5 sentences
    paragraphs: list[str] = []
    idx = 0
    while idx < len(sentences):
        p_len = r.randint(3, 5)
        p_sentences = sentences[idx : idx + p_len]
        paragraphs.append(" ".join(p_sentences))
        idx += p_len

    return "\n\n".join(paragraphs)


def generate_synthetic_documents(
    num_docs: int = 20,
    num_words: int = 500,
    seed: int = 42,
) -> list[str]:
    """Generate a list of synthetic documents."""
    rng = random.Random(seed)
    return [
        generate_synthetic_document(num_words=num_words, rng=rng)
        for _ in range(num_docs)
    ]


def generate_synthetic_embeddings(
    num_docs: int = 20,
    dim: int = 384,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate normalized synthetic dense embedding vectors."""
    np_rng = np.random.default_rng(seed)
    embeddings = []
    for _ in range(num_docs):
        vec = np_rng.standard_normal(dim).astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        embeddings.append(vec)
    return embeddings


# ── Similarity Functions ────────────────────────────────────────────────────────


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D or 2D vector arrays."""
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(sim, 0.0, 1.0))


def compute_jaccard_similarity(doc_a: str, doc_b: str) -> float:
    """Compute Jaccard lexical similarity between two documents."""
    return jaccard_similarity(doc_a, doc_b)


def compute_tfidf_similarity(doc_a: str, doc_b: str) -> float:
    """Compute TF-IDF cosine similarity between two documents."""
    return calculate_lexical_similarity(doc_a, doc_b)


def compute_manhattan_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute normalized Manhattan similarity between two embedding vectors."""
    return manhattan_similarity(vec_a, vec_b)


# ── Benchmark Engine ────────────────────────────────────────────────────────────


def create_comparison_pairs(
    items: list[Any],
    num_comparisons: int = 1000,
    seed: int = 42,
) -> list[tuple[Any, Any]]:
    """Create a list of (item_a, item_b) pairs for benchmarking."""
    rng = random.Random(seed)
    n = len(items)
    if n < 2:
        raise ValueError("Need at least 2 items to generate comparison pairs.")
    pairs: list[tuple[Any, Any]] = []
    for _ in range(num_comparisons):
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        pairs.append((items[i], items[j]))
    return pairs


def benchmark_single_algorithm(
    name: str,
    category: str,
    func: Callable[[Any, Any], float],
    pairs: list[tuple[Any, Any]],
) -> dict[str, Any]:
    """Execute pairwise similarity benchmark for a single algorithm.

    Args:
        name: Name of algorithm (e.g. 'Cosine', 'Jaccard').
        category: Algorithm category ('Vector' or 'Lexical').
        func: Pairwise similarity function (a, b) -> float.
        pairs: List of input pairs.

    Returns:
        Dictionary of benchmark metrics.
    """
    num_comparisons = len(pairs)
    scores: list[float] = []

    # Warmup
    for i in range(min(5, num_comparisons)):
        func(pairs[i][0], pairs[i][1])

    start_time = time.perf_counter()
    for item_a, item_b in pairs:
        score = func(item_a, item_b)
        scores.append(score)
    end_time = time.perf_counter()

    total_time_s = end_time - start_time
    if total_time_s <= 0:
        total_time_s = 1e-9

    time_per_op_us = (total_time_s / num_comparisons) * 1_000_000
    latency_per_1000_ms = (total_time_s / num_comparisons) * 1000 * 1000
    throughput_ops_sec = num_comparisons / total_time_s
    avg_score = float(np.mean(scores)) if scores else 0.0

    return {
        "algorithm": name,
        "category": category,
        "comparisons": num_comparisons,
        "total_time_s": total_time_s,
        "latency_per_1000_ms": latency_per_1000_ms,
        "time_per_op_us": time_per_op_us,
        "throughput_ops_sec": throughput_ops_sec,
        "mean_similarity": avg_score,
    }


def run_similarity_benchmarks(
    num_comparisons: int = 1000,
    word_count: int = 500,
    num_docs: int = 20,
    embedding_dim: int = 384,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run benchmark across Jaccard, TF-IDF, Manhattan, and Cosine similarity.

    Args:
        num_comparisons: Number of comparisons per algorithm (default: 1000).
        word_count: Words per synthetic document (default: 500).
        num_docs: Number of synthetic documents to pool from (default: 20).
        embedding_dim: Dimension of synthetic vectors (default: 384).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        List of benchmark metric dictionaries for each algorithm.
    """
    logger.info(
        f"Generating {num_docs} synthetic {word_count}-word documents (seed={seed})..."
    )
    docs = generate_synthetic_documents(
        num_docs=num_docs, num_words=word_count, seed=seed
    )
    doc_pairs = create_comparison_pairs(
        docs, num_comparisons=num_comparisons, seed=seed
    )

    logger.info(
        f"Generating {num_docs} synthetic {embedding_dim}-D embedding vectors..."
    )
    embeddings = generate_synthetic_embeddings(
        num_docs=num_docs, dim=embedding_dim, seed=seed
    )
    vec_pairs = create_comparison_pairs(
        embeddings, num_comparisons=num_comparisons, seed=seed
    )

    benchmarks_to_run = [
        ("Jaccard", "Lexical", compute_jaccard_similarity, doc_pairs),
        ("TF-IDF", "Lexical", compute_tfidf_similarity, doc_pairs),
        ("Manhattan", "Vector", compute_manhattan_similarity, vec_pairs),
        ("Cosine", "Vector", compute_cosine_similarity, vec_pairs),
    ]

    results: list[dict[str, Any]] = []
    for name, cat, func, pairs in benchmarks_to_run:
        logger.info(f"Benchmarking {name} similarity ({len(pairs)} comparisons)...")
        metric = benchmark_single_algorithm(name, cat, func, pairs)
        results.append(metric)

    return results


# ── Reporting & Output ──────────────────────────────────────────────────────────


def format_summary_table(results: list[dict[str, Any]], word_count: int = 500) -> str:
    """Format benchmark results into a clear summary table."""
    header_title = f"Similarity Algorithm Benchmark Summary ({word_count} words/doc)"
    divider = "=" * 106

    lines = [
        divider,
        f"{header_title:^106}",
        divider,
        f"| {'Algorithm':<12} | {'Category':<10} | {'Comparisons':>12} | {'Total Time':>12} | {'Latency / 1k':>15} | {'Time / Op':>14} | {'Throughput':>16} |",
        f"|:{'-'*12}-|-{'-'*10}-|-{'-'*12}:|-{'-'*12}:|-{'-'*15}:|-{'-'*14}:|-{'-'*16}:|",
    ]

    for r in results:
        tot_time_str = f"{r['total_time_s']:.4f} s"
        lat_1k_str = f"{r['latency_per_1000_ms']:.2f} ms"
        time_op_str = f"{r['time_per_op_us']:.2f} us"
        tp_str = f"{r['throughput_ops_sec']:,.0f} ops/s"
        comp_str = f"{r['comparisons']:,}"

        line = (
            f"| {r['algorithm']:<12} "
            f"| {r['category']:<10} "
            f"| {comp_str:>12} "
            f"| {tot_time_str:>12} "
            f"| {lat_1k_str:>15} "
            f"| {time_op_str:>14} "
            f"| {tp_str:>16} |"
        )
        lines.append(line)

    lines.append(divider)
    return "\n".join(lines)


def print_summary_table(results: list[dict[str, Any]], word_count: int = 500) -> None:
    """Print the formatted summary table to standard output."""
    print("\n" + format_summary_table(results, word_count=word_count) + "\n")


# ── CLI Interface ───────────────────────────────────────────────────────────────


def parse_arguments(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark comparison script for similarity algorithms (Jaccard, TF-IDF, Manhattan, Cosine).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-comparisons",
        "-n",
        type=int,
        default=1000,
        help="Number of pairwise comparisons per algorithm.",
    )
    parser.add_argument(
        "--word-count",
        "-w",
        type=int,
        default=500,
        help="Word count for each synthetic document.",
    )
    parser.add_argument(
        "--num-docs",
        "-d",
        type=int,
        default=20,
        help="Number of unique synthetic documents to generate in the pool.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=384,
        help="Embedding vector dimension for vector similarity metrics.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducible document and vector generation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output benchmark results in JSON format.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output markdown table only.",
    )
    return parser.parse_args(args)


def main(args: Optional[list[str]] = None) -> int:
    """Main CLI entrypoint."""
    parsed_args = parse_arguments(args)
    random.seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)

    logger.info("=" * 80)
    logger.info("Starting Similarity Algorithms Benchmark Suite (Issue #3499)...")
    logger.info("=" * 80)

    results = run_similarity_benchmarks(
        num_comparisons=parsed_args.num_comparisons,
        word_count=parsed_args.word_count,
        num_docs=parsed_args.num_docs,
        embedding_dim=parsed_args.embedding_dim,
        seed=parsed_args.seed,
    )

    if parsed_args.json:
        output_payload = {
            "config": {
                "num_comparisons": parsed_args.num_comparisons,
                "word_count": parsed_args.word_count,
                "num_docs": parsed_args.num_docs,
                "embedding_dim": parsed_args.embedding_dim,
                "seed": parsed_args.seed,
            },
            "results": results,
        }
        print(json.dumps(output_payload, indent=2))
    else:
        print_summary_table(results, word_count=parsed_args.word_count)

    logger.info("Benchmark execution complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
