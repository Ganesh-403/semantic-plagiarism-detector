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
run_benchmark.py
----------------
Synthetic benchmark runner for evaluating vector search latency and
embedding throughput in the Semantic Plagiarism Detection System.

Generates dummy documents, measures execution time for embedding and
FAISS indexing, and prints formatted latency metrics tables.

Usage:
    python scripts/run_benchmark.py --num-docs 50 --chunks-per-doc 20 --device cpu

Acceptance Criteria (Issue #955):
- Support --num-docs, --chunks-per-doc, and --device flags.
- Print formatted latency metrics table (ms/chunk, throughput docs/sec).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.embedding_model import _detect_device, embed_chunks, embed_documents
from src.core.faiss_index import build_index
from src.core.text_chunking import chunk_documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Synthetic Data Generation ──────────────────────────────────────────────────

# Vocabulary for generating synthetic text
_SYNTHETIC_VOCABULARY = [
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
]


def generate_synthetic_sentence(min_words: int = 8, max_words: int = 15) -> str:
    """Generate a single synthetic sentence from the vocabulary."""
    num_words = random.randint(min_words, max_words)
    words = [random.choice(_SYNTHETIC_VOCABULARY) for _ in range(num_words)]
    # Capitalize first word and add punctuation
    words[0] = words[0].capitalize()
    return " ".join(words) + random.choice([".", "!", "?", ";"])


def generate_synthetic_paragraph(min_sentences: int = 3, max_sentences: int = 6) -> str:
    """Generate a synthetic paragraph consisting of multiple sentences."""
    num_sentences = random.randint(min_sentences, max_sentences)
    sentences = [generate_synthetic_sentence() for _ in range(num_sentences)]
    return " ".join(sentences)


def generate_synthetic_document(
    num_paragraphs: int = 5,
    paragraphs_per_chunk: int = 2,
) -> str:
    """
    Generate a complete synthetic document.

    Args:
        num_paragraphs: Total number of paragraphs in the document.
        paragraphs_per_chunk: How many paragraphs constitute one logical chunk.

    Returns:
        A string containing the full synthetic document text.
    """
    paragraphs = [generate_synthetic_paragraph() for _ in range(num_paragraphs)]
    return "\n\n".join(paragraphs)


def generate_synthetic_corpus(
    num_docs: int,
    chunks_per_doc: int,
) -> dict[str, list[str]]:
    """
    Generate a synthetic corpus of documents and their pre-chunked text.

    Args:
        num_docs: Number of documents to generate.
        chunks_per_doc: Target number of chunks per document.

    Returns:
        Dictionary mapping document names to lists of text chunks.
    """
    corpus = {}
    logger.info(
        f"Generating synthetic corpus: {num_docs} docs, {chunks_per_doc} chunks/doc"
    )

    for i in range(num_docs):
        doc_name = f"synthetic_doc_{i:04d}.txt"
        # Generate enough text to produce approximately chunks_per_doc chunks
        # Assuming ~500 chars per chunk, generate ~600 chars per chunk
        total_chars = chunks_per_doc * 600
        paragraphs_needed = max(1, total_chars // 300)

        doc_text = generate_synthetic_document(num_paragraphs=paragraphs_needed)

        # Chunk the document using the project's standard chunking logic
        chunks = chunk_documents(
            [doc_text],
            chunk_size=500,
            chunk_overlap=50,
        )

        # Ensure we have at least some chunks, pad if necessary
        if not chunks:
            chunks = [generate_synthetic_paragraph()]

        # Trim to exact chunks_per_doc if chunking produced too many
        if len(chunks) > chunks_per_doc:
            chunks = chunks[:chunks_per_doc]

        corpus[doc_name] = chunks

    return corpus


# ── Benchmark Execution ────────────────────────────────────────────────────────


def benchmark_embedding_throughput(
    corpus: dict[str, list[str]],
    batch_size: int = 32,
) -> dict[str, float]:
    """
    Measure the throughput and latency of the embedding pipeline.

    Args:
        corpus: Dictionary mapping document names to lists of chunks.
        batch_size: Batch size to use for embedding.

    Returns:
        Dictionary containing latency and throughput metrics.
    """
    all_chunks = []
    for chunks in corpus.values():
        all_chunks.extend(chunks)

    total_chunks = len(all_chunks)
    logger.info(f"Benchmarking embedding throughput for {total_chunks} chunks...")

    start_time = time.perf_counter()
    embeddings = embed_chunks(all_chunks, batch_size=batch_size)
    end_time = time.perf_counter()

    total_time_seconds = end_time - start_time
    ms_per_chunk = (total_time_seconds / total_chunks) * 1000 if total_chunks > 0 else 0
    chunks_per_second = (
        total_chunks / total_time_seconds if total_time_seconds > 0 else 0
    )

    return {
        "total_chunks": total_chunks,
        "total_time_ms": total_time_seconds * 1000,
        "ms_per_chunk": ms_per_chunk,
        "chunks_per_second": chunks_per_second,
        "embedding_dim": embeddings.shape[1] if embeddings.size > 0 else 0,
    }


def benchmark_document_embedding(
    corpus: dict[str, list[str]],
    batch_size: int = 32,
) -> dict[str, float]:
    """
    Measure the throughput of document-level embedding (embed_documents).

    Args:
        corpus: Dictionary mapping document names to lists of chunks.
        batch_size: Batch size to use for embedding.

    Returns:
        Dictionary containing document-level throughput metrics.
    """
    num_docs = len(corpus)
    logger.info(f"Benchmarking document embedding for {num_docs} documents...")

    start_time = time.perf_counter()
    embed_documents(corpus, batch_size=batch_size)
    end_time = time.perf_counter()

    total_time_seconds = end_time - start_time
    docs_per_second = num_docs / total_time_seconds if total_time_seconds > 0 else 0
    ms_per_doc = (total_time_seconds / num_docs) * 1000 if num_docs > 0 else 0

    return {
        "total_docs": num_docs,
        "total_time_ms": total_time_seconds * 1000,
        "ms_per_doc": ms_per_doc,
        "docs_per_second": docs_per_second,
    }


def benchmark_faiss_indexing(
    corpus: dict[str, list[str]],
    batch_size: int = 32,
) -> dict[str, float]:
    """
    Measure the time required to build a FAISS index from the corpus.

    Args:
        corpus: Dictionary mapping document names to lists of chunks.
        batch_size: Batch size to use for embedding before indexing.

    Returns:
        Dictionary containing FAISS indexing metrics.
    """
    logger.info("Benchmarking FAISS index build time...")

    # First, embed all chunks
    all_chunks = []
    for chunks in corpus.values():
        all_chunks.extend(chunks)

    embeddings = embed_chunks(all_chunks, batch_size=batch_size)

    # Create mock registry for FAISS
    from src.core.faiss_index import ChunkRecord

    registry = [
        ChunkRecord(f"doc_{i}", i, f"chunk_{i}") for i in range(len(all_chunks))
    ]

    start_time = time.perf_counter()
    index = build_index(embeddings, registry)
    end_time = time.perf_counter()

    total_time_seconds = end_time - start_time
    vectors_per_second = (
        len(all_chunks) / total_time_seconds if total_time_seconds > 0 else 0
    )

    return {
        "total_vectors": len(all_chunks),
        "index_build_time_ms": total_time_seconds * 1000,
        "vectors_indexed_per_second": vectors_per_second,
        "index_ntotal": index.ntotal if index else 0,
    }


# ── Reporting & Output ─────────────────────────────────────────────────────────


def print_metrics_table(
    title: str,
    metrics: dict[str, float],
    device: str,
) -> None:
    """
    Print a formatted ASCII table of benchmark metrics.

    Args:
        title: Title of the benchmark section.
        metrics: Dictionary of metric names and values.
        device: The compute device used for the benchmark.
    """
    print("\n" + "=" * 70)
    print(f"  {title} (Device: {device})")
    print("=" * 70)
    print(f"{'Metric':<35} | {'Value':<30}")
    print("-" * 70)

    for key, value in metrics.items():
        # Format metric name to be more readable
        readable_key = key.replace("_", " ").title()

        # Format value based on type
        if isinstance(value, float):
            if value > 1000:
                formatted_value = f"{value:,.2f}"
            else:
                formatted_value = f"{value:.4f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)

        print(f"{readable_key:<35} | {formatted_value:<30}")

    print("=" * 70 + "\n")


def save_results_to_json(
    results: dict[str, dict[str, float]],
    output_path: str,
    args: argparse.Namespace,
) -> None:
    """
    Save benchmark results to a JSON file for historical tracking.

    Args:
        results: Dictionary of benchmark section results.
        output_path: Path to save the JSON file.
        args: Parsed command line arguments.
    """
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "num_docs": args.num_docs,
            "chunks_per_doc": args.chunks_per_doc,
            "device": args.device,
            "batch_size": args.batch_size,
        },
        "results": results,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Benchmark results saved to {output_file}")


# ── CLI Argument Parsing ───────────────────────────────────────────────────────


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Semantic Plagiarism Detection System - Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num-docs",
        type=int,
        default=20,
        help="Number of synthetic documents to generate.",
    )
    parser.add_argument(
        "--chunks-per-doc",
        type=int,
        default=10,
        help="Target number of text chunks per document.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Compute device to use for embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding mini-batches.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save results as JSON.",
    )
    parser.add_argument(
        "--skip-faiss",
        action="store_true",
        help="Skip FAISS indexing benchmark.",
    )

    return parser.parse_args()


# ── Main Execution ─────────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point for the benchmark runner script."""
    args = parse_arguments()

    logger.info("=" * 70)
    logger.info("Semantic Plagiarism Detection System - Benchmark Runner")
    logger.info("=" * 70)
    logger.info(
        f"Parameters: {args.num_docs} docs, {args.chunks_per_doc} chunks/doc, "
        f"batch_size={args.batch_size}, device={args.device}"
    )

    # Force device selection if specified
    if args.device != "auto":
        os.environ["SEMANTIC_PLAGIARISM_DEVICE"] = args.device

    # Generate synthetic corpus
    corpus = generate_synthetic_corpus(
        num_docs=args.num_docs,
        chunks_per_doc=args.chunks_per_doc,
    )

    # Detect actual device being used
    active_device = _detect_device()
    logger.info(f"Active compute device: {active_device}")

    results = {}

    # 1. Benchmark Embedding Throughput (Chunk-level)
    embed_metrics = benchmark_embedding_throughput(corpus, batch_size=args.batch_size)
    print_metrics_table(
        "Embedding Throughput (Chunk-Level)", embed_metrics, active_device
    )
    results["embedding_throughput"] = embed_metrics

    # 2. Benchmark Document-Level Embedding
    doc_metrics = benchmark_document_embedding(corpus, batch_size=args.batch_size)
    print_metrics_table("Document Embedding Throughput", doc_metrics, active_device)
    results["document_embedding"] = doc_metrics

    # 3. Benchmark FAISS Indexing
    if not args.skip_faiss:
        faiss_metrics = benchmark_faiss_indexing(corpus, batch_size=args.batch_size)
        print_metrics_table(
            "FAISS Index Build Performance", faiss_metrics, active_device
        )
        results["faiss_indexing"] = faiss_metrics

    # Save results to JSON if requested
    if args.output_json:
        save_results_to_json(results, args.output_json, args)

    logger.info("Benchmark execution complete.")


if __name__ == "__main__":
    main()
