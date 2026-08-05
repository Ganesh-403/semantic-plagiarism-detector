"""
scripts/benchmark_chunking.py
-----------------------------

Benchmark the latency of the project's text chunking utilities over a
synthetic corpus of 10,000 sentences.

Three chunking strategies from :mod:`src.core.text_chunking` are measured:

* ``chunk_text``         – fixed character-count chunking with word-boundary
  awareness and overlap (original strategy).
* ``chunk_by_sentences`` – sentence-boundary-aware chunking that groups whole
  sentences without splitting mid-sentence (Issue #919).
* ``chunk_text_dynamic`` – sliding-window chunking that shifts boundaries to
  the nearest sentence-ending punctuation (Issue #1352).

Chunking throughput (sentences per second) is reported for the chunk sizes
250, 500 and 1000 characters, and a formatted timing-results summary table
is printed on completion.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TEXT_CHUNKING_PATH = os.path.join(ROOT_DIR, "src", "core", "text_chunking.py")

DEFAULT_CHUNK_SIZES = (250, 500, 1000)
DEFAULT_NUM_SENTENCES = 10_000
DEFAULT_RUNS = 1
DEFAULT_MAX_CHUNKS = 100_000
RANDOM_SEED = 42

NOUN_PHRASES = [
    "the rapid advancement of artificial intelligence",
    "modern machine learning models",
    "the automated detection of plagiarism",
    "high-dimensional text embeddings",
    "the quality of extracted document text",
    "sentence-boundary-aware chunking",
    "large-scale document corpora",
    "the semantic similarity between passages",
    "optimized similarity search algorithms",
    "carefully curated training datasets",
]

VERB_PHRASES = [
    "has significantly improved",
    "relies on",
    "requires careful consideration of",
    "can dramatically reduce",
    "demonstrates the importance of",
    "builds upon",
    "depends heavily on",
    "plays a crucial role in",
    "offers a robust solution for",
    "continues to evolve through",
]

TAIL_PHRASES = [
    "the accuracy of downstream plagiarism detection tasks.",
    "high-dimensional vector representations of unstructured text.",
    "fast and reliable retrieval over millions of indexed chunks.",
    "the trade-off between latency and retrieval quality.",
    "robust preprocessing pipelines that normalize noisy input.",
    "the reproducible evaluation of retrieval pipelines.",
    "adaptive strategies that balance speed and semantic fidelity.",
    "end-to-end document processing workflows.",
    "benchmarking efforts across varying chunk configurations.",
    "the practical deployment of large-scale search systems.",
]


@dataclass
class BenchmarkConfig:
    chunk_sizes: tuple[int, ...]
    num_sentences: int
    runs: int
    max_chunks: int


@dataclass
class BenchmarkResult:
    chunker_name: str
    chunk_size: int
    elapsed_seconds: float
    num_sentences: int
    num_chunks: int

    @property
    def throughput(self) -> float:
        return self.num_sentences / self.elapsed_seconds


@dataclass(frozen=True)
class ChunkTarget:
    name: str
    func: Callable[..., list[str]]
    size_parameter: str
    extra_kwargs: dict[str, int] = field(default_factory=dict)


def _parse_chunk_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "chunk sizes must be comma-separated integers"
        ) from exc
    if not sizes:
        raise argparse.ArgumentTypeError("at least one chunk size is required")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("chunk sizes must be positive integers")
    return sizes


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark text chunking throughput over 10,000 sentences."
    )
    parser.add_argument(
        "--chunk-sizes",
        type=_parse_chunk_sizes,
        default=DEFAULT_CHUNK_SIZES,
        metavar="SIZES",
        help=(
            "Comma-separated chunk sizes in characters "
            f"(default: {','.join(str(s) for s in DEFAULT_CHUNK_SIZES)})."
        ),
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=DEFAULT_NUM_SENTENCES,
        metavar="COUNT",
        help=f"Number of sentences to chunk (default: {DEFAULT_NUM_SENTENCES}).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        metavar="COUNT",
        help="Number of timing repetitions per configuration (default: "
        f"{DEFAULT_RUNS}, the fastest run is reported).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        metavar="COUNT",
        help="Upper bound on chunks produced by ``chunk_text`` so the full "
        "corpus is processed instead of being truncated at the default "
        f"limit of 1000 (default: {DEFAULT_MAX_CHUNKS}).",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    namespace = build_argument_parser().parse_args(argv)
    if namespace.sentences <= 0:
        raise argparse.ArgumentTypeError("--sentences must be a positive integer")
    if namespace.runs <= 0:
        raise argparse.ArgumentTypeError("--runs must be a positive integer")
    if namespace.max_chunks <= 0:
        raise argparse.ArgumentTypeError("--max-chunks must be a positive integer")
    return BenchmarkConfig(
        chunk_sizes=namespace.chunk_sizes,
        num_sentences=namespace.sentences,
        runs=namespace.runs,
        max_chunks=namespace.max_chunks,
    )


def load_chunking_module() -> Any:
    """Load ``src/core/text_chunking.py`` as a standalone module.

    The file is self-contained (standard-library imports only), so it can be
    imported by path without executing the rest of the package.
    """
    spec = importlib.util.spec_from_file_location("text_chunking", TEXT_CHUNKING_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load chunking module: {TEXT_CHUNKING_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_sentences(count: int, seed: int = RANDOM_SEED) -> list[str]:
    """Return *count* deterministic, grammatically plausible sentences."""
    rng = random.Random(seed)
    sentences = []
    for _ in range(count):
        sentence = (
            f"{rng.choice(NOUN_PHRASES).capitalize()} "
            f"{rng.choice(VERB_PHRASES)} {rng.choice(TAIL_PHRASES)}"
        )
        sentences.append(sentence)
    return sentences


def build_chunk_targets(module: Any, max_chunks: int) -> list[ChunkTarget]:
    return [
        ChunkTarget(
            "chunk_text",
            module.chunk_text,
            "chunk_size",
            {"max_chunks": max_chunks},
        ),
        ChunkTarget(
            "chunk_by_sentences",
            module.chunk_by_sentences,
            "max_chunk_size",
        ),
        ChunkTarget(
            "chunk_text_dynamic",
            module.chunk_text_dynamic,
            "target_size",
        ),
    ]


def measure_chunker(
    target: ChunkTarget,
    text: str,
    chunk_size: int,
    sentence_count: int,
    runs: int,
) -> BenchmarkResult:
    call_kwargs = {target.size_parameter: chunk_size, **target.extra_kwargs}
    call = functools.partial(target.func, text, **call_kwargs)
    best_elapsed = float("inf")
    last_num_chunks = 0
    for _ in range(runs):
        start = time.perf_counter()
        chunks = call()
        elapsed = time.perf_counter() - start
        last_num_chunks = len(chunks)
        if elapsed < best_elapsed:
            best_elapsed = elapsed
    return BenchmarkResult(
        chunker_name=target.name,
        chunk_size=chunk_size,
        elapsed_seconds=best_elapsed,
        num_sentences=sentence_count,
        num_chunks=last_num_chunks,
    )


def run_benchmarks(
    module: Any,
    text: str,
    sentence_count: int,
    chunk_sizes: tuple[int, ...],
    runs: int,
    max_chunks: int,
) -> list[BenchmarkResult]:
    results = []
    for target in build_chunk_targets(module, max_chunks):
        for chunk_size in chunk_sizes:
            results.append(
                measure_chunker(target, text, chunk_size, sentence_count, runs)
            )
    return results


def print_corpus_summary(text: str, sentence_count: int, runs: int, max_chunks: int) -> None:
    word_count = len(text.split())
    print(f"Corpus: {sentence_count} sentences, {word_count} words, {len(text)} characters")
    print(f"Timing repetitions per configuration: {runs}")
    print(f"Chunk cap for chunk_text: {max_chunks}")
    print()


def print_results_table(results: list[BenchmarkResult]) -> None:
    print("Text Chunking Benchmark Results")
    print("===============================")
    print()
    print(
        f"{'Chunker':<18} {'Chunk Size':>10} {'Elapsed (s)':>12} "
        f"{'Chunks':>7} {'Sentences/s':>12}"
    )
    print("-" * 65)
    for result in results:
        print(
            f"{result.chunker_name:<18} {result.chunk_size:>10} "
            f"{result.elapsed_seconds:>12.4f} {result.num_chunks:>7} "
            f"{result.throughput:>12.1f}"
        )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    module = load_chunking_module()
    sentences = generate_sentences(config.num_sentences)
    text = " ".join(sentences)

    print_corpus_summary(text, config.num_sentences, config.runs, config.max_chunks)
    results = run_benchmarks(
        module,
        text,
        config.num_sentences,
        config.chunk_sizes,
        config.runs,
        config.max_chunks,
    )
    print_results_table(results)


if __name__ == "__main__":
    main()
