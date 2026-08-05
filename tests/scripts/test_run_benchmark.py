from __future__ import annotations

"""
test_run_benchmark.py
---------------------
Unit tests for the benchmark runner script (scripts/run_benchmark.py).

Validates:
- Synthetic data generation functions
- Argument parsing logic
- Metrics calculation accuracy
- Output formatting helpers
"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Add scripts directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_benchmark


# ─── Synthetic Data Generation Tests ──────────────────────────────────────────


def test_generate_synthetic_sentence_length():
    """Verify sentence generation respects min/max word constraints."""
    sentence = run_benchmark.generate_synthetic_sentence(min_words=5, max_words=5)
    # Remove punctuation for word count
    words = sentence[:-1].split()
    assert len(words) == 5


def test_generate_synthetic_sentence_punctuation():
    """Verify sentences end with valid punctuation."""
    sentence = run_benchmark.generate_synthetic_sentence()
    assert sentence[-1] in [".", "!", "?", ";"]


def test_generate_synthetic_paragraph_structure():
    """Verify paragraph contains multiple sentences."""
    paragraph = run_benchmark.generate_synthetic_paragraph(
        min_sentences=3, max_sentences=3
    )
    # Count sentences by splitting on punctuation
    sentences = [
        s.strip()
        for s in paragraph.replace("!", ".")
        .replace("?", ".")
        .replace(";", ".")
        .split(".")
        if s.strip()
    ]
    assert len(sentences) == 3


def test_generate_synthetic_document():
    """Verify document generation produces non-empty text."""
    doc = run_benchmark.generate_synthetic_document(num_paragraphs=2)
    assert len(doc) > 0
    assert "\n\n" in doc  # Paragraphs separated by double newlines


def test_generate_synthetic_corpus_keys():
    """Verify corpus generation creates correct number of documents."""
    corpus = run_benchmark.generate_synthetic_corpus(num_docs=5, chunks_per_doc=2)
    assert len(corpus) == 5
    for doc_name in corpus:
        assert doc_name.startswith("synthetic_doc_")
        assert doc_name.endswith(".txt")


def test_generate_synthetic_corpus_chunk_count():
    """Verify each document has approximately the requested number of chunks."""
    corpus = run_benchmark.generate_synthetic_corpus(num_docs=3, chunks_per_doc=5)
    for doc_name, chunks in corpus.items():
        # Allow some variance due to chunking logic, but should be close
        assert len(chunks) >= 1
        assert len(chunks) <= 10


# ─── Benchmark Calculation Tests ──────────────────────────────────────────────


@patch("run_benchmark.embed_chunks")
def test_benchmark_embedding_throughput_calculation(mock_embed):
    """Verify throughput calculations are mathematically correct."""
    # Mock embed_chunks to take exactly 0.5 seconds for 100 chunks
    mock_embed.return_value = np.random.rand(100, 384)

    corpus = {"doc1": [f"chunk_{i}" for i in range(100)]}

    with patch("time.perf_counter", side_effect=[0.0, 0.5]):
        metrics = run_benchmark.benchmark_embedding_throughput(corpus)

    assert metrics["total_chunks"] == 100
    assert metrics["total_time_ms"] == 500.0
    assert metrics["ms_per_chunk"] == 5.0
    assert metrics["chunks_per_second"] == 200.0


@patch("run_benchmark.embed_documents")
def test_benchmark_document_embedding_calculation(mock_embed):
    """Verify document-level throughput calculations."""
    mock_embed.return_value = {"doc1": np.random.rand(10, 384)}

    corpus = {f"doc_{i}": ["chunk"] for i in range(20)}

    with patch("time.perf_counter", side_effect=[0.0, 1.0]):
        metrics = run_benchmark.benchmark_document_embedding(corpus)

    assert metrics["total_docs"] == 20
    assert metrics["total_time_ms"] == 1000.0
    assert metrics["ms_per_doc"] == 50.0
    assert metrics["docs_per_second"] == 20.0


def test_benchmark_empty_corpus():
    """Verify benchmark handles empty corpus gracefully."""
    corpus = {}
    metrics = run_benchmark.benchmark_embedding_throughput(corpus)
    assert metrics["total_chunks"] == 0
    assert metrics["ms_per_chunk"] == 0


# ─── Argument Parsing Tests ───────────────────────────────────────────────────


def test_parse_arguments_defaults():
    """Verify default argument values."""
    with patch("sys.argv", ["run_benchmark.py"]):
        args = run_benchmark.parse_arguments()

    assert args.num_docs == 20
    assert args.chunks_per_doc == 10
    assert args.device == "auto"
    assert args.batch_size == 32
    assert args.output_json is None
    assert args.skip_faiss is False


def test_parse_arguments_custom_values():
    """Verify custom argument values are parsed correctly."""
    test_args = [
        "run_benchmark.py",
        "--num-docs",
        "50",
        "--chunks-per-doc",
        "15",
        "--device",
        "cuda",
        "--batch-size",
        "64",
        "--output-json",
        "results.json",
        "--skip-faiss",
    ]

    with patch("sys.argv", test_args):
        args = run_benchmark.parse_arguments()

    assert args.num_docs == 50
    assert args.chunks_per_doc == 15
    assert args.device == "cuda"
    assert args.batch_size == 64
    assert args.output_json == "results.json"
    assert args.skip_faiss is True


def test_parse_arguments_invalid_device():
    """Verify invalid device choice raises SystemExit."""
    test_args = ["run_benchmark.py", "--device", "tpu"]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit):
            run_benchmark.parse_arguments()


# ─── Output Formatting Tests ──────────────────────────────────────────────────


def test_print_metrics_table_output(capsys):
    """Verify print_metrics_table produces formatted output."""
    metrics = {
        "total_chunks": 100,
        "ms_per_chunk": 5.12345,
        "chunks_per_second": 195.5,
    }

    run_benchmark.print_metrics_table("Test Benchmark", metrics, "cpu")

    captured = capsys.readouterr()
    assert "Test Benchmark" in captured.out
    assert "cpu" in captured.out
    assert "Total Chunks" in captured.out
    assert "100" in captured.out
    assert "5.1235" in captured.out  # Formatted to 4 decimal places


def test_save_results_to_json(tmp_path):
    """Verify results are correctly saved to JSON file."""
    results = {"embedding_throughput": {"total_chunks": 100, "ms_per_chunk": 5.0}}
    output_file = tmp_path / "test_results.json"

    args = argparse.Namespace(
        num_docs=10,
        chunks_per_doc=5,
        device="cpu",
        batch_size=32,
    )

    run_benchmark.save_results_to_json(results, str(output_file), args)

    assert output_file.exists()

    with open(output_file, "r") as f:
        saved_data = json.load(f)

    assert "timestamp" in saved_data
    assert saved_data["parameters"]["num_docs"] == 10
    assert saved_data["results"]["embedding_throughput"]["total_chunks"] == 100
