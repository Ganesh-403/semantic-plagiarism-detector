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
tests/scripts/test_benchmark_similarity.py
-------------------------------------------
Unit tests for the similarity benchmark comparison script (scripts/benchmark_similarity.py).

Validates:
- Synthetic document and embedding vector generation
- Jaccard, TF-IDF, Manhattan, and Cosine similarity computation
- Benchmark execution engine and metric calculations
- Summary table formatting and JSON reporting
- CLI argument parsing and main entrypoint
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Add scripts directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_similarity


class TestSyntheticDataGeneration:
    """Test suite for synthetic document and embedding generation."""

    def test_generate_synthetic_sentence(self):
        """Verify sentence word count and punctuation constraints."""
        sentence = benchmark_similarity.generate_synthetic_sentence(
            min_words=6, max_words=10
        )
        words = sentence[:-1].split()
        assert 6 <= len(words) <= 10
        assert sentence[-1] in [".", "!", "?"]

    def test_generate_synthetic_document_word_count(self):
        """Verify synthetic document generates approximately 500 words."""
        doc = benchmark_similarity.generate_synthetic_document(num_words=500)
        words = [w for w in doc.split() if w]
        assert 450 <= len(words) <= 550
        assert "\n\n" in doc  # Document structured into paragraphs

    def test_generate_synthetic_documents_count(self):
        """Verify generation of multiple documents."""
        docs = benchmark_similarity.generate_synthetic_documents(
            num_docs=5, num_words=100, seed=42
        )
        assert len(docs) == 5
        assert all(isinstance(d, str) and len(d) > 0 for d in docs)

    def test_generate_synthetic_embeddings(self):
        """Verify embedding dimensions and L2 normalization."""
        embeddings = benchmark_similarity.generate_synthetic_embeddings(
            num_docs=4, dim=128, seed=42
        )
        assert len(embeddings) == 4
        for vec in embeddings:
            assert vec.shape == (128,)
            norm = np.linalg.norm(vec)
            assert np.isclose(norm, 1.0, atol=1e-5)


class TestSimilarityFunctions:
    """Test suite for individual similarity metric calculations."""

    def test_compute_cosine_similarity(self):
        """Verify cosine similarity behavior on known vectors."""
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([1.0, 0.0, 0.0])
        vec_c = np.array([0.0, 1.0, 0.0])

        assert np.isclose(
            benchmark_similarity.compute_cosine_similarity(vec_a, vec_b), 1.0
        )
        assert np.isclose(
            benchmark_similarity.compute_cosine_similarity(vec_a, vec_c), 0.0
        )

        # Zero vector handling
        assert benchmark_similarity.compute_cosine_similarity(np.zeros(3), vec_a) == 0.0

    def test_compute_manhattan_similarity(self):
        """Verify Manhattan similarity behavior on identical and distinct vectors."""
        vec_a = np.array([0.5, 0.5])
        vec_b = np.array([0.5, 0.5])
        vec_c = np.array([10.0, 10.0])

        sim_identical = benchmark_similarity.compute_manhattan_similarity(vec_a, vec_b)
        sim_distant = benchmark_similarity.compute_manhattan_similarity(vec_a, vec_c)

        assert np.isclose(sim_identical, 1.0)
        assert 0.0 <= sim_distant < 0.2

    def test_compute_jaccard_similarity(self):
        """Verify Jaccard similarity behavior."""
        text_a = "machine learning neural network algorithm"
        text_b = "machine learning neural network algorithm"
        text_c = "quantum chemistry biology photosynthesis"

        sim_identical = benchmark_similarity.compute_jaccard_similarity(text_a, text_b)
        sim_distinct = benchmark_similarity.compute_jaccard_similarity(text_a, text_c)

        assert np.isclose(sim_identical, 1.0)
        assert np.isclose(sim_distinct, 0.0)

    def test_compute_tfidf_similarity(self):
        """Verify TF-IDF cosine similarity calculation."""
        text_a = "natural language processing with transformers and deep learning"
        text_b = "natural language processing with transformers and deep learning"
        text_c = "astrophysics black hole gravitational wave observation"

        sim_identical = benchmark_similarity.compute_tfidf_similarity(text_a, text_b)
        sim_distinct = benchmark_similarity.compute_tfidf_similarity(text_a, text_c)

        assert np.isclose(sim_identical, 1.0)
        assert 0.0 <= sim_distinct < sim_identical


class TestBenchmarkExecution:
    """Test suite for benchmark execution and latency metrics."""

    def test_create_comparison_pairs(self):
        """Verify comparison pair generation."""
        items = ["doc1", "doc2", "doc3"]
        pairs = benchmark_similarity.create_comparison_pairs(
            items, num_comparisons=50, seed=42
        )
        assert len(pairs) == 50
        for a, b in pairs:
            assert a in items and b in items

    def test_create_comparison_pairs_error_handling(self):
        """Verify error raised when items list has fewer than 2 items."""
        with pytest.raises(ValueError, match="Need at least 2 items"):
            benchmark_similarity.create_comparison_pairs(
                ["single_item"], num_comparisons=10
            )

    def test_benchmark_single_algorithm_metrics(self):
        """Verify single algorithm benchmarking returns all expected metric fields."""
        pairs = [("doc1", "doc2")] * 20
        metric = benchmark_similarity.benchmark_single_algorithm(
            name="TestAlgo",
            category="TestCategory",
            func=lambda a, b: 0.85,
            pairs=pairs,
        )

        assert metric["algorithm"] == "TestAlgo"
        assert metric["category"] == "TestCategory"
        assert metric["comparisons"] == 20
        assert metric["total_time_s"] > 0
        assert metric["latency_per_1000_ms"] > 0
        assert metric["time_per_op_us"] > 0
        assert metric["throughput_ops_sec"] > 0
        assert np.isclose(metric["mean_similarity"], 0.85)

    def test_run_similarity_benchmarks(self):
        """Verify running the full benchmark suite across all 4 algorithms."""
        results = benchmark_similarity.run_similarity_benchmarks(
            num_comparisons=20,
            word_count=50,
            num_docs=4,
            embedding_dim=64,
            seed=42,
        )

        algorithms = [r["algorithm"] for r in results]
        assert "Jaccard" in algorithms
        assert "TF-IDF" in algorithms
        assert "Manhattan" in algorithms
        assert "Cosine" in algorithms
        assert len(results) == 4


class TestReportingAndFormatting:
    """Test suite for summary table formatting."""

    def test_format_summary_table(self):
        """Verify summary table formatting contains headers and algorithms."""
        mock_results = [
            {
                "algorithm": "Jaccard",
                "category": "Lexical",
                "comparisons": 1000,
                "total_time_s": 0.12,
                "latency_per_1000_ms": 120.0,
                "time_per_op_us": 120.0,
                "throughput_ops_sec": 8333.3,
                "mean_similarity": 0.5,
            },
            {
                "algorithm": "Cosine",
                "category": "Vector",
                "comparisons": 1000,
                "total_time_s": 0.005,
                "latency_per_1000_ms": 5.0,
                "time_per_op_us": 5.0,
                "throughput_ops_sec": 200000.0,
                "mean_similarity": 0.5,
            },
        ]

        table_str = benchmark_similarity.format_summary_table(
            mock_results, word_count=500
        )
        assert "Similarity Algorithm Benchmark Summary" in table_str
        assert "Jaccard" in table_str
        assert "Cosine" in table_str
        assert "1,000" in table_str
        assert "Throughput" in table_str

    def test_print_summary_table(self, capsys):
        """Verify print_summary_table prints output to stdout."""
        mock_results = [
            {
                "algorithm": "Cosine",
                "category": "Vector",
                "comparisons": 1000,
                "total_time_s": 0.01,
                "latency_per_1000_ms": 10.0,
                "time_per_op_us": 10.0,
                "throughput_ops_sec": 100000.0,
                "mean_similarity": 0.7,
            }
        ]
        benchmark_similarity.print_summary_table(mock_results)
        captured = capsys.readouterr()
        assert "Cosine" in captured.out


class TestCLIArgumentsAndMain:
    """Test suite for argument parsing and CLI entrypoint."""

    def test_parse_arguments_defaults(self):
        """Verify default CLI arguments."""
        with patch("sys.argv", ["benchmark_similarity.py"]):
            args = benchmark_similarity.parse_arguments()

        assert args.num_comparisons == 1000
        assert args.word_count == 500
        assert args.num_docs == 20
        assert args.embedding_dim == 384
        assert args.seed == 42
        assert not args.json

    def test_parse_arguments_custom(self):
        """Verify custom CLI arguments."""
        test_args = [
            "benchmark_similarity.py",
            "--num-comparisons",
            "500",
            "--word-count",
            "250",
            "--num-docs",
            "10",
            "--embedding-dim",
            "128",
            "--seed",
            "99",
            "--json",
        ]
        with patch("sys.argv", test_args):
            args = benchmark_similarity.parse_arguments()

        assert args.num_comparisons == 500
        assert args.word_count == 250
        assert args.num_docs == 10
        assert args.embedding_dim == 128
        assert args.seed == 99
        assert args.json is True

    def test_main_cli_execution(self, capsys):
        """Verify main() executes and prints summary table with return code 0."""
        exit_code = benchmark_similarity.main(
            ["--num-comparisons", "20", "--word-count", "50", "--num-docs", "4"]
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Similarity Algorithm Benchmark Summary" in captured.out

    def test_main_cli_json_output(self, capsys):
        """Verify main() with --json outputs valid JSON string."""
        exit_code = benchmark_similarity.main(
            [
                "--num-comparisons",
                "10",
                "--word-count",
                "50",
                "--num-docs",
                "4",
                "--json",
            ]
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "config" in data
        assert "results" in data
        assert len(data["results"]) == 4
