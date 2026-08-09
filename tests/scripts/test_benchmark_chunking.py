"""
tests/scripts/test_benchmark_chunking.py
----------------------------------------
Unit tests for the text chunking benchmark script.

Validates:
- Synthetic data generation
- Benchmark execution logic
- Results table formatting
"""

import sys
from pathlib import Path
from unittest.mock import patch


# Add scripts directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_chunking


class TestSyntheticDataGeneration:
    """Test suite for synthetic corpus generation."""

    def test_generate_synthetic_sentence_length(self):
        """Verify sentence generation respects min/max word constraints."""
        sentence = benchmark_chunking.generate_synthetic_sentence(min_words=5, max_words=5)
        words = sentence[:-1].split()  # Remove punctuation
        assert len(words) == 5

    def test_generate_synthetic_sentence_punctuation(self):
        """Verify sentences end with valid punctuation."""
        sentence = benchmark_chunking.generate_synthetic_sentence()
        assert sentence[-1] in [".", "!", "?"]

    def test_generate_synthetic_corpus_count(self):
        """Verify corpus generation creates the requested number of sentences."""
        corpus = benchmark_chunking.generate_synthetic_corpus(num_sentences=100)
        assert len(corpus) == 100
        assert all(isinstance(s, str) for s in corpus)


class TestBenchmarkExecution:
    """Test suite for benchmark execution logic."""

    @patch("benchmark_chunking.chunk_documents")
    def test_benchmark_chunking_calculates_metrics(self, mock_chunk):
        """Verify benchmark calculates time, chunks, and throughput correctly."""
        mock_chunk.return_value = ["chunk1", "chunk2", "chunk3"]
        sentences = ["word " * 10] * 100  # 100 sentences
        
        # Mock time.perf_counter to simulate 0.5 seconds elapsed
        with patch("benchmark_chunking.time.perf_counter", side_effect=[0.0, 0.5]):
            results = benchmark_chunking.benchmark_chunking(
                sentences=sentences,
                chunk_sizes=[500],
                overlap=50,
            )
            
        assert 500 in results
        metrics = results[500]
        
        assert metrics["time_ms"] == 500.0
        assert metrics["chunks_created"] == 3
        assert metrics["sentences_per_sec"] == 200.0  # 100 sentences / 0.5 sec

    @patch("benchmark_chunking.chunk_documents")
    def test_benchmark_multiple_chunk_sizes(self, mock_chunk):
        """Verify benchmark tests all requested chunk sizes."""
        mock_chunk.return_value = ["chunk"]
        sentences = ["test"] * 10
        
        results = benchmark_chunking.benchmark_chunking(
            sentences=sentences,
            chunk_sizes=[250, 500, 1000],
        )
        
        assert set(results.keys()) == {250, 500, 1000}
        assert mock_chunk.call_count == 3


class TestResultsReporting:
    """Test suite for results table formatting."""

    def test_print_results_table_output(self, capsys):
        """Verify print_results_table produces formatted ASCII output."""
        results = {
            250: {
                "time_ms": 10.5,
                "chunks_created": 50,
                "sentences_per_sec": 1000.0,
                "chars_per_sec": 5000.0,
            },
            500: {
                "time_ms": 15.2,
                "chunks_created": 25,
                "sentences_per_sec": 800.0,
                "chars_per_sec": 4000.0,
            }
        }
        
        benchmark_chunking.print_results_table(results)
        
        captured = capsys.readouterr()
        assert "250" in captured.out
        assert "500" in captured.out
        assert "10.50" in captured.out
        assert "1,000" in captured.out  # Formatted with comma


class TestCLIArguments:
    """Test suite for CLI argument parsing."""

    def test_parse_arguments_defaults(self):
        """Verify default CLI argument values."""
        with patch("sys.argv", ["benchmark_chunking.py"]):
            args = benchmark_chunking.parse_arguments()
            
        assert args.num_sentences == 10000
        assert args.chunk_sizes == [250, 500, 1000]
        assert args.overlap == 50
        assert args.seed == 42

    def test_parse_arguments_custom(self):
        """Verify custom CLI argument values are parsed correctly."""
        test_args = [
            "benchmark_chunking.py",
            "--num-sentences", "5000",
            "--chunk-sizes", "100", "200",
            "--overlap", "20",
            "--seed", "123",
        ]
        
        with patch("sys.argv", test_args):
            args = benchmark_chunking.parse_arguments()
            
        assert args.num_sentences == 5000
        assert args.chunk_sizes == [100, 200]
        assert args.overlap == 20
        assert args.seed == 123
