"""Exercise benchmarks against production chunkers with small input samples."""
import random
import tracemalloc
from unittest.mock import Mock

import pytest
from scripts import benchmark_chunking as benchmark


@pytest.mark.parametrize("size", [0, 0.001, 0.01])
def test_synthetic_corpus_has_requested_byte_size(size):
    text = benchmark.generate_synthetic_text(size)
    assert len(text.encode("utf-8")) == int(size * 1024 * 1024)
    if size:
        assert " " in text
        assert len(text.split()) > 1


def test_reproducible_sample():
    random.seed(42)
    first = benchmark.generate_synthetic_text(0.01)
    random.seed(42)
    assert benchmark.generate_synthetic_text(0.01) == first


def test_negative_sample_rejected():
    with pytest.raises(ValueError):
        benchmark.generate_synthetic_text(-1)


def test_benchmark_calculates_metrics(monkeypatch):
    ticks = iter([0., 0.5])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    chunker = Mock(return_value=["one", "two"])
    result = benchmark.benchmark_algorithm("example", chunker, "a" * 1024 * 1024, 1)
    assert result["Time (s)"] == "0.5000s"
    assert result["Throughput (MB/s)"] == "2.00 MB/s"
    assert result["Chunks"] == 2
    chunker.assert_called_once()
    assert not tracemalloc.is_tracing()


def test_algorithm_failure_is_reported_and_tracing_stops():
    result = benchmark.benchmark_algorithm("broken", Mock(side_effect=ValueError), "text", 1)
    assert result["Time (s)"] == "FAILED"
    assert result["Throughput (MB/s)"] == "N/A"
    assert result["Chunks"] == 0
    assert not tracemalloc.is_tracing()


def test_real_algorithms_and_report(capsys):
    results = benchmark.run_benchmarks(sizes=[0.001, 0.002])
    assert len(results) == 6
    assert {r["Algorithm"] for r in results} == {
        "chunk_text", "chunk_by_sentences", "ContextPreservingChunker",
    }
    assert all(r["Time (s)"] != "FAILED" and r["Chunks"] > 0 for r in results)
    benchmark.print_markdown_table(results)
    output = capsys.readouterr().out
    assert "Throughput (MB/s)" in output
    for row in results:
        assert row["Algorithm"] in output and row["Size"] in output


@pytest.mark.parametrize("args, expected", [([], 42), (["--seed", "123"], 123)])
def test_cli_arguments(monkeypatch, args, expected):
    monkeypatch.setattr("sys.argv", ["benchmark_chunking.py", *args])
    assert benchmark.parse_arguments().seed == expected


def test_main_runs_and_reports(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["benchmark_chunking.py"])
    runner = Mock(return_value=[])
    monkeypatch.setattr(benchmark, "run_benchmarks", runner)
    benchmark.main()
    runner.assert_called_once_with()
    assert "Chunking Benchmark Results" in capsys.readouterr().out
