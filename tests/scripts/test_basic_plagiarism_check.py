from __future__ import annotations

"""
test_basic_plagiarism_check.py
------------------------------
Unit tests for the quickstart example script
(examples/basic_plagiarism_check.py).

Validates:
- Argument parsing (usage message on wrong arg count)
- Missing file detection
- Pipeline execution flow (extract -> chunk -> embed -> score -> flag)
- Output formatting of the similarity matrix and flagged pairs

The core pipeline functions are patched so no embedding model is loaded
during tests.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Add repository root to path (mirrors tests/scripts/test_coverage_report.py)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = ROOT_DIR / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import basic_plagiarism_check  # noqa: E402

# ─── Argument Handling Tests ──────────────────────────────────────────────────


def test_main_usage_on_wrong_arg_count(capsys):
    """Verify a usage message is printed and the process exits(1)."""
    with patch("sys.argv", ["basic_plagiarism_check.py", "only_one_file.txt"]):
        try:
            basic_plagiarism_check.main()
        except SystemExit as exc:
            assert exc.code == 1
    captured = capsys.readouterr()
    assert "Usage: python examples/basic_plagiarism_check.py <file1> <file2>" in captured.out


def test_main_requires_two_files():
    """Verify main() exits when invoked with too many arguments."""
    with patch(
        "sys.argv",
        ["basic_plagiarism_check.py", "a.txt", "b.txt", "c.txt"],
    ):
        try:
            basic_plagiarism_check.main()
        except SystemExit as exc:
            assert exc.code == 1


def test_missing_file_exits(tmp_path, capsys):
    """Verify a missing input file produces an error and exits(1)."""
    existing = tmp_path / "existing.txt"
    existing.write_text("Some existing text.", encoding="utf-8")
    missing = tmp_path / "missing.txt"

    with patch("sys.argv", ["basic_plagiarism_check.py", str(missing), str(existing)]):
        try:
            basic_plagiarism_check.main()
        except SystemExit as exc:
            assert exc.code == 1
    captured = capsys.readouterr()
    assert "Error: file not found" in captured.out
    assert "missing.txt" in captured.out


# ─── Pipeline Flow Tests ──────────────────────────────────────────────────────


def _make_dummy_matrix(names):
    """Build a 2x2 symmetric similarity DataFrame."""
    return pd.DataFrame(
        [[1.0, 0.85], [0.85, 1.0]],
        index=names,
        columns=names,
    )


def test_main_pipeline_prints_similarity_and_flags(tmp_path, capsys):
    """Verify the full pipeline runs and prints scores + flagged pairs."""
    file1 = tmp_path / "doc1.txt"
    file2 = tmp_path / "doc2.txt"
    file1.write_text("First document text.", encoding="utf-8")
    file2.write_text("Second document text.", encoding="utf-8")

    names = ["doc1.txt", "doc2.txt"]
    dummy_df = _make_dummy_matrix(names)
    dummy_flags = [
        {
            "doc_a": "doc1.txt",
            "doc_b": "doc2.txt",
            "similarity": 0.85,
            "severity": "Medium",
        }
    ]

    with (
        patch("sys.argv", ["basic_plagiarism_check.py", str(file1), str(file2)]),
        patch("basic_plagiarism_check.extract_text", side_effect=lambda data, name: f"text of {name}"),
        patch("basic_plagiarism_check.chunk_text", side_effect=lambda text: [text]),
        patch("basic_plagiarism_check.embed_documents", return_value={"doc1.txt": None, "doc2.txt": None}),
        patch("basic_plagiarism_check.compute_similarity_matrix", return_value=dummy_df),
        patch("basic_plagiarism_check.flag_plagiarism", return_value=dummy_flags),
    ):
        basic_plagiarism_check.main()

    captured = capsys.readouterr()
    assert "Semantic Similarity Matrix" in captured.out
    assert "doc1.txt" in captured.out
    assert "doc2.txt" in captured.out
    assert "Flagged pairs: 1" in captured.out
    assert "doc1.txt <-> doc2.txt: 0.8500 (Medium)" in captured.out


def test_main_pipeline_no_flags(tmp_path, capsys):
    """Verify the no-flag message is printed when nothing crosses the threshold."""
    file1 = tmp_path / "doc1.txt"
    file2 = tmp_path / "doc2.txt"
    file1.write_text("First document text.", encoding="utf-8")
    file2.write_text("Second document text.", encoding="utf-8")

    dummy_df = _make_dummy_matrix(["doc1.txt", "doc2.txt"])

    with (
        patch("sys.argv", ["basic_plagiarism_check.py", str(file1), str(file2)]),
        patch("basic_plagiarism_check.extract_text", return_value="text"),
        patch("basic_plagiarism_check.chunk_text", side_effect=lambda text: [text]),
        patch("basic_plagiarism_check.embed_documents", return_value={"doc1.txt": None, "doc2.txt": None}),
        patch("basic_plagiarism_check.compute_similarity_matrix", return_value=dummy_df),
        patch("basic_plagiarism_check.flag_plagiarism", return_value=[]),
    ):
        basic_plagiarism_check.main()

    captured = capsys.readouterr()
    assert "No pairs exceeded the plagiarism threshold." in captured.out
