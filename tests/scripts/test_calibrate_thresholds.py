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
tests/scripts/test_calibrate_thresholds.py
------------------------------------------
Unit tests for the threshold calibration & backtest harness CLI
(``scripts/calibrate_thresholds.py``, Issue #2267).

Validates CLI parsing, CSV label normalization, grid generation, the F1
sweep, and the end-to-end acceptance criterion: running the script on a
small labeled CSV prints an F1 table and writes a recommended config.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add scripts directory to path (mirrors other script test modules).
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import calibrate_thresholds as calibrate  # noqa: E402


def _write_labeled_csv(tmp_path: Path, *, labels=(1, 1, 0, 0, 1)) -> Path:
    """Write a small labeled CSV with precomputed scores."""
    csv_path = tmp_path / "labeled_pairs.csv"
    n = len(labels)
    rows = []
    for i, label in enumerate(labels):
        # Scores separate plagiarized (high) from original (low) pairs.
        score = 0.55 + 0.4 * (i / max(1, n - 1)) if label == 1 else 0.1 + 0.2 * (i % 3)
        rows.append(
            {
                "doc_a": f"doc_a_{i}",
                "doc_b": f"doc_b_{i}",
                "label": label,
                "score": round(score, 4),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


# ─── CLI argument parsing ─────────────────────────────────────────────────────


def test_parse_arguments_defaults():
    with patch("sys.argv", ["calibrate_thresholds.py", "--csv", "pairs.csv"]):
        args = calibrate.parse_arguments()

    assert args.csv == "pairs.csv"
    assert args.score_column is None
    assert args.compute_scores is False
    assert args.grid_start == 0.30
    assert args.grid_stop == 0.96
    assert args.grid_step == 0.01
    assert args.output.endswith("thresholds.recommended.json")


def test_parse_arguments_custom():
    test_args = [
        "calibrate_thresholds.py",
        "--csv",
        "data/pairs.csv",
        "--score-column",
        "hybrid",
        "--grid-start",
        "0.1",
        "--grid-stop",
        "0.9",
        "--grid-step",
        "0.05",
        "--medium",
        "0.7",
        "--high",
        "0.95",
        "--output",
        "out/rec.json",
        "--format",
        "yaml",
    ]
    with patch("sys.argv", test_args):
        args = calibrate.parse_arguments()

    assert args.score_column == "hybrid"
    assert args.grid_start == 0.1
    assert args.grid_stop == 0.9
    assert args.grid_step == 0.05
    assert args.medium == 0.7
    assert args.high == 0.95
    assert args.output == "out/rec.json"
    assert args.format == "yaml"


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--grid-start", "-0.5"],
        ["--grid-start", "1.2"],
        ["--grid-stop", "1.5"],
        ["--grid-stop", "0.0"],
        ["--grid-step", "0.0"],
        ["--grid-step", "-0.1"],
        ["--grid-start", "0.7", "--grid-stop", "0.5"],
    ],
)
def test_parse_arguments_rejects_invalid_grid(bad_args):
    test_args = ["calibrate_thresholds.py", "--csv", "pairs.csv", *bad_args]
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit):
            calibrate.parse_arguments()


def test_parse_arguments_requires_csv():
    with patch("sys.argv", ["calibrate_thresholds.py"]):
        with pytest.raises(SystemExit):
            calibrate.parse_arguments()


# ─── Grid generation ──────────────────────────────────────────────────────────


def test_build_grid_inclusive_start_exclusive_stop():
    grid = calibrate.build_grid(0.0, 0.5, 0.1)
    assert grid == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert 0.5 not in grid


# ─── CSV loading ──────────────────────────────────────────────────────────────


def test_load_labeled_dataset_normalizes_labels(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {"doc_a": "a", "doc_b": "b", "label": "plagiarized", "score": 0.9},
            {"doc_a": "c", "doc_b": "d", "label": "original", "score": 0.1},
            {"doc_a": "e", "doc_b": "f", "label": 1, "score": 0.8},
            {"doc_a": "g", "doc_b": "h", "label": "0", "score": 0.2},
        ]
    ).to_csv(csv_path, index=False)

    df = calibrate.load_labeled_dataset(str(csv_path))

    assert df["label_int"].tolist() == [1, 0, 1, 0]


def test_load_labeled_dataset_missing_columns(tmp_path):
    csv_path = tmp_path / "missing.csv"
    pd.DataFrame([{"doc_a": "a", "doc_b": "b"}]).to_csv(csv_path, index=False)

    with pytest.raises(SystemExit, match="missing required column"):
        calibrate.load_labeled_dataset(str(csv_path))


def test_load_labeled_dataset_missing_file():
    with pytest.raises(SystemExit, match="does not exist"):
        calibrate.load_labeled_dataset("does/not/exist.csv")


# ─── Score column detection & sweep ───────────────────────────────────────────


def test_detect_score_columns_prefers_known_order(tmp_path):
    csv_path = tmp_path / "cols.csv"
    pd.DataFrame(
        [
            {"doc_a": "a", "doc_b": "b", "label": 1, "hybrid": 0.9, "ai": 0.8},
            {"doc_a": "c", "doc_b": "d", "label": 0, "hybrid": 0.2, "ai": 0.1},
        ]
    ).to_csv(csv_path, index=False)
    df = calibrate.load_labeled_dataset(str(csv_path))

    assert calibrate.detect_score_columns(df) == ["hybrid", "ai"]


def test_run_calibration_sweeps_scores(tmp_path):
    csv_path = _write_labeled_csv(tmp_path)
    df = calibrate.load_labeled_dataset(str(csv_path))
    grid = calibrate.build_grid(0.0, 1.0, 0.1)

    evaluations = calibrate.run_calibration(df, "score", grid)

    assert len(evaluations) == len(grid)
    assert evaluations == sorted(evaluations, key=lambda r: r["threshold"])
    assert max(r["f1"] for r in evaluations) > 0.0


# ─── End-to-end acceptance criterion ──────────────────────────────────────────


def test_main_prints_f1_table_and_writes_config(tmp_path, capsys):
    """Acceptance criterion: running the script on a small labeled CSV prints
    an F1 table and writes a recommended config."""
    csv_path = _write_labeled_csv(tmp_path)
    output = tmp_path / "thresholds.recommended.json"

    test_args = [
        "calibrate_thresholds.py",
        "--csv",
        str(csv_path),
        "--grid-start",
        "0.1",
        "--grid-stop",
        "0.9",
        "--grid-step",
        "0.05",
        "--output",
        str(output),
    ]
    with patch("sys.argv", test_args):
        calibrate.main()

    captured = capsys.readouterr()

    # Prints an F1 table.
    assert "THRESHOLD SWEEP" in captured.out
    assert "Precision" in captured.out
    assert "Recommended threshold" in captured.out

    # Writes a recommended config compatible with the config loader.
    assert output.exists()
    with open(output, encoding="utf-8") as stream:
        data = json.load(stream)

    assert set(("plagiarism", "medium", "high")) <= set(data)
    assert "calibration" in data
    calibration = data["calibration"]
    assert "recommended_threshold" in calibration
    assert calibration["recommended_threshold"] == data["plagiarism"]
    assert calibration["dataset"] == str(csv_path)
    assert "sweep" in calibration and calibration["sweep"]

    from src.core.config import load_threshold_config

    loaded = load_threshold_config(str(output))
    assert loaded.plagiarism == data["plagiarism"]


def test_main_writes_report_csv(tmp_path):
    csv_path = _write_labeled_csv(tmp_path)
    output = tmp_path / "thresholds.recommended.json"
    report = tmp_path / "sweep.csv"

    test_args = [
        "calibrate_thresholds.py",
        "--csv",
        str(csv_path),
        "--output",
        str(output),
        "--report",
        str(report),
    ]
    with patch("sys.argv", test_args):
        calibrate.main()

    assert report.exists()
    report_df = pd.read_csv(report)
    assert {"threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"} <= set(
        report_df.columns
    )


def test_main_fails_without_score_column(tmp_path):
    csv_path = tmp_path / "no_scores.csv"
    pd.DataFrame(
        [
            {"doc_a": "a", "doc_b": "b", "label": 1},
            {"doc_a": "c", "doc_b": "d", "label": 0},
        ]
    ).to_csv(csv_path, index=False)

    test_args = [
        "calibrate_thresholds.py",
        "--csv",
        str(csv_path),
    ]
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit, match="no recognized score column"):
            calibrate.main()


def test_main_preserves_medium_high_boundaries(tmp_path):
    csv_path = _write_labeled_csv(tmp_path)
    output = tmp_path / "thresholds.recommended.json"

    test_args = [
        "calibrate_thresholds.py",
        "--csv",
        str(csv_path),
        "--medium",
        "0.72",
        "--high",
        "0.93",
        "--output",
        str(output),
    ]
    with patch("sys.argv", test_args):
        calibrate.main()

    with open(output, encoding="utf-8") as stream:
        data = json.load(stream)
    assert data["medium"] == 0.72
    assert data["high"] == 0.93
