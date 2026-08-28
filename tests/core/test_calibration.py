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
tests/core/test_calibration.py
------------------------------
Unit tests for the automated threshold calibration & backtest harness
(``src/core/calibration.py``, Issue #2267).

Covers synthetic-similarity-score edge cases: all-plagiarized,
none-plagiarized, ties, and the acceptance criterion that
``best_threshold()`` returns the grid point maximizing F1.
"""

from __future__ import annotations

import json

import pytest

from src.core.calibration import (
    as_label,
    best_evaluation,
    best_threshold,
    build_recommendation,
    confusion_matrix,
    evaluate_thresholds,
    f1_score,
    load_calibration_report,
    write_recommended_config,
)
from src.core.config import DEFAULT_THRESHOLDS, load_threshold_config

# ── f1_score ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("precision", "recall", "expected"),
    [
        (1.0, 1.0, 1.0),
        (0.5, 0.5, 0.5),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.9, 0.8, pytest.approx(0.8470588235)),
    ],
)
def test_f1_score(precision, recall, expected):
    assert f1_score(precision, recall) == expected


# ── as_label ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (0, 0),
        (True, 1),
        (False, 0),
        (1.0, 1),
        (0.0, 0),
        ("1", 1),
        ("0", 0),
        ("plagiarized", 1),
        ("original", 0),
        ("PLAGIARIZED", 1),
        ("yes", 1),
        ("no", 0),
        ("true", 1),
        ("false", 0),
    ],
)
def test_as_label_known_values(value, expected):
    assert as_label(value) == expected


@pytest.mark.parametrize("value", ["unknown", "maybe", "highly suspicious", None, "2"])
def test_as_label_rejects_unknown_values(value):
    with pytest.raises(ValueError):
        as_label(value)


# ── confusion_matrix ──────────────────────────────────────────────────────────


def test_confusion_matrix_known_counts():
    scores = [0.2, 0.5, 0.6, 0.9]
    labels = [0, 0, 1, 1]

    assert confusion_matrix(scores, labels, 0.5) == {
        "tp": 2,
        "fp": 1,
        "fn": 0,
        "tn": 1,
    }
    # Threshold boundary is inclusive (score == threshold is flagged).
    assert confusion_matrix(scores, labels, 0.6) == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}
    assert confusion_matrix(scores, labels, 1.1) == {"tp": 0, "fp": 0, "fn": 2, "tn": 2}


def test_confusion_matrix_length_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_matrix([0.1, 0.2], [0], 0.5)


# ── evaluate_thresholds ───────────────────────────────────────────────────────


def test_evaluate_thresholds_known_row():
    scores = [0.2, 0.5, 0.6, 0.9]
    labels = [0, 0, 1, 1]

    rows = evaluate_thresholds(scores, labels, [0.3, 0.5, 0.7])

    assert [r["threshold"] for r in rows] == [0.3, 0.5, 0.7]

    row = next(r for r in rows if r["threshold"] == 0.5)
    assert row["precision"] == pytest.approx(2 / 3)
    assert row["recall"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(0.8)
    assert row["tp"] == 2 and row["fp"] == 1 and row["fn"] == 0 and row["tn"] == 1


def test_evaluate_thresholds_is_sorted_by_threshold():
    rows = evaluate_thresholds([0.5, 0.6], [1, 1], [0.9, 0.1, 0.5])
    assert [r["threshold"] for r in rows] == [0.1, 0.5, 0.9]


def test_evaluate_thresholds_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_thresholds([0.1, 0.2], [0], [0.5])


# ── best_threshold / best_evaluation ──────────────────────────────────────────


def test_best_threshold_returns_f1_maximizing_grid_point():
    """Acceptance criterion: best_threshold returns the grid point max F1."""
    scores = [0.2, 0.5, 0.6, 0.9]
    labels = [0, 0, 1, 1]
    grid = [0.1, 0.3, 0.5, 0.7, 0.9]

    rows = evaluate_thresholds(scores, labels, grid)
    max_f1 = max(r["f1"] for r in rows)

    chosen = best_threshold(rows)
    chosen_row = next(r for r in rows if r["threshold"] == chosen)

    # Chosen threshold must achieve the maximum F1 across the grid.
    assert chosen_row["f1"] == pytest.approx(max_f1)
    assert best_evaluation(rows)["f1"] == pytest.approx(max_f1)


def test_best_threshold_all_plagiarized():
    """Edge case: every pair is plagiarized -> any threshold below the
    minimum score is perfect; the tie-break picks the highest such threshold."""
    scores = [0.7, 0.8, 0.9]
    labels = [1, 1, 1]
    grid = [0.0, 0.5, 0.75, 0.85, 1.0]

    rows = evaluate_thresholds(scores, labels, grid)
    best = best_evaluation(rows)

    assert best["f1"] == pytest.approx(1.0)
    assert best["precision"] == pytest.approx(1.0)
    assert best["recall"] == pytest.approx(1.0)
    # Highest grid threshold that still keeps a perfect F1.
    assert best_threshold(rows) == 0.5


def test_best_threshold_none_plagiarized():
    """Edge case: nothing is plagiarized -> F1 is 0 everywhere and the
    most conservative (highest) threshold is recommended."""
    scores = [0.1, 0.2, 0.3]
    labels = [0, 0, 0]
    grid = [0.0, 0.15, 0.5, 0.9]

    rows = evaluate_thresholds(scores, labels, grid)
    assert all(r["f1"] == 0.0 for r in rows)
    assert best_threshold(rows) == 0.9


def test_best_threshold_ties_are_deterministic():
    """Edge case: multiple thresholds share the max F1. The chosen grid
    point must maximize F1 and be deterministic (highest threshold wins)."""
    scores = [0.2, 0.9]
    labels = [0, 1]
    grid = [0.1, 0.5, 0.8, 0.95]

    rows = evaluate_thresholds(scores, labels, grid)
    max_f1 = max(r["f1"] for r in rows)

    chosen = best_threshold(rows)
    chosen_row = next(r for r in rows if r["threshold"] == chosen)

    assert chosen_row["f1"] == pytest.approx(max_f1)
    # Both 0.5 and 0.8 reach the max F1; the tie-break prefers 0.8.
    assert chosen == 0.8

    # Deterministic across repeated calls.
    assert best_threshold(rows) == best_threshold(rows)


def test_best_threshold_empty_raises():
    with pytest.raises(ValueError):
        best_threshold([])


# ── recommendation build / write / load ───────────────────────────────────────


def test_build_recommendation_shape():
    rows = evaluate_thresholds([0.2, 0.5, 0.6, 0.9], [0, 0, 1, 1], [0.3, 0.5, 0.7])
    best = best_evaluation(rows)

    rec = build_recommendation(best, sweep=rows, dataset="pairs.csv", samples=4)

    # Keys compatible with src.core.config.load_threshold_config.
    assert set(("plagiarism", "medium", "high")) <= set(rec)
    assert rec["plagiarism"] == best["threshold"]
    assert rec["medium"] == DEFAULT_THRESHOLDS.medium
    assert rec["high"] == DEFAULT_THRESHOLDS.high

    calibration = rec["calibration"]
    assert calibration["recommended_threshold"] == best["threshold"]
    assert calibration["dataset"] == "pairs.csv"
    assert calibration["samples"] == 4
    assert len(calibration["sweep"]) == 3
    assert set(("tp", "fp", "fn", "tn")) <= set(calibration["confusion_matrix"])


def test_write_and_load_recommended_config_roundtrip(tmp_path):
    rows = evaluate_thresholds([0.2, 0.5, 0.6, 0.9], [0, 0, 1, 1], [0.3, 0.5, 0.7])
    best = best_evaluation(rows)
    rec = build_recommendation(best, sweep=rows)

    output = tmp_path / "thresholds.recommended.json"
    written = write_recommended_config(rec, str(output))

    assert written == str(output)
    assert output.exists()

    calibration = load_calibration_report(str(output))
    assert calibration is not None
    assert calibration["recommended_threshold"] == best["threshold"]
    assert len(calibration["sweep"]) == 3

    loaded = load_threshold_config(str(output))
    assert loaded.plagiarism == best["threshold"]
    assert loaded.medium == DEFAULT_THRESHOLDS.medium
    assert loaded.high == DEFAULT_THRESHOLDS.high


def test_write_yaml_roundtrip_when_available(tmp_path):
    pytest.importorskip("yaml")
    rec = build_recommendation(
        {
            "threshold": 0.63,
            "precision": 0.9,
            "recall": 0.8,
            "f1": 0.85,
            "accuracy": 0.84,
            "tp": 8,
            "fp": 1,
            "fn": 2,
            "tn": 9,
        }
    )

    output = tmp_path / "thresholds.recommended.yaml"
    written = write_recommended_config(rec, str(output), fmt="yaml")

    assert written == str(output)
    assert output.exists()

    calibration = load_calibration_report(str(output))
    assert calibration is not None
    assert calibration["recommended_threshold"] == 0.63


def test_load_calibration_report_missing_file_returns_none(tmp_path):
    assert load_calibration_report(str(tmp_path / "missing.json")) is None


def test_load_threshold_config_defaults_when_no_file(tmp_path):
    # Missing file -> defaults, behavior unchanged.
    assert load_threshold_config(str(tmp_path / "nope.json")) == DEFAULT_THRESHOLDS


def test_load_threshold_config_invalid_json_falls_back(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{", encoding="utf-8")
    assert load_threshold_config(str(bad)) == DEFAULT_THRESHOLDS


def test_load_threshold_config_out_of_order_values_fall_back(tmp_path):
    cfg = tmp_path / "bad_order.json"
    cfg.write_text(json.dumps({"plagiarism": 0.9, "medium": 0.5}), encoding="utf-8")
    assert load_threshold_config(str(cfg)) == DEFAULT_THRESHOLDS
