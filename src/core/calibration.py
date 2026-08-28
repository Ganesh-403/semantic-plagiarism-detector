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
src/core/calibration.py
-----------------------
Automated threshold calibration and backtest harness (Issue #2267).

Detection thresholds such as ``PLAGIARISM_THRESHOLD`` and the severity
boundaries have historically been hard-coded in ``src.core.config``.  This
module turns ground truth into tuned configuration: it sweeps candidate
threshold values against a labelled corpus, reports precision / recall /
F1 / confusion-matrix metrics at every grid point, and produces a
recommended config that the ``src.core.config`` loader can consume.

All functions here are pure-Python and dependency-free so the harness stays
lightweight and unit-testable without loading the ML stack (torch / faiss /
sentence-transformers).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.core.config import DEFAULT_THRESHOLDS, SimilarityThresholds

logger = logging.getLogger(__name__)

# Recognised positive / negative label strings.  The set is deliberately
# broad so CSV files exported by different tools (0/1, True/False,
# "plagiarized"/"original", "yes"/"no", ...) all work without preprocessing.
_POSITIVE_LABELS = {
    "1",
    "true",
    "yes",
    "y",
    "plagiarized",
    "plagiarism",
    "plagiarised",
    "positive",
    "pos",
}
_NEGATIVE_LABELS = {
    "0",
    "false",
    "no",
    "n",
    "original",
    "clean",
    "not_plagiarized",
    "not-plagiarized",
    "not plagiarized",
    "negative",
    "neg",
}


def as_label(value: Any) -> int:
    """Normalize a ground-truth label to a binary 0/1 integer.

    Accepts booleans, numbers, and common textual representations such as
    ``"plagiarized"`` / ``"original"`` / ``"yes"`` / ``"no"``.

    Raises:
        ValueError: If the value cannot be mapped to a binary label.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text in _POSITIVE_LABELS:
        return 1
    if text in _NEGATIVE_LABELS:
        return 0
    raise ValueError(f"Unrecognized label: {value!r}")


def _clamp(value: float) -> float:
    """Clamp a score to the inclusive ``[0.0, 1.0]`` range."""
    return min(1.0, max(0.0, float(value)))


def f1_score(precision: float, recall: float) -> float:
    """Return the harmonic mean of *precision* and *recall*.

    Returns ``0.0`` when both values are zero (undefined harmonic mean).
    """
    precision = float(precision)
    recall = float(recall)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def confusion_matrix(
    scores: Sequence[float],
    labels: Sequence[Any],
    threshold: float,
) -> dict[str, int]:
    """Build the ``{tp, fp, fn, tn}`` confusion matrix at a threshold.

    A prediction is positive when ``score >= threshold`` (inclusive, matching
    the behaviour of ``src.core.config.is_plagiarism``).

    Args:
        scores: Predicted similarity scores.
        labels: Ground-truth labels (see :func:`as_label`).
        threshold: Decision boundary.

    Returns:
        Dict with ``tp``, ``fp``, ``fn`` and ``tn`` integer counts.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")

    tp = fp = fn = tn = 0
    for score, label in zip(scores, labels):
        predicted = 1 if float(score) >= float(threshold) else 0
        actual = as_label(label)
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 1:
            fp += 1
        elif actual == 1:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate_thresholds(
    results: Sequence[float],
    labels: Sequence[Any],
    grid: Sequence[float],
) -> list[dict[str, Any]]:
    """Evaluate detection quality across candidate thresholds.

    Sweeps every threshold in *grid* against the predicted *results* and the
    ground-truth *labels*, producing precision, recall, F1, accuracy, and the
    confusion matrix at each grid point.

    Args:
        results: Predicted similarity scores (one per labelled pair).
        labels: Ground-truth flags (see :func:`as_label`).
        grid: Candidate threshold values to sweep (sorted ascending output).

    Returns:
        A list of metric dicts, one per threshold, sorted by threshold:

        ``{"threshold", "precision", "recall", "f1", "accuracy",
        "tp", "fp", "fn", "tn"}``

    Raises:
        ValueError: If ``results`` and ``labels`` have different lengths.
    """
    scores = [float(s) for s in results]
    y = [as_label(l) for l in labels]
    if len(scores) != len(y):
        raise ValueError("results and labels must have the same length")

    evaluations: list[dict[str, Any]] = []
    for raw_threshold in grid:
        threshold = float(raw_threshold)
        counts = confusion_matrix(scores, y, threshold)
        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        evaluations.append(
            {
                "threshold": round(threshold, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1_score(precision, recall), 6),
                "accuracy": round(accuracy, 6),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    evaluations.sort(key=lambda row: row["threshold"])
    return evaluations


def best_threshold(evaluations: Sequence[Mapping[str, Any]]) -> float:
    """Return the grid threshold that maximizes F1.

    Ties are broken deterministically: higher precision, then higher recall,
    then the higher (more conservative) threshold.  Preferring the higher
    threshold on a tie avoids recommending an unnecessarily aggressive cutoff
    when several thresholds perform identically on the calibration data.

    Args:
        evaluations: Per-threshold metric rows from :func:`evaluate_thresholds`.

    Returns:
        The threshold value (a grid point) that maximizes F1.

    Raises:
        ValueError: If *evaluations* is empty.
    """
    if not evaluations:
        raise ValueError("evaluations must not be empty")

    best = max(
        evaluations,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            float(row["recall"]),
            float(row["threshold"]),
        ),
    )
    return float(best["threshold"])


def best_evaluation(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the full metric row at the threshold selected by :func:`best_threshold`."""
    threshold = best_threshold(evaluations)
    for row in evaluations:
        if abs(float(row["threshold"]) - threshold) < 1e-9:
            return dict(row)
    return {}


def build_recommendation(
    best_row: Mapping[str, Any],
    current_thresholds: SimilarityThresholds = DEFAULT_THRESHOLDS,
    *,
    source: Optional[str] = None,
    dataset: Optional[str] = None,
    score_column: Optional[str] = None,
    samples: Optional[int] = None,
    n_plagiarized: Optional[int] = None,
    sweep: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Build a config-loader-compatible threshold recommendation dict.

    The returned dict uses the same top-level keys (``plagiarism``, ``medium``,
    ``high``) that :func:`src.core.config.load_threshold_config` understands,
    with a ``calibration`` block holding the backtest metadata used by the
    dashboard's calibration report.

    Args:
        best_row: Metric row at the best threshold.
        current_thresholds: Base thresholds whose severity boundaries
            (``medium`` / ``high``) are preserved.
        source: Free-form description of where the recommendation came from.
        dataset: Path/name of the labelled dataset used for calibration.
        score_column: Score column name the sweep was run over.
        samples: Number of labelled pairs in the dataset.
        n_plagiarized: Number of plagiarized (positive) pairs.
        sweep: Optional full sweep (all grid points) to embed for reporting.

    Returns:
        Recommendation dict, JSON/YAML-serialisable.
    """
    recommended_threshold = float(best_row["threshold"])
    calibration: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "recommended_threshold": round(recommended_threshold, 6),
        "precision": round(float(best_row["precision"]), 6),
        "recall": round(float(best_row["recall"]), 6),
        "f1": round(float(best_row["f1"]), 6),
        "accuracy": round(float(best_row["accuracy"]), 6),
        "confusion_matrix": {
            "tp": int(best_row["tp"]),
            "fp": int(best_row["fp"]),
            "fn": int(best_row["fn"]),
            "tn": int(best_row["tn"]),
        },
    }
    for key, value in (
        ("source", source),
        ("dataset", dataset),
        ("score_column", score_column),
        ("samples", samples),
        ("n_plagiarized", n_plagiarized),
    ):
        if value is not None:
            calibration[key] = value

    if sweep is not None:
        calibration["sweep"] = [dict(row) for row in sweep]

    return {
        "plagiarism": round(recommended_threshold, 6),
        "medium": round(float(current_thresholds.medium), 6),
        "high": round(float(current_thresholds.high), 6),
        "calibration": calibration,
    }


def write_recommended_config(
    recommendation: Mapping[str, Any],
    path: str,
    fmt: str = "json",
) -> str:
    """Serialize a recommendation dict to disk.

    ``json`` is always supported; ``yaml`` is used when PyYAML is installed and
    silently falls back to JSON otherwise.

    Args:
        recommendation: Dict from :func:`build_recommendation`.
        path: Destination file path.
        fmt: Output format, ``"json"`` or ``"yaml"``.

    Returns:
        The path that was written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if fmt.lower() == "yaml":
        try:
            import yaml
        except ImportError:
            logger.warning(
                "PyYAML is not installed; writing '%s' as JSON instead.",
                destination,
            )
        else:
            with open(destination, "w", encoding="utf-8") as stream:
                yaml.safe_dump(
                    dict(recommendation),
                    stream,
                    sort_keys=False,
                    default_flow_style=False,
                )
            logger.info("Wrote recommended threshold config to %s", destination)
            return str(destination)

    with open(destination, "w", encoding="utf-8") as stream:
        json.dump(dict(recommendation), stream, indent=2, ensure_ascii=False)
    logger.info("Wrote recommended threshold config to %s", destination)
    return str(destination)


def load_calibration_report(path: str) -> Optional[dict[str, Any]]:
    """Load the ``calibration`` metadata block from a recommendation file.

    Returns ``None`` when the file is missing, unreadable, or contains no
    calibration block, so callers can render a graceful empty state.
    """
    report_path = Path(path)
    try:
        if report_path.suffix.lower() == ".yaml":
            import yaml

            data = yaml.safe_load(report_path.read_text(encoding="utf-8"))
        else:
            data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError, ImportError):
        return None

    if not isinstance(data, dict):
        return None
    return data.get("calibration")


def _clamp_score(value: float) -> float:
    """Normalize an evaluation score into the ``[0.0, 1.0]`` range.

    Provided as a convenience for callers that collect raw model scores.
    """
    return _clamp(value)


__all__ = [
    "as_label",
    "best_evaluation",
    "best_threshold",
    "build_recommendation",
    "confusion_matrix",
    "evaluate_thresholds",
    "f1_score",
    "load_calibration_report",
    "write_recommended_config",
]
