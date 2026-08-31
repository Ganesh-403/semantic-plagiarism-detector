"""
src/core/threshold_calibration.py
----------------------------------
Threshold calibration pipeline using labeled evaluation data to compute
precision, recall, F1, and other classification metrics across candidate
thresholds (Issue #3912).

A labeled dataset maps document pairs to plagiarism/non-plagiarism labels.
This module sweeps through candidate thresholds and calculates metrics to
support informed threshold selection.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Classification metrics for a single threshold value."""

    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


@dataclass
class CalibrationResult:
    """Results from a complete calibration run."""

    calibration_id: str
    timestamp: str
    dataset_name: str
    score_type: str
    metrics: List[CalibrationMetrics]
    optimal_threshold: float
    optimal_f1: float
    metadata: Dict[str, Any]


def _compute_confusion_matrix(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Tuple[int, int, int, int]:
    """
    Compute TP, TN, FP, FN for a single threshold.
    
    Args:
        scores: Array of similarity scores.
        labels: Array of binary labels (1 = plagiarism, 0 = no plagiarism).
        threshold: Classification threshold.
    
    Returns:
        Tuple of (TP, TN, FP, FN).
    """
    predictions = (scores >= threshold).astype(int)
    
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    return int(tp), int(tn), int(fp), int(fn)


def compute_calibration_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> List[CalibrationMetrics]:
    """
    Evaluate classification metrics across candidate thresholds.
    
    Args:
        scores: Array of similarity scores from similarity computation.
        labels: Array of binary labels (1 = plagiarism, 0 = no plagiarism).
        thresholds: List of threshold values to evaluate. If None, auto-generates
                   50 evenly-spaced values from min(scores) to max(scores).
    
    Returns:
        List of CalibrationMetrics, one per threshold, sorted by threshold.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have same length")
    
    if len(scores) == 0:
        logger.warning("Empty scores/labels provided to calibration")
        return []
    
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    
    if thresholds is None:
        thresholds = list(np.linspace(float(np.min(scores)), float(np.max(scores)), 50))
    
    metrics_list = []
    
    for thresh in thresholds:
        tp, tn, fp, fn = _compute_confusion_matrix(scores, labels, thresh)
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1: 2 * (precision * recall) / (precision + recall)
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # False Positive Rate: FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False Negative Rate: FN / (FN + TP)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        metrics_list.append(
            CalibrationMetrics(
                threshold=float(thresh),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                false_positive_rate=float(fpr),
                false_negative_rate=float(fnr),
                true_positives=tp,
                true_negatives=tn,
                false_positives=fp,
                false_negatives=fn,
            )
        )
    
    # Sort by threshold for consistent output
    metrics_list.sort(key=lambda m: m.threshold)
    return metrics_list


def find_optimal_threshold(
    metrics: List[CalibrationMetrics],
    strategy: str = "f1",
) -> Tuple[float, float]:
    """
    Select the best threshold from calibration metrics using specified strategy.
    
    Args:
        metrics: List of CalibrationMetrics from compute_calibration_metrics().
        strategy: Selection strategy - 'f1' (max F1), 'precision' (max precision),
                 'recall' (max recall), or 'balanced' (minimize abs(FPR - FNR)).
    
    Returns:
        Tuple of (optimal_threshold, metric_value).
    """
    if not metrics:
        raise ValueError("metrics list is empty")
    
    if strategy == "f1":
        best = max(metrics, key=lambda m: m.f1)
        return best.threshold, best.f1
    elif strategy == "precision":
        best = max(metrics, key=lambda m: m.precision)
        return best.threshold, best.precision
    elif strategy == "recall":
        best = max(metrics, key=lambda m: m.recall)
        return best.threshold, best.recall
    elif strategy == "balanced":
        best = min(metrics, key=lambda m: abs(m.false_positive_rate - m.false_negative_rate))
        return best.threshold, abs(best.false_positive_rate - best.false_negative_rate)
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")


def save_calibration_result(
    result: CalibrationResult,
    output_path: Optional[str] = None,
) -> str:
    """
    Save calibration result to JSON, versioned by calibration_id.
    
    Args:
        result: CalibrationResult to persist.
        output_path: Path to calibrated_thresholds.json. If None, uses
                    config/calibrated_thresholds.json.
    
    Returns:
        Path where the result was saved.
    """
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "config",
            "calibrated_thresholds.json",
        )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load existing calibrations or start fresh
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to load existing calibrations: %s. Starting fresh.", e)
            data = {"calibrations": {}}
    else:
        data = {"calibrations": {}}
    
    # Store this calibration
    data["calibrations"][result.calibration_id] = {
        "timestamp": result.timestamp,
        "dataset_name": result.dataset_name,
        "score_type": result.score_type,
        "optimal_threshold": result.optimal_threshold,
        "optimal_f1": result.optimal_f1,
        "metrics": [
            {
                "threshold": m.threshold,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "false_positive_rate": m.false_positive_rate,
                "false_negative_rate": m.false_negative_rate,
                "tp": m.true_positives,
                "tn": m.true_negatives,
                "fp": m.false_positives,
                "fn": m.false_negatives,
            }
            for m in result.metrics
        ],
        "metadata": result.metadata,
    }
    
    # Mark as latest
    data["latest_calibration_id"] = result.calibration_id
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    logger.info("Saved calibration result to %s (id=%s)", output_path, result.calibration_id)
    return output_path


def calibrate_thresholds(
    similarity_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    calibration_id: Optional[str] = None,
    score_type: str = "semantic",
    dataset_name: str = "evaluation_set",
    thresholds_to_eval: Optional[List[float]] = None,
    strategy: str = "f1",
    save: bool = True,
) -> CalibrationResult:
    """
    End-to-end calibration pipeline: evaluate multiple thresholds, select optimal,
    and optionally save to disk.
    
    Args:
        similarity_df: N×N DataFrame of similarity scores.
        labels_df: N×N DataFrame of binary labels (1/0). Same shape as similarity_df.
                  1 = pair is plagiarism, 0 = pair is not plagiarism.
        calibration_id: Unique identifier for this calibration. If None, auto-generates
                       from timestamp.
        score_type: Type of score being calibrated ('semantic', 'lexical', 'hybrid').
        dataset_name: Name of the evaluation dataset for logging.
        thresholds_to_eval: Candidate thresholds to sweep. If None, auto-generates 50.
        strategy: Threshold selection strategy ('f1', 'precision', 'recall', 'balanced').
        save: Whether to save result to disk.
    
    Returns:
        CalibrationResult with metrics and optimal threshold.
    """
    if similarity_df.shape != labels_df.shape:
        raise ValueError("similarity_df and labels_df must have same shape")
    
    if calibration_id is None:
        calibration_id = f"cal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Flatten upper triangle (exclude diagonal self-similarity)
    n = similarity_df.shape[0]
    scores_flat = []
    labels_flat = []
    
    for i in range(n):
        for j in range(i + 1, n):
            scores_flat.append(float(similarity_df.iloc[i, j]))
            labels_flat.append(int(labels_df.iloc[i, j]))
    
    scores_arr = np.array(scores_flat, dtype=float)
    labels_arr = np.array(labels_flat, dtype=int)
    
    # Compute metrics across thresholds
    metrics = compute_calibration_metrics(scores_arr, labels_arr, thresholds_to_eval)
    
    # Find optimal threshold
    optimal_thresh, optimal_val = find_optimal_threshold(metrics, strategy=strategy)
    
    result = CalibrationResult(
        calibration_id=calibration_id,
        timestamp=datetime.now().isoformat(),
        dataset_name=dataset_name,
        score_type=score_type,
        metrics=metrics,
        optimal_threshold=optimal_thresh,
        optimal_f1=metrics[[m.threshold for m in metrics].index(optimal_thresh)].f1,
        metadata={
            "selection_strategy": strategy,
            "num_pairs": len(scores_arr),
            "num_positives": int(np.sum(labels_arr)),
            "num_negatives": int(len(labels_arr) - np.sum(labels_arr)),
        },
    )
    
    if save:
        save_calibration_result(result)
    
    return result


def export_calibration_metrics_csv(
    result: CalibrationResult,
    output_path: str,
) -> str:
    """
    Export calibration metrics to CSV for analysis.
    
    Args:
        result: CalibrationResult to export.
        output_path: Path to write CSV file.
    
    Returns:
        Path where CSV was written.
    """
    df = pd.DataFrame([
        {
            "threshold": m.threshold,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "fpr": m.false_positive_rate,
            "fnr": m.false_negative_rate,
            "tp": m.true_positives,
            "tn": m.true_negatives,
            "fp": m.false_positives,
            "fn": m.false_negatives,
        }
        for m in result.metrics
    ])
    
    df.to_csv(output_path, index=False)
    logger.info("Exported calibration metrics to %s", output_path)
    return output_path