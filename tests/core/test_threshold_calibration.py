"""
tests/core/test_threshold_calibration.py
-----------------------------------------
Unit tests for threshold calibration pipeline (Issue #3912).
Verifies metric computation, threshold selection, and result persistence.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.core.threshold_calibration import (
    CalibrationMetrics,
    calibrate_thresholds,
    compute_calibration_metrics,
    export_calibration_metrics_csv,
    find_optimal_threshold,
    save_calibration_result,
)


def test_compute_confusion_matrix_metrics():
    """Verify precision, recall, F1 computation for a known dataset."""
    # Simple case: scores = [0.9, 0.8, 0.3, 0.2], labels = [1, 1, 0, 0]
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 1, 0, 0])
    
    metrics = compute_calibration_metrics(scores, labels, thresholds=[0.5])
    assert len(metrics) == 1
    
    m = metrics[0]
    # At threshold 0.5: predictions = [1, 1, 0, 0]
    # TP=2 (0.9, 0.8), FP=0, FN=0, TN=2
    assert m.true_positives == 2
    assert m.false_positives == 0
    assert m.false_negatives == 0
    assert m.true_negatives == 2
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_compute_calibration_metrics_auto_thresholds():
    """Verify auto-generation of 50 candidate thresholds when none provided."""
    scores = np.linspace(0.0, 1.0, 100)
    labels = np.array([1 if s >= 0.5 else 0 for s in scores])
    
    metrics = compute_calibration_metrics(scores, labels)
    
    assert len(metrics) == 50
    assert metrics[0].threshold < metrics[-1].threshold
    for m in metrics:
        assert 0.0 <= m.f1 <= 1.0
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0


def test_find_optimal_threshold_by_f1():
    """Verify F1-based threshold selection."""
    # Create metrics where F1 peaks at threshold=0.7
    metrics = [
        CalibrationMetrics(0.3, 0.5, 0.9, 0.64, 0.1, 0.1, 9, 1, 9, 1),
        CalibrationMetrics(0.5, 0.7, 0.8, 0.75, 0.2, 0.2, 8, 8, 3, 2),
        CalibrationMetrics(0.7, 0.9, 0.7, 0.78, 0.05, 0.3, 7, 19, 1, 3),
        CalibrationMetrics(0.9, 0.95, 0.5, 0.67, 0.01, 0.5, 5, 99, 1, 5),
    ]
    
    optimal_thresh, optimal_f1 = find_optimal_threshold(metrics, strategy="f1")
    assert optimal_thresh == 0.7
    assert optimal_f1 == 0.78


def test_find_optimal_threshold_by_strategy():
    """Verify different selection strategies produce different results."""
    metrics = [
        CalibrationMetrics(0.5, 0.8, 0.6, 0.69, 0.2, 0.4, 6, 8, 2, 4),
        CalibrationMetrics(0.7, 0.6, 0.8, 0.69, 0.4, 0.2, 8, 6, 4, 2),
    ]
    
    thresh_f1, _ = find_optimal_threshold(metrics, strategy="f1")
    thresh_precision, _ = find_optimal_threshold(metrics, strategy="precision")
    thresh_recall, _ = find_optimal_threshold(metrics, strategy="recall")
    
    assert thresh_f1 == 0.5 or thresh_f1 == 0.7
    assert thresh_precision == 0.5
    assert thresh_recall == 0.7


def test_calibrate_thresholds_end_to_end():
    """Verify end-to-end calibration pipeline."""
    # Create a 5×5 similarity matrix
    sim_df = pd.DataFrame(
        np.array([
            [1.00, 0.95, 0.60, 0.20, 0.10],
            [0.95, 1.00, 0.55, 0.25, 0.15],
            [0.60, 0.55, 1.00, 0.30, 0.20],
            [0.20, 0.25, 0.30, 1.00, 0.80],
            [0.10, 0.15, 0.20, 0.80, 1.00],
        ]),
        index=[f"d{i}" for i in range(5)],
        columns=[f"d{i}" for i in range(5)],
    )
    
    # Labels: pairs (0,1), (0,2), (1,2) are plagiarism (1); others not (0)
    labels_df = pd.DataFrame(
        np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ]),
        index=[f"d{i}" for i in range(5)],
        columns=[f"d{i}" for i in range(5)],
    )
    
    result = calibrate_thresholds(
        sim_df,
        labels_df,
        calibration_id="test_cal_001",
        save=False,
    )
    
    assert result.calibration_id == "test_cal_001"
    assert len(result.metrics) > 0
    assert result.optimal_threshold >= 0.0
    assert result.optimal_threshold <= 1.0
    assert result.optimal_f1 >= 0.0
    assert result.optimal_f1 <= 1.0
    assert result.metadata["num_pairs"] == 10  # upper triangle


def test_save_and_load_calibration_result():
    """Verify calibration results persist to JSON."""
    sim_df = pd.DataFrame(
        np.random.rand(3, 3),
        index=["d1", "d2", "d3"],
        columns=["d1", "d2", "d3"],
    )
    labels_df = pd.DataFrame(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        index=["d1", "d2", "d3"],
        columns=["d1", "d2", "d3"],
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "cal_test.json")
        
        result = calibrate_thresholds(
            sim_df,
            labels_df,
            calibration_id="test_persist",
            save=False,
        )
        
        save_calibration_result(result, output_path=output_path)
        
        # Verify file was created
        assert os.path.exists(output_path)
        
        # Verify content can be loaded
        import json
        with open(output_path, "r") as f:
            data = json.load(f)
        
        assert "test_persist" in data["calibrations"]
        assert data["latest_calibration_id"] == "test_persist"


def test_export_calibration_metrics_csv():
    """Verify CSV export of calibration metrics."""
    sim_df = pd.DataFrame(
        np.random.rand(4, 4),
        index=[f"d{i}" for i in range(4)],
        columns=[f"d{i}" for i in range(4)],
    )
    labels_df = pd.DataFrame(
        np.ones((4, 4), dtype=int),
        index=[f"d{i}" for i in range(4)],
        columns=[f"d{i}" for i in range(4)],
    )
    
    result = calibrate_thresholds(sim_df, labels_df, save=False)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "metrics.csv")
        export_calibration_metrics_csv(result, csv_path)
        
        assert os.path.exists(csv_path)
        
        # Verify CSV structure
        df = pd.read_csv(csv_path)
        assert len(df) == len(result.metrics)
        assert "threshold" in df.columns
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1" in df.columns


def test_calibrate_thresholds_deterministic():
    """Verify same dataset always produces same calibration results."""
    sim_df = pd.DataFrame(
        np.array([
            [1.0, 0.8, 0.5],
            [0.8, 1.0, 0.6],
            [0.5, 0.6, 1.0],
        ]),
        index=["d1", "d2", "d3"],
        columns=["d1", "d2", "d3"],
    )
    labels_df = pd.DataFrame(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        index=["d1", "d2", "d3"],
        columns=["d1", "d2", "d3"],
    )
    
    result1 = calibrate_thresholds(sim_df, labels_df, save=False)
    result2 = calibrate_thresholds(sim_df, labels_df, save=False)
    
    assert result1.optimal_threshold == result2.optimal_threshold
    assert result1.optimal_f1 == result2.optimal_f1
    assert len(result1.metrics) == len(result2.metrics)