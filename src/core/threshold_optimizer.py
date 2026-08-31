"""
Automated Threshold Optimization for Different Document Types.

Provides functionality to find optimal plagiarism detection thresholds
based on document characteristics, dataset homogeneity, and user feedback.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar  # noqa: F401
from sklearn.metrics import auc, precision_recall_curve, roc_curve  # noqa: F401

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class ThresholdConfig:
    """Configuration for threshold optimization."""

    # Base thresholds
    plagiarism: float = 0.59
    medium: float = 0.75
    high: float = 0.90

    # Optimization parameters
    min_threshold: float = 0.10
    max_threshold: float = 0.99
    step: float = 0.01

    # Weights for optimization
    precision_weight: float = 0.5
    recall_weight: float = 0.5
    f1_weight: float = 0.7

    # Document type specific adjustments
    homogeneous_bonus: float = 0.05  # Bonus for homogeneous datasets
    heterogeneous_penalty: float = 0.05  # Penalty for heterogeneous

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plagiarism": self.plagiarism,
            "medium": self.medium,
            "high": self.high,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "step": self.step,
            "precision_weight": self.precision_weight,
            "recall_weight": self.recall_weight,
            "f1_weight": self.f1_weight,
            "homogeneous_bonus": self.homogeneous_bonus,
            "heterogeneous_penalty": self.heterogeneous_penalty,
        }


@dataclass
class OptimizationResult:
    """Result of threshold optimization."""

    optimal_threshold: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    document_type: str
    method: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# DOCUMENT TYPE DETECTION
# ============================================================================


def detect_document_type(scores: List[float], texts: List[str]) -> str:
    """
    Detect document type based on similarity distribution and content.

    Args:
        scores: List of similarity scores
        texts: List of document texts

    Returns:
        Document type: 'homogeneous', 'heterogeneous', 'mixed', 'unknown'
    """
    if not scores or len(scores) < 2:
        return "unknown"

    # Analyze score distribution
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Coefficient of variation (normalized spread)
    cv = std_score / mean_score if mean_score > 0 else 0

    # Analyze skewness
    skewness = 0
    if len(scores) > 3:
        from scipy.stats import skew

        skewness = skew(scores)

    # Determine document type
    if cv < 0.15 and skewness > 0.5:
        return "homogeneous"  # Tight distribution, many high scores
    elif cv > 0.4:
        return "heterogeneous"  # Wide distribution
    else:
        return "mixed"


def detect_document_homogeneity(scores: list[float]) -> float:
    """
    Calculate document homogeneity score.

    Args:
        scores: List of similarity scores

    Returns:
        Homogeneity score between 0 and 1
    """
    if not scores:
        return 0.0

    mean_score = np.mean(scores)
    std_score = np.std(scores) if len(scores) > 1 else 0

    # Lower standard deviation = more homogeneous
    homogeneity = 1.0 - min(1.0, std_score / (mean_score + 0.01))
    return homogeneity


def detect_document_complexity(texts: list[str]) -> float:
    """
    Estimate document complexity based on text length and vocabulary.

    Args:
        texts: List of document texts

    Returns:
        Complexity score between 0 and 1
    """
    if not texts:
        return 0.5

    total_words = 0
    total_chars = 0
    unique_words = set()

    for text in texts:
        if isinstance(text, str):
            words = text.split()
            total_words += len(words)
            total_chars += len(text)
            unique_words.update(w.lower() for w in words)

    if total_words == 0:
        return 0.5

    # Vocabulary richness
    vocab_ratio = len(unique_words) / total_words if total_words > 0 else 0

    # Length factor (normalized)
    avg_length = total_chars / len(texts) if texts else 0
    length_factor = min(1.0, avg_length / 1000)

    # Complexity score
    complexity = 0.5 * vocab_ratio + 0.3 * length_factor + 0.2 * 0.5
    return min(1.0, max(0.0, complexity))


# ============================================================================
# THRESHOLD OPTIMIZATION ALGORITHMS
# ============================================================================


def optimize_threshold_f1(
    scores: list[float],
    labels: list[int],
    min_threshold: float = 0.10,
    max_threshold: float = 0.99,
    step: float = 0.01,
) -> Tuple[float, float, float, float]:
    """
    Optimize threshold using F1 score.

    Args:
        scores: List of similarity scores
        labels: List of ground truth labels (1 = plagiarism, 0 = not)
        min_threshold: Minimum threshold to consider
        max_threshold: Maximum threshold to consider
        step: Step size for threshold sweep

    Returns:
        Tuple of (optimal_threshold, precision, recall, f1)
    """
    if not scores or not labels or len(scores) != len(labels):
        return 0.59, 0.0, 0.0, 0.0

    best_threshold = 0.59
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    thresholds = np.arange(min_threshold, max_threshold + step, step)

    for threshold in thresholds:
        predictions = [1 if s >= threshold else 0 for s in scores]

        # Calculate metrics
        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)  # noqa: E741
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)  # noqa: E741
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)  # noqa: E741

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return best_threshold, best_precision, best_recall, best_f1


def optimize_threshold_roc(
    scores: list[float],
    labels: list[int],
    min_threshold: float = 0.10,
    max_threshold: float = 0.99,
    step: float = 0.01,
) -> Tuple[float, float]:
    """
    Optimize threshold using ROC curve (Youden's J statistic).

    Args:
        scores: List of similarity scores
        labels: List of ground truth labels
        min_threshold: Minimum threshold
        max_threshold: Maximum threshold
        step: Step size

    Returns:
        Tuple of (optimal_threshold, roc_auc)
    """
    if not scores or not labels or len(scores) != len(labels):
        return 0.59, 0.0

    thresholds = np.arange(min_threshold, max_threshold + step, step)
    best_threshold = 0.59
    best_youden = 0.0

    for threshold in thresholds:
        predictions = [1 if s >= threshold else 0 for s in scores]

        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)  # noqa: E741
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)  # noqa: E741
        tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)  # noqa: E741
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)  # noqa: E741

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Youden's J = sensitivity + specificity - 1
        youden = sensitivity + specificity - 1

        if youden > best_youden:
            best_youden = youden
            best_threshold = threshold

    # Calculate ROC AUC
    try:
        from sklearn.metrics import roc_auc_score

        roc_auc = roc_auc_score(labels, scores)
    except Exception:
        roc_auc = 0.0

    return best_threshold, roc_auc


def optimize_threshold_adaptive(
    scores: list[float],
    labels: list[int],
    document_type: str = "mixed",
    precision_weight: float = 0.5,
    recall_weight: float = 0.5,
) -> OptimizationResult:
    """
    Adaptively optimize threshold based on document type.

    Args:
        scores: List of similarity scores
        labels: List of ground truth labels
        document_type: Type of documents
        precision_weight: Weight for precision in objective
        recall_weight: Weight for recall in objective

    Returns:
        OptimizationResult object
    """
    # First, optimize using F1
    optimal_threshold, precision, recall, f1 = optimize_threshold_f1(scores, labels)

    # Adjust based on document type
    adjustment = 0.0

    if document_type == "homogeneous":
        # Lower threshold for homogeneous datasets (more similar)
        adjustment = -0.03
    elif document_type == "heterogeneous":
        # Higher threshold for heterogeneous (less similar)
        adjustment = +0.03
    elif document_type == "mixed":
        adjustment = 0.0

    adjusted_threshold = optimal_threshold + adjustment
    adjusted_threshold = max(0.10, min(0.99, adjusted_threshold))

    # Calculate ROC AUC
    try:
        from sklearn.metrics import roc_auc_score

        roc_auc = roc_auc_score(labels, scores)
    except Exception:
        roc_auc = 0.0

    # Calculate confidence based on data quality
    n = len(scores)
    confidence = min(1.0, n / 50) if n > 0 else 0.0

    return OptimizationResult(
        optimal_threshold=adjusted_threshold,
        precision=precision,
        recall=recall,
        f1_score=f1,
        roc_auc=roc_auc,
        document_type=document_type,
        method="adaptive_f1",
        confidence=confidence,
        metadata={
            "original_threshold": optimal_threshold,
            "adjustment": adjustment,
            "samples": n,
            "labels_distribution": {
                "positive": sum(labels),
                "negative": len(labels) - sum(labels),
            },
        },
    )


# ============================================================================
# THRESHOLD OPTIMIZER
# ============================================================================


class ThresholdOptimizer:
    """Main threshold optimizer class."""

    def __init__(self, config: Optional[ThresholdConfig] = None):
        self.config = config or ThresholdConfig()
        self._results: Dict[str, OptimizationResult] = {}
        self._history: List[Dict[str, Any]] = []

    def optimize_from_data(
        self,
        scores: List[float],
        labels: List[int],
        texts: Optional[List[str]] = None,
        method: str = "adaptive",
    ) -> OptimizationResult:
        """
        Optimize threshold from data.

        Args:
            scores: List of similarity scores
            labels: List of ground truth labels
            texts: List of document texts (optional)
            method: Optimization method ('f1', 'roc', 'adaptive')

        Returns:
            OptimizationResult
        """
        if len(scores) != len(labels):
            raise ValueError("Scores and labels must have same length")

        # Detect document type
        document_type = detect_document_type(scores, texts or [])

        # Optimize based on method
        if method == "f1":
            threshold, precision, recall, f1 = optimize_threshold_f1(
                scores,
                labels,
                self.config.min_threshold,
                self.config.max_threshold,
                self.config.step,
            )

            roc_auc = 0.0
            try:
                from sklearn.metrics import roc_auc_score

                roc_auc = roc_auc_score(labels, scores)
            except Exception:
                pass

            result = OptimizationResult(
                optimal_threshold=threshold,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                document_type=document_type,
                method="f1",
                confidence=min(1.0, len(scores) / 50),
            )

        elif method == "roc":
            threshold, roc_auc = optimize_threshold_roc(
                scores,
                labels,
                self.config.min_threshold,
                self.config.max_threshold,
                self.config.step,
            )

            # Recalculate metrics at optimal threshold
            predictions = [1 if s >= threshold else 0 for s in scores]
            tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)  # noqa: E741
            fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)  # noqa: E741
            fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)  # noqa: E741

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            result = OptimizationResult(
                optimal_threshold=threshold,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                document_type=document_type,
                method="roc",
                confidence=min(1.0, len(scores) / 50),
            )

        else:  # adaptive
            result = optimize_threshold_adaptive(
                scores,
                labels,
                document_type,
                self.config.precision_weight,
                self.config.recall_weight,
            )

        # Store result
        self._results[document_type] = result
        self._history.append(
            {
                "timestamp": pd.Timestamp.now(),
                "document_type": document_type,
                "threshold": result.optimal_threshold,
                "f1_score": result.f1_score,
                "samples": len(scores),
            }
        )

        return result

    def get_threshold_for_document_type(self, document_type: str) -> float:
        """
        Get optimal threshold for a document type.

        Args:
            document_type: Type of documents

        Returns:
            Optimal threshold
        """
        if document_type in self._results:
            return self._results[document_type].optimal_threshold
        return self.config.plagiarism

    def get_results(self) -> Dict[str, OptimizationResult]:
        """Get all optimization results."""
        return self._results

    def get_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self._history

    def reset(self) -> None:
        """Reset optimizer state."""
        self._results.clear()
        self._history.clear()


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_optimizer: Optional[ThresholdOptimizer] = None
_optimizer_lock = threading.Lock()


def get_threshold_optimizer() -> ThresholdOptimizer:
    """Get global threshold optimizer instance."""
    global _optimizer
    with _optimizer_lock:
        if _optimizer is None:
            _optimizer = ThresholdOptimizer()
        return _optimizer
