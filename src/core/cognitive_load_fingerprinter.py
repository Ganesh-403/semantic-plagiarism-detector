"""
src/core/cognitive_load_fingerprinter.py
----------------------------------------
Cognitive Load Fingerprinter for AI Text Detection.

Applies statistical variance analysis to readability time-series to detect
unnatural uniformity characteristic of AI-generated text.
"""

import logging
from typing import List, Dict, Any
from statistics import variance, mean

logger = logging.getLogger(__name__)


def compute_cognitive_load_variance(
    timeseries: List[Dict[str, float]],
) -> Dict[str, Any]:
    """Compute variance in readability metrics across the document.

    AI-generated text typically exhibits low variance in readability scores
    because it generates text with uniform complexity. Human writing features
    natural spikes in cognitive load (e.g., introducing complex concepts,
    followed by simple explanations).

    Args:
        timeseries: List of readability metrics from sliding windows.

    Returns:
        Dictionary containing variance metrics and AI probability.
    """
    if len(timeseries) < 3:
        return {
            "fk_variance": 0.0,
            "cli_variance": 0.0,
            "is_synthetic": False,
            "ai_probability": 0.0,
        }

    fk_scores = [t["fk_grade"] for t in timeseries]
    cli_scores = [t["cli"] for t in timeseries]

    fk_var = variance(fk_scores) if len(fk_scores) > 1 else 0.0
    cli_var = variance(cli_scores) if len(cli_scores) > 1 else 0.0

    # Heuristic thresholds for AI detection
    # Low variance indicates uniform complexity (likely AI)
    is_synthetic = fk_var < 2.0 and cli_var < 5.0

    # Compute AI probability based on how low the variance is
    # Normalize variance to a 0-1 probability score
    fk_prob = max(0.0, min(1.0, 1.0 - (fk_var / 10.0)))
    cli_prob = max(0.0, min(1.0, 1.0 - (cli_var / 20.0)))

    ai_probability = (fk_prob * 0.6) + (cli_prob * 0.4)

    return {
        "fk_variance": round(fk_var, 4),
        "cli_variance": round(cli_var, 4),
        "is_synthetic": is_synthetic,
        "ai_probability": round(ai_probability, 4),
    }


def analyze_cognitive_load(timeseries: List[Dict[str, float]]) -> Dict[str, Any]:
    """Analyze cognitive load fingerprints for a document.

    Args:
        timeseries: Readability time-series data.

    Returns:
        Dictionary containing cognitive load metrics and AI flags.
    """
    variance_metrics = compute_cognitive_load_variance(timeseries)

    return {
        "variance_metrics": variance_metrics,
        "is_ai_generated": variance_metrics["ai_probability"] > 0.65,
    }
