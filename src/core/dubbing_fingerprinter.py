"""
src/core/dubbing_fingerprinter.py
-------------------------------
Dubbing and Splicing Fingerprinter.

Computes pitch contour similarity and background noise alignment to detect
dubbed, spliced, or stolen multimedia tracks.
"""

import logging
from typing import List, Dict, Any
from src.core.av_sync_extractor import AVFeatures

logger = logging.getLogger(__name__)


def compute_pitch_similarity(contour_a: List[float], contour_b: List[float]) -> float:
    """Compute cross-correlation proxy between two pitch contours."""
    if not contour_a or not contour_b:
        return 0.0

    min_len = min(len(contour_a), len(contour_b))
    if min_len == 0:
        return 0.0

    # Compute mean absolute error as a proxy for similarity
    mae = (
        sum(abs(a - b) for a, b in zip(contour_a[:min_len], contour_b[:min_len]))
        / min_len
    )

    # Convert MAE to similarity score (lower error = higher similarity)
    return round(max(0.0, 1.0 - mae), 4)


def analyze_dubbing(features_a: AVFeatures, features_b: AVFeatures) -> Dict[str, Any]:
    """Analyze two multimedia files for signs of dubbing or splicing."""
    if not features_a.pitch_contour and not features_b.pitch_contour:
        return {
            "pitch_similarity": 0.0,
            "noise_match": False,
            "overall_score": 0.0,
            "is_dubbed": False,
        }

    pitch_sim = compute_pitch_similarity(
        features_a.pitch_contour, features_b.pitch_contour
    )
    noise_match = (
        features_a.background_noise_hash == features_b.background_noise_hash
        and features_a.background_noise_hash != ""
    )

    # High AV sync variance + low pitch similarity to original = likely dubbed
    # If noise matches but pitch is different, it's a strong indicator of voice-over dubbing
    dubbing_score = 0.0
    if noise_match and pitch_sim < 0.5:
        dubbing_score = 0.9  # High confidence in dubbing
    elif features_a.av_sync_variance > 0.5 and features_b.av_sync_variance > 0.5:
        dubbing_score = 0.7  # High sync variance in both suggests splicing

    overall_score = (pitch_sim * 0.4) + (dubbing_score * 0.6)

    # Flag as dubbed if pitch is significantly different but background noise matches
    is_dubbed = noise_match and pitch_sim < 0.6

    return {
        "pitch_similarity": pitch_sim,
        "noise_match": noise_match,
        "av_sync_variance_a": features_a.av_sync_variance,
        "av_sync_variance_b": features_b.av_sync_variance,
        "dubbing_probability": round(dubbing_score, 4),
        "overall_score": round(overall_score, 4),
        "is_dubbed": is_dubbed,
    }
