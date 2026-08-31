"""
src/core/cadence_fingerprinter.py
---------------------------------
Cadence Fingerprinter for TTS Detection.

Computes cadence fingerprints and detects synthetic TTS markers,
such as unnaturally uniform pause distributions.
"""

import logging
from typing import List, Dict, Any
from statistics import variance, mean

logger = logging.getLogger(__name__)


def compute_cadence_fingerprint(pauses: List[float]) -> Dict[str, float]:
    """Compute cadence metrics from pause durations.

    Args:
        pauses: List of pause durations in seconds.

    Returns:
        Dictionary containing cadence metrics.
    """
    if not pauses:
        return {"mean_pause": 0.0, "pause_variance": 0.0, "is_tts": False}

    mean_pause = mean(pauses)
    pause_var = variance(pauses) if len(pauses) > 1 else 0.0

    # TTS engines often produce unnaturally uniform pauses (low variance)
    # Human speech has higher variance due to natural breathing and thinking pauses
    is_tts = pause_var < 0.01 and mean_pause > 0.1

    return {
        "mean_pause": round(mean_pause, 4),
        "pause_variance": round(pause_var, 6),
        "is_tts": is_tts,
    }


def analyze_prosody(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze full prosody of a transcript.

    Args:
        transcript: Timestamped transcript.

    Returns:
        Dictionary containing prosody metrics and TTS flags.
    """
    from src.core.audio_transcript_analyzer import (
        extract_pause_durations,
        compute_speech_rate_variance,
    )

    pauses = extract_pause_durations(transcript)
    cadence = compute_cadence_fingerprint(pauses)
    rate_var = compute_speech_rate_variance(transcript)

    return {
        "cadence": cadence,
        "speech_rate_variance": round(rate_var, 4),
        "is_synthetic": cadence["is_tts"],
    }
