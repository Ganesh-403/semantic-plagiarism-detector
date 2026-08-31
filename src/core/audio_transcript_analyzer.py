"""
src/core/audio_transcript_analyzer.py
-------------------------------------
Audio Transcript Prosody Analyzer.

Extracts pause durations, speech rate variance, and punctuation-to-pause
alignment from timestamped transcripts to detect read-aloud plagiarism
or synthetic TTS generation.
"""

import logging
from typing import List, Dict, Any
from statistics import variance, mean

logger = logging.getLogger(__name__)


def extract_pause_durations(transcript: List[Dict[str, Any]]) -> List[float]:
    """Extract pause durations between words from a timestamped transcript.

    Args:
        transcript: List of dicts with 'word', 'start', 'end' keys.

    Returns:
        List of pause durations in seconds.
    """
    if not transcript or len(transcript) < 2:
        return []

    pauses = []
    for i in range(len(transcript) - 1):
        end_time = transcript[i].get("end", 0.0)
        next_start = transcript[i + 1].get("start", 0.0)
        pause = next_start - end_time
        if pause > 0.05:  # Ignore micro-pauses < 50ms
            pauses.append(pause)

    return pauses


def compute_speech_rate_variance(transcript: List[Dict[str, Any]]) -> float:
    """Compute the variance in speech rate (words per second).

    Args:
        transcript: Timestamped transcript.

    Returns:
        Variance of speech rate across 5-second windows.
    """
    if not transcript:
        return 0.0

    total_time = transcript[-1].get("end", 0.0) - transcript[0].get("start", 0.0)
    if total_time <= 0:
        return 0.0

    # Split into 5-second windows
    window_size = 5.0
    start_time = transcript[0].get("start", 0.0)
    rates = []

    current_window_start = start_time
    word_count = 0

    for word in transcript:
        word_end = word.get("end", 0.0)
        if word_end - current_window_start >= window_size:
            rate = word_count / window_size
            rates.append(rate)
            current_window_start = word_end
            word_count = 0
        word_count += 1

    if len(rates) < 2:
        return 0.0

    return variance(rates)
