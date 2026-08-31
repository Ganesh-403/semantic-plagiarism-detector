"""
src/core/readability_analyzer.py
--------------------------------
Readability and Cognitive Load Analyzer.

Computes sliding-window readability metrics (Flesch-Kincaid, Coleman-Liau)
and cognitive load proxies to detect synthetic text generation.
"""

import re
import math
import logging
from typing import List, Dict, Any, Tuple
from statistics import mean

logger = logging.getLogger(__name__)


def count_syllables(word: str) -> int:
    """Estimate the number of syllables in a word using a simple heuristic."""
    word = word.lower()
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count = 1
    return count


def compute_flesch_kincaid(text: str) -> float:
    """Compute the Flesch-Kincaid Grade Level for a text block."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return 0.0

    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0

    total_syllables = sum(count_syllables(w) for w in words)

    num_sentences = len(sentences)
    num_words = len(words)

    # Flesch-Kincaid Grade Level formula
    fk_grade = (
        0.39 * (num_words / num_sentences)
        + 11.8 * (total_syllables / num_words)
        - 15.59
    )
    return max(0.0, fk_grade)


def compute_coleman_liau(text: str) -> float:
    """Compute the Coleman-Liau Index for a text block."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return 0.0

    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0

    num_sentences = len(sentences)
    num_words = len(words)
    num_chars = sum(len(w) for w in words)

    L = (num_chars / num_words) * 100
    S = (num_sentences / num_words) * 100

    cli = 0.0588 * L - 0.296 * S - 15.8
    return max(0.0, cli)


def extract_readability_timeseries(
    text: str, window_size: int = 100
) -> List[Dict[str, float]]:
    """Extract readability metrics over a sliding window of words.

    Args:
        text: The full document text.
        window_size: Number of words per window.

    Returns:
        List of dictionaries containing readability metrics for each window.
    """
    words = re.findall(r"\b\w+\b", text)
    if len(words) < window_size:
        # If text is shorter than window, compute for the whole text
        fk = compute_flesch_kincaid(text)
        cli = compute_coleman_liau(text)
        return [{"fk_grade": fk, "cli": cli, "word_count": len(words)}]

    timeseries = []
    for i in range(0, len(words) - window_size + 1, window_size // 2):
        window_words = words[i : i + window_size]
        window_text = " ".join(window_words) + "."  # Add period for sentence splitting

        fk = compute_flesch_kincaid(window_text)
        cli = compute_coleman_liau(window_text)

        timeseries.append(
            {
                "fk_grade": round(fk, 2),
                "cli": round(cli, 2),
                "word_count": len(window_words),
                "start_idx": i,
            }
        )

    return timeseries
