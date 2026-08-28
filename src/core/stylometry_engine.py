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
src/core/stylometry_engine.py
-----------------------------
Stylometric Authorship Attribution and Ghostwriting Detection Engine.

Analyzes the subconscious writing habits of an author to create a unique
"writer fingerprint." This engine extracts features like Type-Token Ratio (TTR),
average sentence length, sentence length variance, punctuation frequency,
and vocabulary richness. By comparing a submission's stylometric profile
against a student's historical baseline, the system can detect ghostwriting
or AI-generated text that has been heavily paraphrased.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StylometricProfile:
    """Represents the stylometric fingerprint of a text or author."""

    type_token_ratio: float  # Vocabulary richness (unique words / total words)
    avg_sentence_length: float  # Average words per sentence
    sentence_length_variance: float  # Variance in sentence lengths (burstiness)
    avg_word_length: float  # Average characters per word
    punctuation_frequency: float  # Punctuation marks per 100 words
    yules_k: float  # Yule's characteristic K (vocabulary richness)

    def to_dict(self) -> dict[str, float]:
        """Convert the profile to a dictionary."""
        return asdict(self)

    def compute_deviation_score(self, baseline: "StylometricProfile") -> float:
        """Compute the normalized Euclidean distance from a baseline profile.

        A higher score indicates a greater deviation from the author's
        historical writing style, suggesting potential ghostwriting.
        """
        weights = {
            "type_token_ratio": 1.5,
            "avg_sentence_length": 1.0,
            "sentence_length_variance": 2.0,  # High weight for burstiness (AI vs Human)
            "avg_word_length": 0.5,
            "punctuation_frequency": 1.0,
            "yules_k": 1.5,
        }

        squared_diffs = []
        for key, weight in weights.items():
            current_val = getattr(self, key)
            baseline_val = getattr(baseline, key)
            squared_diffs.append(weight * ((current_val - baseline_val) ** 2))

        # Return the weighted Euclidean distance
        return math.sqrt(sum(squared_diffs))


def _tokenize_words(text: str) -> list[str]:
    """Extract alphanumeric words from text, lowercased."""
    return re.findall(r"\b\w+\b", text.lower())


def _tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries."""
    # Simple regex for sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def compute_type_token_ratio(words: list[str]) -> float:
    """Compute Type-Token Ratio (TTR).

    TTR = (Number of unique words) / (Total number of words).
    Higher TTR indicates a richer, more diverse vocabulary.
    """
    if not words:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)


def compute_yules_k(words: list[str]) -> float:
    """Compute Yule's characteristic K for vocabulary richness.

    Yule's K is a measure of vocabulary richness that is independent of
    text length. Lower values indicate a richer vocabulary.
    Formula: K = 10^4 * ( (sum(f_i * i^2) / N^2) - (1/N) )
    where f_i is the number of words occurring exactly i times.
    """
    if not words:
        return 0.0

    n = len(words)
    freq_counter = Counter(words)

    # Calculate frequency of frequencies
    freq_of_freqs = Counter(freq_counter.values())

    sum_fi_i2 = sum(f * (i**2) for i, f in freq_of_freqs.items())

    m2 = sum_fi_i2 / (n**2)
    k = 10000.0 * (m2 - (1.0 / n))

    return max(0.0, k)  # Ensure non-negative


def compute_sentence_stats(sentences: list[str]) -> tuple[float, float]:
    """Compute average sentence length and variance."""
    if not sentences:
        return 0.0, 0.0

    lengths = [len(_tokenize_words(s)) for s in sentences]
    n = len(lengths)

    if n == 0:
        return 0.0, 0.0

    mean_len = sum(lengths) / n

    if n == 1:
        return mean_len, 0.0

    variance = sum((x - mean_len) ** 2 for x in lengths) / (n - 1)
    return mean_len, variance


def compute_punctuation_frequency(text: str, words: list[str]) -> float:
    """Compute the number of punctuation marks per 100 words."""
    if not words:
        return 0.0

    # Count standard punctuation marks
    punct_count = len(re.findall(r"[.!?;:,\'\"()\[\]{}\-]", text))
    return (punct_count / len(words)) * 100.0


def compute_avg_word_length(words: list[str]) -> float:
    """Compute the average character length of words."""
    if not words:
        return 0.0
    total_chars = sum(len(w) for w in words)
    return total_chars / len(words)


def extract_stylometric_profile(text: str) -> StylometricProfile:
    """Extract a complete stylometric profile from a text string.

    Args:
        text: The raw text to analyze.

    Returns:
        A populated StylometricProfile object.
    """
    if not text or not text.strip():
        logger.warning("Cannot extract stylometric profile from empty text.")
        return StylometricProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    words = _tokenize_words(text)
    sentences = _tokenize_sentences(text)

    ttr = compute_type_token_ratio(words)
    yules_k = compute_yules_k(words)
    avg_sent_len, sent_var = compute_sentence_stats(sentences)
    punct_freq = compute_punctuation_frequency(text, words)
    avg_word_len = compute_avg_word_length(words)

    profile = StylometricProfile(
        type_token_ratio=round(ttr, 4),
        avg_sentence_length=round(avg_sent_len, 2),
        sentence_length_variance=round(sent_var, 2),
        avg_word_length=round(avg_word_len, 2),
        punctuation_frequency=round(punct_freq, 2),
        yules_k=round(yules_k, 2),
    )

    logger.info(
        "Extracted stylometric profile: TTR=%.3f, Yule's K=%.2f, SentVar=%.2f",
        profile.type_token_ratio,
        profile.yules_k,
        profile.sentence_length_variance,
    )
    return profile
