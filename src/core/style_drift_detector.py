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
src/core/style_drift_detector.py
--------------------------------
Intra-Document Writing Style Drift Detection Engine.

Computes sliding-window stylometric features across document chunks to
identify localized anomalies in vocabulary richness, sentence structure,
and syntactic complexity. This is critical for detecting contract cheating
where a student writes part of a paper themselves but outsources other sections.
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _tokenize_words(text: str) -> List[str]:
    """Extract alphanumeric words from text, lowercased."""
    return re.findall(r"\b\w+\b", text.lower())


def _tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def compute_type_token_ratio(words: List[str]) -> float:
    """Compute Type-Token Ratio (TTR) for vocabulary richness."""
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_avg_sentence_length(sentences: List[str]) -> float:
    """Compute average words per sentence."""
    if not sentences:
        return 0.0
    lengths = [len(_tokenize_words(s)) for s in sentences]
    return sum(lengths) / len(lengths)


def compute_sentence_length_variance(sentences: List[str]) -> float:
    """Compute variance in sentence lengths (burstiness)."""
    if len(sentences) <= 1:
        return 0.0
    lengths = [len(_tokenize_words(s)) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    return sum((x - mean_len) ** 2 for x in lengths) / (len(lengths) - 1)


def compute_yules_k(words: List[str]) -> float:
    """Compute Yule's characteristic K for vocabulary richness."""
    if not words:
        return 0.0
    n = len(words)
    freq_counter = Counter(words)
    freq_of_freqs = Counter(freq_counter.values())
    sum_fi_i2 = sum(f * (i**2) for i, f in freq_of_freqs.items())
    m2 = sum_fi_i2 / (n**2)
    k = 10000.0 * (m2 - (1.0 / n))
    return max(0.0, k)


def extract_sliding_window_features(
    text: str, window_size: int = 500, step_size: int = 250
) -> List[Dict[str, float]]:
    """Extract stylometric features using a sliding window approach.

    Args:
        text: The full document text.
        window_size: Number of words per window.
        step_size: Number of words to step forward for the next window.

    Returns:
        List of dictionaries containing features for each window.
    """
    words = _tokenize_words(text)
    if len(words) < window_size:
        # If text is shorter than window, treat as single window
        windows = [words]
    else:
        windows = [
            words[i : i + window_size]
            for i in range(0, len(words) - window_size + 1, step_size)
        ]

    features_list = []
    current_pos = 0

    for window_words in windows:
        # Reconstruct text for sentence splitting
        window_text = " ".join(window_words)
        sentences = _tokenize_sentences(window_text)

        features = {
            "start_word": current_pos,
            "end_word": current_pos + len(window_words),
            "ttr": compute_type_token_ratio(window_words),
            "avg_sent_len": compute_avg_sentence_length(sentences),
            "sent_len_var": compute_sentence_length_variance(sentences),
            "yules_k": compute_yules_k(window_words),
        }
        features_list.append(features)
        current_pos += step_size

    logger.info("Extracted sliding window features for %d windows.", len(features_list))
    return features_list
