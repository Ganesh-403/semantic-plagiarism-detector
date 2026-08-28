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
src/core/translation_invariance_scorer.py
-----------------------------------------
Translation Invariance Scorer for Back-Translation Defense.

Computes semantic drift between original and back-translated text to detect
evasion techniques. Uses a lightweight lexical overlap and structural variance
proxy to approximate semantic invariance without requiring heavy embedding models.
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _tokenize_words(text: str) -> List[str]:
    """Extract alphanumeric words, lowercased."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_lexical_drift(original: str, translated: str) -> float:
    """Compute lexical drift using Jaccard distance.

    Measures the proportion of vocabulary that changed between the original
    and back-translated text. Higher values indicate more semantic drift.

    Args:
        original: The original text.
        translated: The back-translated text.

    Returns:
        Drift score between 0.0 (identical) and 1.0 (completely different).
    """
    words_orig = set(_tokenize_words(original))
    words_trans = set(_tokenize_words(translated))

    if not words_orig and not words_trans:
        return 0.0

    intersection = len(words_orig.intersection(words_trans))
    union = len(words_orig.union(words_trans))

    jaccard_sim = intersection / union if union > 0 else 0.0
    return 1.0 - jaccard_sim


def compute_structural_variance(original: str, translated: str) -> float:
    """Compute structural variance using sentence length differences.

    Back-translation often alters sentence boundaries and lengths. This metric
    measures the normalized absolute difference in sentence counts and average
    sentence lengths.
    """
    sents_orig = re.split(r"(?<=[.!?])\s+", original.strip())
    sents_trans = re.split(r"(?<=[.!?])\s+", translated.strip())

    if not sents_orig or not sents_trans:
        return 0.0

    len_orig = len(sents_orig)
    len_trans = len(sents_trans)

    # Sentence count variance
    count_diff = abs(len_orig - len_trans) / max(len_orig, len_trans, 1)

    # Average sentence length variance
    avg_len_orig = sum(len(s.split()) for s in sents_orig) / len_orig
    avg_len_trans = sum(len(s.split()) for s in sents_trans) / len_trans

    avg_diff = abs(avg_len_orig - avg_len_trans) / max(avg_len_orig, avg_len_trans, 1)

    return (count_diff + avg_diff) / 2.0


def score_translation_invariance(original: str, translated: str) -> Dict[str, float]:
    """Compute a comprehensive translation invariance score.

    Combines lexical drift and structural variance to determine if a text
    has been heavily obfuscated via back-translation.

    Args:
        original: The original text.
        translated: The back-translated text.

    Returns:
        Dictionary containing drift metrics and an overall invariance score.
    """
    lexical_drift = compute_lexical_drift(original, translated)
    structural_var = compute_structural_variance(original, translated)

    # Overall invariance score (1.0 = highly invariant, 0.0 = heavily drifted)
    # We weight lexical drift higher as it's the primary indicator of synonym swapping
    invariance_score = 1.0 - ((lexical_drift * 0.7) + (structural_var * 0.3))
    invariance_score = max(0.0, min(1.0, invariance_score))

    return {
        "lexical_drift": round(lexical_drift, 4),
        "structural_variance": round(structural_var, 4),
        "invariance_score": round(invariance_score, 4),
        "is_obfuscated": invariance_score < 0.6,  # Threshold for flagging
    }
