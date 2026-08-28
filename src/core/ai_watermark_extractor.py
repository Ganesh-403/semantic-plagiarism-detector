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
src/core/ai_watermark_extractor.py
----------------------------------
AI-Generated Text Watermark Extraction Engine.

Extracts token probability distributions and n-gram frequencies to detect
invisible statistical watermarks (e.g., the Maryland watermarking scheme)
embedded by advanced AI text generators.
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Simulated "Green List" of tokens that are biased by the watermarking scheme.
# In a real implementation, this would be derived from the secret key and
# the model's vocabulary. For this implementation, we use a fixed subset
# of common words to simulate the detection logic.
GREEN_LIST_TOKENS = {
    "the",
    "and",
    "is",
    "in",
    "to",
    "of",
    "a",
    "that",
    "it",
    "for",
    "on",
    "with",
    "as",
    "this",
    "but",
    "from",
    "or",
    "were",
    "are",
}


def extract_token_distribution(text: str) -> Dict[str, Any]:
    """Extract token frequencies and n-gram distributions from text.

    Args:
        text: The input text string.

    Returns:
        Dictionary containing token counts and green list metrics.
    """
    if not text or not isinstance(text, str):
        return {"total_tokens": 0, "green_list_count": 0, "green_list_ratio": 0.0}

    tokens = re.findall(r"\b\w+\b", text.lower())
    total_tokens = len(tokens)

    if total_tokens == 0:
        return {"total_tokens": 0, "green_list_count": 0, "green_list_ratio": 0.0}

    # Count tokens that fall in the simulated "green list"
    green_list_count = sum(1 for token in tokens if token in GREEN_LIST_TOKENS)
    green_list_ratio = green_list_count / total_tokens

    return {
        "total_tokens": total_tokens,
        "green_list_count": green_list_count,
        "green_list_ratio": round(green_list_ratio, 4),
    }


def compute_ngram_frequencies(text: str, n: int = 2) -> Dict[str, int]:
    """Compute n-gram frequencies for the text.

    Args:
        text: The input text.
        n: Size of the n-gram.

    Returns:
        Dictionary mapping n-gram strings to their frequencies.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) < n:
        return {}

    ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return dict(Counter(ngrams))
