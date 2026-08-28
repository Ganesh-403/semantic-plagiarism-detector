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
src/core/semantic_citation_aligner.py
-------------------------------------
Semantic Citation Alignment Engine.

Computes semantic similarity between the citation context and the source
abstract to detect misleading or fabricated citations (citation bluffing).
Uses a lightweight TF-IDF cosine similarity approach to avoid heavy dependencies.
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace and punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_tf_idf_cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using TF-IDF weighting.

    This is a lightweight implementation that doesn't require scikit-learn.
    It computes Term Frequency (TF) and applies an Inverse Document Frequency
    (IDF) proxy based on the combined vocabulary.

    Args:
        text_a: The citation context text.
        text_b: The reference abstract text.

    Returns:
        Cosine similarity score between 0.0 and 1.0.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    # Compute Term Frequencies
    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)

    # Combined vocabulary for IDF proxy
    vocab = set(tf_a.keys()).union(set(tf_b.keys()))

    # Compute IDF (simple proxy: log(N / df))
    # Since we only have 2 documents, df is either 1 or 2.
    idf = {}
    for term in vocab:
        df = (1 if term in tf_a else 0) + (1 if term in tf_b else 0)
        idf[term] = math.log(2.0 / df) + 1.0  # Smoothing to avoid log(0)

    # Compute TF-IDF vectors
    vec_a = [tf_a.get(term, 0) * idf[term] for term in vocab]
    vec_b = [tf_b.get(term, 0) * idf[term] for term in vocab]

    # Compute dot product
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # Compute magnitudes
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    cosine_sim = dot_product / (mag_a * mag_b)
    return max(0.0, min(1.0, cosine_sim))


def analyze_citation_alignment(
    mapped_contexts: List[Dict[str, Any]], bluffing_threshold: float = 0.15
) -> List[Dict[str, Any]]:
    """Analyze semantic alignment for a list of mapped citation contexts.

    Args:
        mapped_contexts: List of contexts enriched with reference abstracts.
        bluffing_threshold: Similarity threshold below which a citation is
                            flagged as potential bluffing.

    Returns:
        List of contexts enriched with alignment scores and bluffing flags.
    """
    results = []
    for ctx in mapped_contexts:
        context_text = ctx.get("context_text", "")
        abstract = ctx.get("reference_abstract", "")

        similarity = compute_tf_idf_cosine_similarity(context_text, abstract)
        is_bluffing = similarity < bluffing_threshold and len(abstract) > 0

        results.append(
            {**ctx, "alignment_score": round(similarity, 4), "is_bluffing": is_bluffing}
        )

    logger.info("Analyzed alignment for %d citations.", len(results))
    return results
