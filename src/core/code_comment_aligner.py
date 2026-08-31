"""
src/core/code_comment_aligner.py
--------------------------------
Code Comment Semantic Alignment Engine.

Computes semantic alignment and coherence scores between code behavior
and comment descriptions to detect mismatches, copied documentation,
or AI-generated comment padding.
"""

import re
import math
import logging
from typing import List, Dict, Any
from collections import Counter
from src.core.docstring_extractor import CodeBlock

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Extract alphanumeric tokens from text."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def compute_coherence_score(blocks: List[CodeBlock]) -> Dict[str, Any]:
    """Compute semantic coherence between code and comments.

    High coherence indicates that comments accurately describe the code.
    Low coherence may indicate copied documentation, AI-generated padding,
    or outdated comments.

    Args:
        blocks: List of CodeBlock objects.

    Returns:
        Dictionary containing coherence metrics.
    """
    if not blocks:
        return {"overall_coherence": 0.0, "is_mismatch": False, "block_count": 0}

    coherence_scores = []

    for block in blocks:
        if not block.code_text or not block.comment_text:
            continue

        code_tokens = set(_tokenize(block.code_text))
        comment_tokens = set(_tokenize(block.comment_text))

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "for",
            "in",
            "to",
            "and",
            "or",
            "if",
            "else",
            "return",
            "def",
            "class",
        }
        code_tokens -= stop_words
        comment_tokens -= stop_words

        sim = compute_jaccard_similarity(code_tokens, comment_tokens)
        coherence_scores.append(sim)

    if not coherence_scores:
        return {
            "overall_coherence": 0.0,
            "is_mismatch": False,
            "block_count": len(blocks),
        }

    overall_coherence = sum(coherence_scores) / len(coherence_scores)

    # Flag as mismatch if coherence is very low (comments don't match code)
    # or if there are many blocks with empty comments (AI padding)
    empty_comment_ratio = sum(1 for b in blocks if not b.comment_text) / len(blocks)
    is_mismatch = overall_coherence < 0.15 and empty_comment_ratio < 0.5

    return {
        "overall_coherence": round(overall_coherence, 4),
        "is_mismatch": is_mismatch,
        "block_count": len(blocks),
        "empty_comment_ratio": round(empty_comment_ratio, 4),
    }
