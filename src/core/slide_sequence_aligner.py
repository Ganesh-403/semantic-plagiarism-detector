"""
src/core/slide_sequence_aligner.py
----------------------------------
Slide Sequence Alignment Engine.

Computes sequence alignment between slide decks using text similarity
and layout edit distance to detect slide-by-slide plagiarism.
"""

import re
import logging
from typing import List, Dict, Any
from src.core.pptx_slide_extractor import PresentationDeck, Slide

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Extract alphanumeric tokens from text."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_slide_text_similarity(slide_a: Slide, slide_b: Slide) -> float:
    """Compute Jaccard text similarity between two slides."""
    tokens_a = set(_tokenize(slide_a.get_full_text() + " " + slide_a.notes))
    tokens_b = set(_tokenize(slide_b.get_full_text() + " " + slide_b.notes))

    if not tokens_a and not tokens_b:
        return 1.0

    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))

    return intersection / union if union > 0 else 0.0


def compute_layout_edit_distance(slide_a: Slide, slide_b: Slide) -> int:
    """Compute a simplified layout edit distance based on element bounding boxes.

    Compares the sequence of element types (text vs visual) and their
    relative spatial positions.
    """
    # Extract sequence of element widths/heights as a proxy for layout
    seq_a = [(e.width, e.height) for e in slide_a.elements]
    seq_b = [(e.width, e.height) for e in slide_b.elements]

    n, m = len(seq_a), len(seq_b)
    if n == 0 and m == 0:
        return 0
    if n == 0 or m == 0:
        return max(n, m)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Cost is 0 if dimensions are within 10% of each other
            w_diff = abs(seq_a[i - 1][0] - seq_b[j - 1][0]) / max(
                seq_a[i - 1][0], seq_b[j - 1][0], 1
            )
            h_diff = abs(seq_a[i - 1][1] - seq_b[j - 1][1]) / max(
                seq_a[i - 1][1], seq_b[j - 1][1], 1
            )
            cost = 0 if (w_diff < 0.1 and h_diff < 0.1) else 1

            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[n][m]


def compute_deck_similarity(
    deck_a: PresentationDeck, deck_b: PresentationDeck
) -> Dict[str, Any]:
    """Compute slide-by-slide similarity between two presentation decks.

    Args:
        deck_a: First presentation deck.
        deck_b: Second presentation deck.

    Returns:
        Dictionary containing sequence alignment scores and plagiarism flags.
    """
    if deck_a.slide_count == 0 or deck_b.slide_count == 0:
        return {
            "text_similarity": 0.0,
            "layout_similarity": 0.0,
            "overall_score": 0.0,
            "is_cloned_deck": False,
        }

    text_sims = []
    layout_sims = []

    # Compare slides pairwise (assuming 1-to-1 mapping for cloned decks)
    min_slides = min(deck_a.slide_count, deck_b.slide_count)

    for i in range(min_slides):
        slide_a = deck_a.slides[i]
        slide_b = deck_b.slides[i]

        text_sim = compute_slide_text_similarity(slide_a, slide_b)
        text_sims.append(text_sim)

        layout_dist = compute_layout_edit_distance(slide_a, slide_b)
        max_elements = max(len(slide_a.elements), len(slide_b.elements), 1)
        layout_sim = 1.0 - (layout_dist / max_elements)
        layout_sims.append(layout_sim)

    avg_text_sim = sum(text_sims) / len(text_sims) if text_sims else 0.0
    avg_layout_sim = sum(layout_sims) / len(layout_sims) if layout_sims else 0.0

    # Overall score weights text higher, but layout is a strong indicator of cloning
    overall_score = (avg_text_sim * 0.6) + (avg_layout_sim * 0.4)

    # Flag as cloned if both text and layout are highly similar
    is_cloned = overall_score > 0.80 and avg_layout_sim > 0.70

    return {
        "text_similarity": round(avg_text_sim, 4),
        "layout_similarity": round(avg_layout_sim, 4),
        "overall_score": round(overall_score, 4),
        "is_cloned_deck": is_cloned,
    }
