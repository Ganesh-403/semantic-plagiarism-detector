"""
src/core/trait_analyzer.py
--------------------------
Analytic Trait Extraction for Automated Essay Scoring.

Extracts intrinsic quality traits from essays independent of plagiarism scores.
Analyzes coherence (sentence-to-sentence semantic flow), lexical complexity
(advanced vocabulary density), and argumentation structure.
"""

import re
import math
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

# List of advanced/academic words for lexical complexity scoring
# (Simplified subset for demonstration)
ACADEMIC_VOCABULARY = {
    "analyze",
    "evaluate",
    "synthesize",
    "hypothesis",
    "methodology",
    "paradigm",
    "empirical",
    "theoretical",
    "framework",
    "implication",
    "subsequent",
    "furthermore",
    "consequently",
    "nevertheless",
    "whereas",
}


def _tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _tokenize_words(text: str) -> List[str]:
    """Extract alphanumeric words, lowercased."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_coherence_score(text: str) -> float:
    """Compute sentence-to-sentence coherence using lexical overlap.

    Measures the average Jaccard similarity between adjacent sentences.
    Higher overlap indicates better local coherence and semantic flow.

    Args:
        text: The essay text.

    Returns:
        Coherence score between 0.0 and 1.0.
    """
    sentences = _tokenize_sentences(text)
    if len(sentences) < 2:
        return 0.0

    similarities = []
    for i in range(len(sentences) - 1):
        words_a = set(_tokenize_words(sentences[i]))
        words_b = set(_tokenize_words(sentences[i + 1]))

        if not words_a or not words_b:
            continue

        intersection = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))
        jaccard = intersection / union if union > 0 else 0.0
        similarities.append(jaccard)

    if not similarities:
        return 0.0

    # Average coherence across all adjacent pairs
    avg_coherence = sum(similarities) / len(similarities)

    # Normalize to 0-1 scale (typical Jaccard for sentences is low, so we scale up)
    # Cap at 1.0
    return min(1.0, avg_coherence * 5.0)


def compute_lexical_complexity(text: str) -> Dict[str, float]:
    """Compute lexical complexity metrics.

    Analyzes Type-Token Ratio (TTR), average word length, and the density
    of advanced/academic vocabulary.

    Args:
        text: The essay text.

    Returns:
        Dictionary containing complexity metrics.
    """
    words = _tokenize_words(text)
    if not words:
        return {"ttr": 0.0, "avg_word_length": 0.0, "academic_density": 0.0}

    unique_words = set(words)
    ttr = len(unique_words) / len(words)

    avg_word_length = sum(len(w) for w in words) / len(words)

    # Compute academic vocabulary density
    academic_count = sum(1 for w in words if w in ACADEMIC_VOCABULARY)
    academic_density = academic_count / len(words)

    return {
        "ttr": round(ttr, 4),
        "avg_word_length": round(avg_word_length, 2),
        "academic_density": round(academic_density, 4),
    }


def compute_argumentation_structure(text: str) -> Dict[str, int]:
    """Analyze argumentation structure by counting discourse markers.

    Counts the frequency of transitional phrases and argumentative markers
    (e.g., "however", "therefore", "for example") to assess structural depth.

    Args:
        text: The essay text.

    Returns:
        Dictionary containing counts of structural markers.
    """
    text_lower = text.lower()

    markers = {
        "contrast": ["however", "but", "although", "nevertheless", "whereas"],
        "addition": ["furthermore", "moreover", "additionally", "also"],
        "consequence": ["therefore", "consequently", "thus", "hence", "as a result"],
        "example": ["for example", "for instance", "such as", "specifically"],
    }

    counts = {}
    for category, phrases in markers.items():
        count = sum(text_lower.count(phrase) for phrase in phrases)
        counts[category] = count

    return counts


def extract_analytic_traits(text: str) -> Dict[str, Any]:
    """Extract all analytic traits from an essay.

    Args:
        text: The essay text.

    Returns:
        Comprehensive dictionary of analytic traits.
    """
    if not text or not text.strip():
        return {
            "coherence": 0.0,
            "lexical_complexity": {
                "ttr": 0.0,
                "avg_word_length": 0.0,
                "academic_density": 0.0,
            },
            "argumentation": {
                "contrast": 0,
                "addition": 0,
                "consequence": 0,
                "example": 0,
            },
            "word_count": 0,
            "sentence_count": 0,
        }

    words = _tokenize_words(text)
    sentences = _tokenize_sentences(text)

    return {
        "coherence": compute_coherence_score(text),
        "lexical_complexity": compute_lexical_complexity(text),
        "argumentation": compute_argumentation_structure(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
    }
