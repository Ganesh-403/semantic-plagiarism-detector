"""
src/core/patchwriting_detector.py
---------------------------------
Mosaic Plagiarism (Patchwriting) Detection Engine.

Computes syntactic similarity between documents by comparing their normalized
Part-of-Speech (POS) tag sequences. Uses n-gram overlap and sequence alignment
to detect structural cloning even when lexical content (words) is changed.
"""

import logging
from typing import List, Dict, Any, Tuple
from collections import Counter

from src.core.pos_normalizer import extract_pos_sequence, compute_pos_ngrams

logger = logging.getLogger(__name__)


def compute_syntactic_jaccard(seq_a: list[str], seq_b: list[str]) -> float:
    """Compute the Jaccard similarity between two POS tag sequences.
    
    Treats the sequences as sets of unique tags to measure overall syntactic
    vocabulary overlap.
    """
    if not seq_a and not seq_b:
        return 1.0
    set_a = set(seq_a)
    set_b = set(seq_b)
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def compute_ngram_overlap(
    seq_a: list[str], 
    seq_b: list[str], 
    n: int = 3
) -> float:
    """Compute the Dice coefficient for POS n-gram overlap.
    
    This is the primary metric for detecting patchwriting. It measures
    how many syntactic structures (trigrams) are shared between the two texts.
    
    Args:
        seq_a: POS sequence from document A.
        seq_b: POS sequence from document B.
        n: N-gram size (default 3).
        
    Returns:
        Dice coefficient between 0.0 and 1.0.
    """
    ngrams_a = compute_pos_ngrams(seq_a, n)
    ngrams_b = compute_pos_ngrams(seq_b, n)
    
    if not ngrams_a or not ngrams_b:
        return 0.0
        
    counter_a = Counter(ngrams_a)
    counter_b = Counter(ngrams_b)
    
    # Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
    intersection = sum((counter_a & counter_b).values())
    total = sum(counter_a.values()) + sum(counter_b.values())
    
    return (2.0 * intersection) / total if total > 0 else 0.0


def detect_patchwriting(
    text_a: str, 
    text_b: str, 
    n: int = 3,
    threshold: float = 0.60
) -> dict[str, Any]:
    """Analyze two texts for mosaic plagiarism (patchwriting).
    
    Args:
        text_a: The first text (e.g., student submission).
        text_b: The second text (e.g., source material).
        n: N-gram size for syntactic comparison.
        threshold: Similarity threshold to flag as patchwriting.
        
    Returns:
        Dictionary containing syntactic similarity scores and a boolean flag.
    """
    pos_a = extract_pos_sequence(text_a)
    pos_b = extract_pos_sequence(text_b)
    
    if not pos_a or not pos_b:
        return {
            "syntactic_jaccard": 0.0,
            "ngram_overlap": 0.0,
            "is_patchwriting": False,
            "pos_sequence_a": [],
            "pos_sequence_b": []
        }
        
    jaccard = compute_syntactic_jaccard(pos_a, pos_b)
    ngram_overlap = compute_ngram_overlap(pos_a, pos_b, n)
    
    # We consider it patchwriting if the structural n-gram overlap is high,
    # even if the lexical similarity (checked elsewhere) is low.
    is_patchwriting = ngram_overlap >= threshold
    
    logger.info(
        "Patchwriting analysis: Jaccard=%.3f, N-gram Overlap=%.3f, Flagged=%s",
        jaccard, ngram_overlap, is_patchwriting
    )
    
    return {
        "syntactic_jaccard": round(jaccard, 4),
        "ngram_overlap": round(ngram_overlap, 4),
        "is_patchwriting": is_patchwriting,
        "pos_sequence_a": pos_a,
        "pos_sequence_b": pos_b
    }


# semantic-plagiarism-detector/src/core/patchwriting_detector.py

from typing import Dict, Any, List
from src.core.pos_normalizer import POSNormalizer
import difflib

class PatchwritingDetector:
    """
    Computes syntactic similarity using n-gram overlap on POS sequences and 
    structural edit distance to detect mosaic plagiarism.
    """

    @staticmethod
    def _get_ngrams(sequence: list[str], n: int = 3) -> set:
        """Generates n-grams from a sequence of POS tags."""
        if len(sequence) < n:
            return {tuple(sequence)}
        return {tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)}

    @classmethod
    def compute_syntactic_similarity(cls, source_text: str, student_text: str, n: int = 3) -> dict[str, Any]:
        """
        Computes structural similarity between source and student text using POS n-grams and sequence matching.
        """
        source_pos = POSNormalizer.extract_pos_sequence(source_text)
        student_pos = POSNormalizer.extract_pos_sequence(student_text)

        if not source_pos or not student_pos:
            return {"similarity_score": 0.0, "matched_patterns": []}

        # N-gram overlap calculation
        source_ngrams = cls._get_ngrams(source_pos, n)
        student_ngrams = cls._get_ngrams(student_pos, n)

        if not source_ngrams or not student_ngrams:
            ngram_similarity = 0.0
        else:
            intersection = source_ngrams.intersection(student_ngrams)
            union = source_ngrams.union(student_ngrams)
            ngram_similarity = len(intersection) / len(union) if union else 0.0

        # Sequence edit-distance alignment score
        matcher = difflib.SequenceMatcher(None, source_pos, student_pos)
        sequence_score = matcher.ratio()

        # Combined composite structural similarity score
        composite_score = round((0.6 * sequence_score) + (0.4 * ngram_similarity), 3)

        return {
            "similarity_score": composite_score,
            "ngram_similarity": round(ngram_similarity, 3),
            "sequence_alignment_score": round(sequence_score, 3),
            "source_pos_sample": "-".join(source_pos[:10]),
            "student_pos_sample": "-".join(student_pos[:10])
        }
