"""Stylometric Author Attribution Engine Service.

Extracts sentence length statistics, Type-Token Ratio (TTR), Hapax Legomena, and
function word vector distributions for authorship verification.
"""

import math
import re
import uuid
from datetime import datetime
from typing import Dict, List, Tuple

from src.models.stylometric_author_model import (
    StylometricAuditReport,
    StylometricAuthorMatch,
    StylometricFingerprint,
)

COMMON_FUNCTION_WORDS = {
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "i",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
}


class StylometricAuthorEngine:
    """Core analytics engine for extracting stylometric features and verifying authorship."""

    @classmethod
    def extract_fingerprint(
        cls, document_id: str, author_alias: str, text_content: str
    ) -> StylometricFingerprint:
        """Extracts high-dimensional stylometric feature vector from text."""
        sentences = [s.strip() for s in re.split(r"[.!?]+", text_content) if s.strip()]
        sentence_lengths = [len(s.split()) for s in sentences] if sentences else [0]

        avg_sent_len = (
            round(sum(sentence_lengths) / len(sentence_lengths), 2)
            if sentence_lengths
            else 0.0
        )
        variance = (
            round(
                sum((l - avg_sent_len) ** 2 for l in sentence_lengths)
                / len(sentence_lengths),
                2,
            )
            if sentence_lengths
            else 0.0
        )

        words = [w.lower() for w in re.findall(r"\b\w+\b", text_content)]
        total_words = len(words)

        if total_words == 0:
            return StylometricFingerprint(
                document_id=document_id,
                author_alias=author_alias,
                average_sentence_length=0.0,
                sentence_length_variance=0.0,
                type_token_ratio=0.0,
                hapax_legomena_ratio=0.0,
                function_word_frequencies={},
                punctuation_density=0.0,
                extracted_at=datetime.utcnow(),
            )

        unique_words = set(words)
        ttr = round(len(unique_words) / total_words, 4)

        # Count Hapax Legomena (words occurring exactly once)
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        hapax_count = sum(1 for w, c in word_counts.items() if c == 1)
        hapax_ratio = round(hapax_count / total_words, 4)

        # Function word frequency vector
        func_freqs = {}
        for fw in COMMON_FUNCTION_WORDS:
            func_freqs[fw] = round(word_counts.get(fw, 0) / total_words, 4)

        punct_count = len(re.findall(r"[.,;:'\"!?\-]", text_content))
        punct_density = round(punct_count / total_words, 4)

        return StylometricFingerprint(
            document_id=document_id,
            author_alias=author_alias,
            average_sentence_length=avg_sent_len,
            sentence_length_variance=variance,
            type_token_ratio=ttr,
            hapax_legomena_ratio=hapax_ratio,
            function_word_frequencies=func_freqs,
            punctuation_density=punct_density,
            extracted_at=datetime.utcnow(),
        )

    @staticmethod
    def calculate_stylometric_distance(
        fp1: StylometricFingerprint, fp2: StylometricFingerprint
    ) -> float:
        """Calculates Euclidean distance between two stylometric fingerprints."""
        sent_diff = (fp1.average_sentence_length - fp2.average_sentence_length) ** 2
        ttr_diff = ((fp1.type_token_ratio - fp2.type_token_ratio) * 10) ** 2
        hapax_diff = ((fp1.hapax_legomena_ratio - fp2.hapax_legomena_ratio) * 10) ** 2

        func_diff = 0.0
        for fw in COMMON_FUNCTION_WORDS:
            f1 = fp1.function_word_frequencies.get(fw, 0.0)
            f2 = fp2.function_word_frequencies.get(fw, 0.0)
            func_diff += ((f1 - f2) * 50) ** 2

        dist = math.sqrt(sent_diff + ttr_diff + hapax_diff + func_diff)
        return round(dist, 4)

    @classmethod
    def compare_authorship(
        cls, query_fp: StylometricFingerprint, candidate_fp: StylometricFingerprint
    ) -> StylometricAuthorMatch:
        """Compares two fingerprints and calculates authorship attribution probability."""
        dist = cls.calculate_stylometric_distance(query_fp, candidate_fp)
        is_same = dist <= 3.50

        # Convert distance to confidence percentage
        confidence = max(0.0, round(100.0 - (dist * 18.0), 2))
        trait = (
            "Function Word Distribution"
            if dist <= 2.0
            else "Sentence Length & Vocabulary TTR"
        )

        return StylometricAuthorMatch(
            match_id=f"STYLE-{uuid.uuid4().hex[:8].upper()}",
            query_document_id=query_fp.document_id,
            candidate_author_alias=candidate_fp.author_alias,
            candidate_document_id=candidate_fp.document_id,
            stylometric_distance=dist,
            attribution_confidence_percentage=confidence,
            is_same_author=is_same,
            dominant_stylometric_trait=trait,
            compared_at=datetime.utcnow(),
        )
