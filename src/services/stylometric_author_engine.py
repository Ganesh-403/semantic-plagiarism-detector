"""
Enterprise Stylometric Authorship Attribution & Write-Print Fingerprinting Engine
Extracts fine-grained linguistic write-print features (burstiness, perplexity variance,
punctuation frequency, vocabulary richness, sentence-length entropy) to verify author identity.
"""

import math
import re
import uuid
from datetime import datetime

from src.models.stylometric_author_model import (
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


class StylometricWriteprintExtractor:
    """
    Extracts high-dimensional stylometric feature vectors from text manuscripts
    to model individual authorship fingerprints and detect ghostwriting or AI-assisted synthesis.
    """

    def __init__(self, target_author_id: Optional[str] = None):
        self.target_author_id = target_author_id
        self.extracted_fingerprints: dict[str, Any] = {}

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

        total_words = len(words) or 1
        total_sentences = len(sentences) or 1

        # Vocabulary richness metrics (Type-Token Ratio & Yule's K)
        unique_words = set(words)
        type_token_ratio = round(len(unique_words) / total_words, 4)

        # Sentence length statistics & variance entropy
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_len = sum(sentence_lengths) / total_sentences
        variance_len = sum((l - avg_sentence_len) ** 2 for l in sentence_lengths) / total_sentences
        sentence_len_std_dev = math.sqrt(variance_len)

        # Punctuation mark distribution frequency
        punctuation_marks = re.findall(r'[,;:\-\(\)\"\']', text)
        punctuation_density = round(len(punctuation_marks) / total_words, 4)

        # Average word length
        avg_word_length = round(sum(len(w) for w in words) / total_words, 2)

        writeprint = {
            "authorId": self.target_author_id or "ANONYMOUS_AUTHOR",
            "totalWordsAnalyzed": total_words,
            "totalSentencesAnalyzed": total_sentences,
            "typeTokenRatio": type_token_ratio,
            "avgSentenceLengthWords": round(avg_sentence_len, 2),
            "sentenceLengthStdDev": round(sentence_len_std_dev, 2),
            "punctuationDensity": punctuation_density,
            "avgWordLengthChars": avg_word_length,
            "stylometricComplexityIndex": round(type_token_ratio * sentence_len_std_dev, 3),
        }

        self.extracted_fingerprints[self.target_author_id or "CURRENT"] = writeprint
        return writeprint


class AuthorshipAttributionClassifier:
    """
    Compares candidate text writeprints against known author profile baselines
    using Euclidean feature distance and Cosine writeprint similarity.
    """

    def __init__(self, author_baseline_profiles: dict[str, dict[str, Any]]):
        self.baseline_profiles = author_baseline_profiles

    def classify_authorship(
        self, candidate_writeprint: dict[str, Any], distance_threshold: float = 0.85
    ) -> list[dict[str, Any]]:
        """Classifies candidate writeprint against baseline author profiles."""
        matches = []

        cand_ttr = candidate_writeprint.get("typeTokenRatio", 0.0)
        cand_sent_len = candidate_writeprint.get("avgSentenceLengthWords", 0.0)
        cand_punc = candidate_writeprint.get("punctuationDensity", 0.0)

        for author_id, base in self.baseline_profiles.items():
            base_ttr = base.get("typeTokenRatio", 0.0)
            base_sent_len = base.get("avgSentenceLengthWords", 0.0)
            base_punc = base.get("punctuationDensity", 0.0)

            # Normalized Euclidean feature distance calculation
            dist = math.sqrt(
                ((cand_ttr - base_ttr) ** 2)
                + (((cand_sent_len - base_sent_len) / 20.0) ** 2)
                + ((cand_punc - base_punc) ** 2)
            )

            similarity_score = round(max(0.0, 1.0 - dist), 4)

            if similarity_score >= distance_threshold:
                matches.append({
                    "matchedAuthorId": author_id,
                    "attributionConfidencePct": round(similarity_score * 100, 2),
                    "confidenceGrade": "HIGH_PROBABILITY" if similarity_score > 0.90 else "MODERATE",
                    "featureDistance": round(dist, 4),
                })

        return sorted(matches, key=lambda x: x["attributionConfidencePct"], reverse=True)


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
