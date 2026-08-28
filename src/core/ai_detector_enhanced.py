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
Enhanced AI-Generated Text Detection Module

Detects LLM-generated text using multiple techniques:
1. Perplexity-based detection
2. Burstiness analysis
3. Pattern recognition
4. Statistical footprint analysis
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AIDetectionResult:
    """AI detection result for a document."""

    document_name: str
    ai_probability: float  # 0-1
    perplexity_score: float
    burstiness_score: float
    pattern_score: float
    sentence_variability: float
    features: dict[str, Any] = field(default_factory=dict)
    is_suspicious: bool = False


class AIDetectorEnhanced:
    """
    Enhanced AI-generated text detector using multiple techniques.

    Detection Methods:
    1. Perplexity: AI text has lower perplexity (more predictable)
    2. Burstiness: Human text has more variation in sentence length
    3. Pattern Analysis: AI text has repetitive patterns
    4. Sentence Variability: Human text has more diverse sentence structures
    """

    def __init__(self):
        self._weights = {
            "perplexity": 0.30,
            "burstiness": 0.25,
            "pattern": 0.25,
            "sentence_variability": 0.20,
        }
        self._threshold = 0.65  # Above this = likely AI-generated

    def detect_document(
        self, text: str, doc_name: str = "unknown"
    ) -> AIDetectionResult:
        """
        Detect if a document is AI-generated.

        Args:
            text: Document text
            doc_name: Document name

        Returns:
            AIDetectionResult with scores
        """
        if not text or len(text) < 100:
            return AIDetectionResult(
                document_name=doc_name,
                ai_probability=0.0,
                perplexity_score=0.0,
                burstiness_score=0.0,
                pattern_score=0.0,
                sentence_variability=0.0,
                is_suspicious=False,
            )

        # Compute individual scores
        perplexity = self._compute_perplexity(text)
        burstiness = self._compute_burstiness(text)
        pattern_score = self._compute_pattern_score(text)
        sentence_var = self._compute_sentence_variability(text)

        # Compute combined probability
        ai_probability = (
            self._weights["perplexity"] * perplexity
            + self._weights["burstiness"] * burstiness
            + self._weights["pattern"] * pattern_score
            + self._weights["sentence_variability"] * sentence_var
        )

        # Normalize to 0-1
        ai_probability = min(1.0, max(0.0, ai_probability))

        return AIDetectionResult(
            document_name=doc_name,
            ai_probability=ai_probability,
            perplexity_score=perplexity,
            burstiness_score=burstiness,
            pattern_score=pattern_score,
            sentence_variability=sentence_var,
            features={
                "word_count": len(text.split()),
                "sentence_count": len(self._split_sentences(text)),
                "avg_word_length": self._avg_word_length(text),
                "unique_words_ratio": self._unique_words_ratio(text),
            },
            is_suspicious=ai_probability >= self._threshold,
        )

    def detect_batch(self, texts: dict[str, str]) -> dict[str, AIDetectionResult]:
        """Detect AI generation for multiple documents."""
        results = {}
        for doc_name, text in texts.items():
            results[doc_name] = self.detect_document(text, doc_name)
        return results

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity score for text.

        AI text typically has lower perplexity (more predictable).
        """
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if len(words) < 10:
            return 0.5

        # Count word frequencies
        freq = Counter(words)
        total = len(words)

        # Compute entropy
        entropy = 0.0
        for count in freq.values():
            prob = count / total
            entropy -= prob * math.log2(prob)

        # Perplexity = 2^entropy
        perplexity = 2**entropy

        # Normalize to 0-1 (higher perplexity = more human-like)
        # Normalize based on typical range: 2-20
        normalized = min(1.0, max(0.0, (perplexity - 2) / 18))
        # Convert to AI probability (lower perplexity = higher AI probability)
        return 1.0 - normalized

    def _compute_burstiness(self, text: str) -> float:
        """
        Compute burstiness score (variation in sentence length).

        Human text has higher burstiness (more variation).
        """
        sentences = self._split_sentences(text)
        if len(sentences) < 3:
            return 0.5

        # Calculate sentence lengths
        lengths = [len(s.split()) for s in sentences]

        # Compute coefficient of variation
        mean_len = np.mean(lengths)
        if mean_len == 0:
            return 0.5

        std_len = np.std(lengths)
        cv = std_len / mean_len

        # Normalize to 0-1 (higher CV = more human-like)
        normalized = min(1.0, cv / 2.0)  # CV typically 0-2

        # Convert to AI probability (lower CV = higher AI probability)
        return 1.0 - normalized

    def _compute_pattern_score(self, text: str) -> float:
        """
        Compute pattern repetition score.

        AI text often has repetitive patterns.
        """
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if len(words) < 20:
            return 0.5

        # Check for repeated phrases (3-grams)
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append((words[i], words[i + 1], words[i + 2]))

        if len(trigrams) < 3:
            return 0.5

        freq = Counter(trigrams)
        repeated = sum(1 for count in freq.values() if count > 1)
        repetition_ratio = repeated / len(freq) if freq else 0

        # Check for common AI phrases
        ai_phrases = [
            "it is important to note",
            "in conclusion",
            "furthermore",
            "in addition",
            "as a result",
            "it should be noted",
            "on the other hand",
            "in particular",
            "it is worth noting",
            "as we can see",
        ]

        phrase_count = 0
        text_lower = text.lower()
        for phrase in ai_phrases:
            if phrase in text_lower:
                phrase_count += 1

        ai_phrase_ratio = min(1.0, phrase_count / 10)

        # Combine scores
        pattern_score = 0.6 * repetition_ratio + 0.4 * ai_phrase_ratio
        return min(1.0, pattern_score)

    def _compute_sentence_variability(self, text: str) -> float:
        """
        Compute sentence structure variability.

        Human text has more varied sentence structures.
        """
        sentences = self._split_sentences(text)
        if len(sentences) < 5:
            return 0.5

        # Analyze sentence starts
        starts = []
        for sent in sentences:
            words = sent.split()
            if words:
                first_word = words[0].lower()
                starts.append(first_word)

        if not starts:
            return 0.5

        # Count unique sentence starts
        unique_starts = len(set(starts))
        start_ratio = unique_starts / len(starts)

        # Analyze sentence length variation
        lengths = [len(s.split()) for s in sentences]
        if len(lengths) > 1:
            length_variation = np.std(lengths) / (np.mean(lengths) + 1)
            length_variation = min(1.0, length_variation / 2)
        else:
            length_variation = 0

        # Lower variability = higher AI probability
        variability = 0.5 * start_ratio + 0.5 * length_variation
        return 1.0 - min(1.0, variability)

    def _avg_word_length(self, text: str) -> float:
        """Compute average word length."""
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        if not words:
            return 0
        return sum(len(w) for w in words) / len(words)

    def _unique_words_ratio(self, text: str) -> float:
        """Compute ratio of unique words."""
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words:
            return 0
        return len(set(words)) / len(words)

    def get_detection_summary(
        self, results: dict[str, AIDetectionResult]
    ) -> dict[str, Any]:
        """Get summary statistics for detection results."""
        if not results:
            return {"total_documents": 0}

        ai_count = sum(1 for r in results.values() if r.is_suspicious)
        avg_ai_prob = sum(r.ai_probability for r in results.values()) / len(results)

        return {
            "total_documents": len(results),
            "suspicious_documents": ai_count,
            "suspicious_percentage": (ai_count / len(results)) * 100,
            "average_ai_probability": avg_ai_prob,
            "threshold": self._threshold,
        }


# ============================================================================
# INTEGRATION WITH EXISTING AI DETECTION
# ============================================================================


def detect_ai_probability_enhanced(
    chunked_docs: dict[str, list[str]],
    threshold: float = 0.65,
) -> dict[str, float]:
    """
    Enhanced AI detection for document chunks.

    Args:
        chunked_docs: Dict mapping doc name to list of chunks
        threshold: AI detection threshold

    Returns:
        Dict mapping doc name to AI probability
    """
    detector = AIDetectorEnhanced()
    detector._threshold = threshold

    results = {}
    for doc_name, chunks in chunked_docs.items():
        text = " ".join(
            chunk.text if hasattr(chunk, "text") else chunk for chunk in chunks
        )
        result = detector.detect_document(text, doc_name)
        results[doc_name] = result.ai_probability

    return results


def classify_ai_generated(
    text: str,
    threshold: float = 0.65,
) -> tuple[bool, float, dict[str, Any]]:
    """
    Classify if text is AI-generated.

    Returns:
        Tuple of (is_ai, confidence, details)
    """
    detector = AIDetectorEnhanced()
    result = detector.detect_document(text)

    return (
        result.is_suspicious,
        result.ai_probability,
        {
            "perplexity": result.perplexity_score,
            "burstiness": result.burstiness_score,
            "pattern": result.pattern_score,
            "sentence_variability": result.sentence_variability,
            "features": result.features,
        },
    )
