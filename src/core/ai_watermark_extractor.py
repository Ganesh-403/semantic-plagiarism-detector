"""
src/core/ai_watermark_extractor.py
----------------------------------
AI-Generated Text Watermark Extraction Engine.

Analyzes text token probability distributions, n-gram frequencies, and extracts
statistical watermark features according to the Maryland watermarking scheme
(Kirchenbauer et al., 2023) and related LLM token distribution partitioning rules.
"""

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Simulated "Green List" of tokens that are biased by the watermarking scheme.
# Retained for compatibility with legacy / simplified extraction routines.
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


@dataclass
class TokenDistributionResult:
    """Represents empirical token probability distribution and entropy."""

    total_tokens: int
    unique_tokens: int
    frequencies: dict[str, int]
    probabilities: dict[str, float]
    entropy: float
    top_tokens: list[tuple[str, float]]


@dataclass
class WatermarkExtractionResult:
    """Represents extracted green/red list token statistics for a document."""

    total_scored_tokens: int
    green_token_count: int
    red_token_count: int
    observed_green_ratio: float
    expected_green_ratio: float
    secret_key: str
    context_window_size: int
    token_details: list[dict[str, Any]] = field(default_factory=list)
    ngram_frequencies: dict[int, dict[str, int]] = field(default_factory=dict)
    token_distribution: Optional[TokenDistributionResult] = None


class AIWatermarkExtractor:
    """Extractor for token distributions, n-gram frequencies, and watermark features."""

    def __init__(
        self,
        secret_key: str = "default_maryland_key",
        gamma: float = 0.5,
        context_window_size: int = 1,
    ):
        """Initialize the AI Watermark Extractor.

        Args:
            secret_key: Secret key or seed used for pseudo-random green/red partitioning.
            gamma: Expected proportion of the vocabulary assigned to the green list (0 < gamma < 1).
            context_window_size: Number of preceding tokens used as conditioning context (k-gram context).
        """
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"Gamma must be between 0.0 and 1.0 (exclusive), got {gamma}")
        if context_window_size < 0:
            raise ValueError(f"Context window size must be non-negative, got {context_window_size}")

        self.secret_key = secret_key
        self.gamma = gamma
        self.context_window_size = context_window_size

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize text into normalized word and punctuation tokens.

        Preserves lexical tokens and punctuation without stripping essential sequence boundaries.
        """
        if not text:
            return []
        # Matches word sequences or individual punctuation marks
        return re.findall(r"\b\w+\b|[^\w\s]", text, re.UNICODE)

    @classmethod
    def extract_token_probability_distribution(cls, tokens: list[str]) -> TokenDistributionResult:
        """Extract empirical token frequency, probability distribution, and Shannon entropy.

        Args:
            tokens: Sequence of tokens.

        Returns:
            TokenDistributionResult with frequencies, probabilities, and entropy.
        """
        total_tokens = len(tokens)
        if total_tokens == 0:
            return TokenDistributionResult(
                total_tokens=0,
                unique_tokens=0,
                frequencies={},
                probabilities={},
                entropy=0.0,
                top_tokens=[],
            )

        freqs = Counter(tokens)
        probs = {token: count / total_tokens for token, count in freqs.items()}

        # Compute Shannon entropy: H = -sum(p * log2(p))
        entropy = -sum(p * math.log2(p) for p in probs.values() if p > 0.0)

        # Sort tokens by descending probability
        top_tokens = sorted(probs.items(), key=lambda item: item[1], reverse=True)

        return TokenDistributionResult(
            total_tokens=total_tokens,
            unique_tokens=len(freqs),
            frequencies=dict(freqs),
            probabilities=probs,
            entropy=round(entropy, 6),
            top_tokens=top_tokens,
        )

    @classmethod
    def extract_ngram_frequencies(
        cls, tokens: list[str], n_values: Optional[list[int]] = None
    ) -> dict[int, dict[str, int]]:
        """Extract n-gram frequencies for given n values.

        Args:
            tokens: Sequence of tokens.
            n_values: List of n sizes to compute (default: [1, 2, 3]).

        Returns:
            Dictionary mapping n -> {ngram_string: frequency_count}.
        """
        if n_values is None:
            n_values = [1, 2, 3]

        result: dict[int, dict[str, int]] = {}
        total = len(tokens)

        for n in n_values:
            if n <= 0:
                continue
            if total < n:
                result[n] = {}
                continue

            ngrams = [" ".join(tokens[i : i + n]) for i in range(total - n + 1)]
            result[n] = dict(Counter(ngrams))

        return result

    def compute_context_hash(self, context_tokens: list[str]) -> int:
        """Compute a deterministic 64-bit integer hash from context tokens and secret key."""
        context_str = "|".join(context_tokens)
        payload = f"{self.secret_key}::{context_str}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def is_green_token(self, token: str, context_tokens: list[str]) -> bool:
        """Determine if a token belongs to the green list given preceding context tokens.

        Implements the deterministic pseudorandom hash partition from Kirchenbauer et al.:
        Hash(secret_key, context_tokens, token) mapped to [0, 1) compared against gamma.
        """
        context_str = "|".join(context_tokens)
        payload = f"{self.secret_key}::{context_str}::{token}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        # Map 32-bit hash chunk to uniform float in [0.0, 1.0)
        hash_val = int(digest[:8], 16)
        normalized_val = hash_val / 0xFFFFFFFF
        return normalized_val < self.gamma

    def extract_features(
        self,
        text: str,
        include_token_details: bool = True,
        ngram_sizes: Optional[list[int]] = None,
    ) -> WatermarkExtractionResult:
        """Extract all statistical watermark and distribution features from raw text.

        Args:
            text: Input document or snippet text.
            include_token_details: Whether to return per-token classification details.
            ngram_sizes: Optional n-gram sizes to extract (defaults to [1, 2, 3]).

        Returns:
            WatermarkExtractionResult containing counts, ratios, distributions, and n-grams.
        """
        tokens = self.tokenize(text)
        token_dist = self.extract_token_probability_distribution(tokens)
        ngrams = self.extract_ngram_frequencies(tokens, ngram_sizes)

        k = self.context_window_size
        total_tokens = len(tokens)

        # Scored tokens start after the initial context window (or index 0 if k == 0)
        if total_tokens <= k:
            return WatermarkExtractionResult(
                total_scored_tokens=0,
                green_token_count=0,
                red_token_count=0,
                observed_green_ratio=0.0,
                expected_green_ratio=self.gamma,
                secret_key=self.secret_key,
                context_window_size=k,
                token_details=[],
                ngram_frequencies=ngrams,
                token_distribution=token_dist,
            )

        green_count = 0
        red_count = 0
        token_details = []

        start_idx = k
        for i in range(start_idx, total_tokens):
            current_token = tokens[i]
            context = tokens[i - k : i] if k > 0 else []
            green = self.is_green_token(current_token, context)

            if green:
                green_count += 1
            else:
                red_count += 1

            if include_token_details:
                token_details.append(
                    {
                        "index": i,
                        "token": current_token,
                        "context": context,
                        "is_green": green,
                    }
                )

        scored_count = green_count + red_count
        observed_ratio = (green_count / scored_count) if scored_count > 0 else 0.0

        return WatermarkExtractionResult(
            total_scored_tokens=scored_count,
            green_token_count=green_count,
            red_token_count=red_count,
            observed_green_ratio=round(observed_ratio, 6),
            expected_green_ratio=self.gamma,
            secret_key=self.secret_key,
            context_window_size=k,
            token_details=token_details,
            ngram_frequencies=ngrams,
            token_distribution=token_dist,
        )


def extract_token_distributions(text: str) -> TokenDistributionResult:
    """Convenience helper to extract token probability distribution and entropy."""
    tokens = AIWatermarkExtractor.tokenize(text)
    return AIWatermarkExtractor.extract_token_probability_distribution(tokens)


def extract_ngram_frequencies(
    text: str, n_values: Optional[list[int]] = None
) -> dict[int, dict[str, int]]:
    """Convenience helper to extract n-gram frequencies from text."""
    tokens = AIWatermarkExtractor.tokenize(text)
    return AIWatermarkExtractor.extract_ngram_frequencies(tokens, n_values)


def extract_watermark_features(
    text: str,
    secret_key: str = "default_maryland_key",
    gamma: float = 0.5,
    context_window_size: int = 1,
) -> WatermarkExtractionResult:
    """Convenience helper to extract watermark token counts and green/red ratio."""
    extractor = AIWatermarkExtractor(
        secret_key=secret_key, gamma=gamma, context_window_size=context_window_size
    )
    return extractor.extract_features(text)


def extract_token_distribution(text: str) -> dict[str, Any]:
    """Extract token frequencies and simulated green list metrics from text.

    Provided for backward compatibility.
    """
    if not text or not isinstance(text, str):
        return {"total_tokens": 0, "green_list_count": 0, "green_list_ratio": 0.0}

    tokens = re.findall(r"\b\w+\b", text.lower())
    total_tokens = len(tokens)

    if total_tokens == 0:
        return {"total_tokens": 0, "green_list_count": 0, "green_list_ratio": 0.0}

    green_list_count = sum(1 for token in tokens if token in GREEN_LIST_TOKENS)
    green_list_ratio = green_list_count / total_tokens

    return {
        "total_tokens": total_tokens,
        "green_list_count": green_list_count,
        "green_list_ratio": round(green_list_ratio, 4),
    }


def compute_ngram_frequencies(text: str, n: int = 2) -> dict[str, int]:
    """Compute n-gram frequencies for the text.

    Provided for backward compatibility.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    if len(tokens) < n:
        return {}

    ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return dict(Counter(ngrams))
