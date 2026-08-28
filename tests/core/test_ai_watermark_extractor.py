"""
tests/core/test_ai_watermark_extractor.py
-----------------------------------------
Unit tests for AI Watermark Extractor (token distribution, n-grams, and green/red list partitioning).
"""

import math
import pytest

from src.core.ai_watermark_extractor import (
    AIWatermarkExtractor,
    TokenDistributionResult,
    WatermarkExtractionResult,
    extract_ngram_frequencies,
    extract_token_distributions,
    extract_watermark_features,
)


class TestAIWatermarkExtractor:
    """Tests for AIWatermarkExtractor token analysis and feature extraction."""

    def test_tokenize_basic_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        tokens = AIWatermarkExtractor.tokenize(text)
        assert len(tokens) == 10  # 9 words + 1 period
        assert tokens[0] == "The"
        assert tokens[-1] == "."

    def test_tokenize_empty_and_whitespace(self):
        assert AIWatermarkExtractor.tokenize("") == []
        assert AIWatermarkExtractor.tokenize("   \n\t  ") == []

    def test_token_probability_distribution(self):
        tokens = ["apple", "banana", "apple", "orange", "apple", "banana"]
        dist = AIWatermarkExtractor.extract_token_probability_distribution(tokens)

        assert isinstance(dist, TokenDistributionResult)
        assert dist.total_tokens == 6
        assert dist.unique_tokens == 3
        assert dist.frequencies["apple"] == 3
        assert dist.frequencies["banana"] == 2
        assert dist.frequencies["orange"] == 1

        assert pytest.approx(dist.probabilities["apple"], 0.001) == 0.5
        assert pytest.approx(dist.probabilities["banana"], 0.001) == 2 / 6
        assert pytest.approx(dist.probabilities["orange"], 0.001) == 1 / 6
        assert pytest.approx(sum(dist.probabilities.values()), 0.001) == 1.0

        # Theoretical entropy: -(0.5*log2(0.5) + (1/3)*log2(1/3) + (1/6)*log2(1/6))
        expected_entropy = -(0.5 * math.log2(0.5) + (1 / 3) * math.log2(1 / 3) + (1 / 6) * math.log2(1 / 6))
        assert pytest.approx(dist.entropy, 0.001) == expected_entropy
        assert dist.top_tokens[0][0] == "apple"

    def test_token_probability_distribution_empty(self):
        dist = AIWatermarkExtractor.extract_token_probability_distribution([])
        assert dist.total_tokens == 0
        assert dist.unique_tokens == 0
        assert dist.frequencies == {}
        assert dist.probabilities == {}
        assert dist.entropy == 0.0
        assert dist.top_tokens == []

    def test_ngram_frequencies(self):
        tokens = ["the", "quick", "brown", "fox", "the", "quick"]
        ngrams = AIWatermarkExtractor.extract_ngram_frequencies(tokens, n_values=[1, 2, 3])

        assert 1 in ngrams
        assert 2 in ngrams
        assert 3 in ngrams

        assert ngrams[1]["the"] == 2
        assert ngrams[2]["the quick"] == 2
        assert ngrams[2]["quick brown"] == 1
        assert ngrams[3]["the quick brown"] == 1
        assert ngrams[3]["quick brown fox"] == 1

    def test_ngram_frequencies_text_shorter_than_n(self):
        tokens = ["hello", "world"]
        ngrams = AIWatermarkExtractor.extract_ngram_frequencies(tokens, n_values=[1, 3, 5])
        assert len(ngrams[1]) == 2
        assert ngrams[3] == {}
        assert ngrams[5] == {}

    def test_deterministic_green_list_partition(self):
        extractor1 = AIWatermarkExtractor(secret_key="my_secret_key_123", gamma=0.5, context_window_size=1)
        extractor2 = AIWatermarkExtractor(secret_key="my_secret_key_123", gamma=0.5, context_window_size=1)
        extractor3 = AIWatermarkExtractor(secret_key="different_key", gamma=0.5, context_window_size=1)

        token = "intelligence"
        context = ["artificial"]

        res1 = extractor1.is_green_token(token, context)
        res2 = extractor2.is_green_token(token, context)
        assert res1 == res2

        # Changing context should deterministically change context hashing
        res_diff_context = extractor1.is_green_token(token, ["biological"])
        assert isinstance(res_diff_context, bool)

    def test_extract_features_counts_and_ratios(self):
        text = (
            "Large language models can generate coherent and fluent natural language. "
            "Statistical watermarking provides a verifiable method to detect synthetic text."
        )
        extractor = AIWatermarkExtractor(secret_key="test_key", gamma=0.5, context_window_size=1)
        result = extractor.extract_features(text, include_token_details=True)

        assert isinstance(result, WatermarkExtractionResult)
        assert result.total_scored_tokens > 0
        assert result.green_token_count + result.red_token_count == result.total_scored_tokens
        assert result.observed_green_ratio == pytest.approx(
            result.green_token_count / result.total_scored_tokens, 0.0001
        )
        assert len(result.token_details) == result.total_scored_tokens
        assert result.token_distribution is not None
        assert result.token_distribution.total_tokens > result.total_scored_tokens

    def test_extract_features_without_token_details(self):
        text = "Short sentence for quick evaluation."
        extractor = AIWatermarkExtractor(secret_key="key", gamma=0.5, context_window_size=1)
        result = extractor.extract_features(text, include_token_details=False)
        assert result.token_details == []
        assert result.total_scored_tokens > 0

    def test_extract_features_empty_and_short_text(self):
        extractor = AIWatermarkExtractor(secret_key="key", gamma=0.5, context_window_size=2)
        res_empty = extractor.extract_features("")
        assert res_empty.total_scored_tokens == 0
        assert res_empty.green_token_count == 0
        assert res_empty.red_token_count == 0

        res_short = extractor.extract_features("One")
        assert res_short.total_scored_tokens == 0

    def test_context_window_size_zero(self):
        extractor = AIWatermarkExtractor(secret_key="key", gamma=0.5, context_window_size=0)
        result = extractor.extract_features("One two three four five")
        assert result.total_scored_tokens == 5
        assert result.green_token_count + result.red_token_count == 5

    def test_invalid_parameters_raise_error(self):
        with pytest.raises(ValueError, match="Gamma must be between"):
            AIWatermarkExtractor(gamma=0.0)

        with pytest.raises(ValueError, match="Gamma must be between"):
            AIWatermarkExtractor(gamma=1.0)

        with pytest.raises(ValueError, match="Context window size must be non-negative"):
            AIWatermarkExtractor(context_window_size=-1)

    def test_convenience_functions(self):
        text = "Natural language processing with machine learning."
        dist = extract_token_distributions(text)
        assert dist.total_tokens == 7  # 6 words + 1 period token

        ngrams = extract_ngram_frequencies(text, [2])
        assert len(ngrams[2]) == 6  # 7 tokens produce 6 bigrams

        features = extract_watermark_features(text, secret_key="k1", gamma=0.4)
        assert features.expected_green_ratio == 0.4
        assert features.secret_key == "k1"
