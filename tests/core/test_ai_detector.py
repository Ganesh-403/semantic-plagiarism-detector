"""
test_ai_detector.py
-------------------
Tests for AI-generated text detection functionality.

Includes tests for:
- Probability detection batch and single text functions
- Confidence tier categorization
- Document-level AI detection statistics
- Text perplexity scoring helper (Issue #1154)
"""

import math
from unittest.mock import MagicMock, patch
import pytest

from src.core.ai_detector import (
    calculate_text_perplexity,
    categorize_ai_probability,
    detect_ai_generated_text,
    detect_ai_probability,
    detect_ai_probability_batch,
    detect_document_ai_probability,
    detect_documents_ai_probability,
)


def test_categorize_ai_probability_boundaries():
    """Verify that the confidence categorization partitions the [0,1] range correctly."""
    assert categorize_ai_probability(0.85) == "High Probability"
    assert categorize_ai_probability(0.80) == "High Probability"
    assert categorize_ai_probability(0.79) == "Moderate Probability"
    assert categorize_ai_probability(0.65) == "Moderate Probability"
    assert categorize_ai_probability(0.50) == "Moderate Probability"
    assert categorize_ai_probability(0.49) == "Low Probability"
    assert categorize_ai_probability(0.20) == "Low Probability"
    assert categorize_ai_probability(0.00) == "Low Probability"


@pytest.fixture(autouse=True)
def mock_transformers_pipeline():
    """Autouse fixture to mock Hugging Face pipeline across all tests in this module."""
    with patch("transformers.pipeline") as mock_pipe:
        mock_classifier = MagicMock()
        # Mock pipeline output format: [{'label': 'Fake', 'score': 0.85}]
        mock_classifier.return_value = [[{"label": "Fake", "score": 0.85}]]
        mock_pipe.return_value = mock_classifier
        yield mock_pipe


def test_detect_ai_probability_empty_text():
    """Test that empty text returns 0.0 probability."""
    result = detect_ai_probability("")
    assert result == 0.0


def test_detect_ai_probability_none():
    """Test that None input returns 0.0 probability."""
    result = detect_ai_probability(None)
    assert result == 0.0


def test_detect_ai_probability_whitespace_only():
    """Test that whitespace-only text returns 0.0 probability."""
    result = detect_ai_probability("   \n\t  ")
    assert result == 0.0


def test_detect_ai_probability_batch_empty():
    """Test that empty list returns empty list."""
    result = detect_ai_probability_batch([])
    assert result == []


def test_detect_document_ai_probability_empty():
    """Test that empty chunks return zero probabilities."""
    result = detect_document_ai_probability([])
    assert result["overall"] == 0.0
    assert result["max"] == 0.0
    assert result["chunk_scores"] == []


def test_detect_documents_ai_probability_empty():
    """Test that empty dict returns empty dict."""
    result = detect_documents_ai_probability({})
    assert result == {}


def test_detect_documents_ai_probability_single_doc():
    """Test AI detection with a single document."""
    chunked_docs = {
        "test_doc.txt": ["This is a test chunk of text.", "Another test chunk here."]
    }
    result = detect_documents_ai_probability(chunked_docs)

    assert "test_doc.txt" in result
    assert "overall" in result["test_doc.txt"]
    assert "max" in result["test_doc.txt"]
    assert "chunk_scores" in result["test_doc.txt"]
    assert len(result["test_doc.txt"]["chunk_scores"]) == 2
    assert 0.0 <= result["test_doc.txt"]["overall"] <= 1.0
    assert 0.0 <= result["test_doc.txt"]["max"] <= 1.0


def test_detect_ai_probability_batch_mixed():
    """Test batch detection with mixed empty and non-empty texts."""
    texts = ["Some text", "", None, "More text"]
    result = detect_ai_probability_batch(texts)

    assert len(result) == 4
    assert result[1] == 0.0  # Empty string
    assert result[2] == 0.0  # None
    assert 0.0 <= result[0] <= 1.0
    assert 0.0 <= result[3] <= 1.0


def test_detect_ai_generated_text_empty():
    """Verify that empty text returns low confidence default dictionary."""
    res = detect_ai_generated_text("")
    assert res["ai_probability"] == 0.0
    assert res["confidence_tier"] == "low"
    assert res["perplexity_score"] == 150.0
    assert res["burstiness_score"] == 0.0
    assert res["ngram_repetitiveness"] == 0.0
    assert res["classification_tier"] == "low"


def test_detect_ai_generated_text_whitespace():
    """Verify that whitespace-only text returns low confidence default dictionary."""
    res = detect_ai_generated_text("   \n\t  ")
    assert res["ai_probability"] == 0.0
    assert res["confidence_tier"] == "low"
    assert res["perplexity_score"] == 150.0
    assert res["burstiness_score"] == 0.0
    assert res["ngram_repetitiveness"] == 0.0


def test_detect_ai_generated_text_tiers():
    """Verify that confidence categorizations partition correctly with multi-metric scores."""
    with patch("src.core.ai_detector.detect_ai_probability") as mock_prob, \
         patch("src.core.ai_detector.calculate_text_perplexity") as mock_perp, \
         patch("src.core.ai_detector._calculate_burstiness") as mock_burst, \
         patch("src.core.ai_detector._calculate_ngram_repetitiveness") as mock_ngram:

        mock_perp.return_value = 50.0
        mock_burst.return_value = 0.3
        mock_ngram.return_value = 0.2

        # High confidence AI (>= 0.75)
        mock_prob.return_value = 0.85
        res = detect_ai_generated_text("Test AI text")
        assert res["ai_probability"] == 0.85
        assert res["confidence_tier"] == "high"
        assert res["classification_tier"] == "high"
        assert res["perplexity_score"] == 50.0
        assert res["burstiness_score"] == 0.3
        assert res["ngram_repetitiveness"] == 0.2

        # Medium confidence (0.40 <= prob < 0.75)
        mock_prob.return_value = 0.55
        res = detect_ai_generated_text("Test medium text")
        assert res["ai_probability"] == 0.55
        assert res["confidence_tier"] == "medium"
        assert res["classification_tier"] == "medium"

        # Low confidence (< 0.40)
        mock_prob.return_value = 0.25
        res = detect_ai_generated_text("Test human text")
        assert res["ai_probability"] == 0.25
        assert res["confidence_tier"] == "low"
        assert res["classification_tier"] == "low"


# ─── Tests for multi-metric classifier (Issue #1356) ─────────────────────────


HUMAN_TEXT = (
    "The autumn leaves danced in the wind. A sudden chill crept through the valley, "
    "reminding everyone that winter was coming. The old man sighed, pulling his coat "
    "tighter. 'Another year gone,' he muttered to no one in particular. The children "
    "playing in the distance didn't notice. They were too busy building a fort out of "
    "branches and old blankets. Meanwhile, the baker across the street was already "
    "preparing for the morning rush. The smell of fresh bread wafted through the air. "
    "Life went on, as it always does, indifferent to the changing seasons."
)

AI_TEXT = (
    "In today's rapidly evolving technological landscape, artificial intelligence has "
    "become an integral part of our daily lives. From smartphones to smart homes, AI "
    "technology is everywhere. Moreover, the impact of artificial intelligence extends "
    "far beyond consumer applications. In the business world, companies are leveraging "
    "AI to streamline operations and improve efficiency. Furthermore, the healthcare "
    "industry has also embraced artificial intelligence for diagnostic purposes. In "
    "conclusion, artificial intelligence will continue to shape our future in profound ways."
)


def test_multi_classifier_returns_all_metrics():
    """detect_ai_generated_text must return all four metric fields."""
    with patch("src.core.ai_detector.detect_ai_probability", return_value=0.7), \
         patch("src.core.ai_detector.calculate_text_perplexity", return_value=42.0), \
         patch("src.core.ai_detector._calculate_burstiness", return_value=0.5), \
         patch("src.core.ai_detector._calculate_ngram_repetitiveness", return_value=0.3):
        result = detect_ai_generated_text("Some test text for analysis.")

    assert "ai_probability" in result
    assert "perplexity_score" in result
    assert "burstiness_score" in result
    assert "ngram_repetitiveness" in result
    assert "classification_tier" in result
    assert "confidence_tier" in result


def test_multi_classifier_synthetic_human_text():
    """Human-like text (low AI prob, high burstiness) should classify as 'low'."""
    with patch("src.core.ai_detector.detect_ai_probability", return_value=0.15), \
         patch("src.core.ai_detector.calculate_text_perplexity", return_value=180.0), \
         patch("src.core.ai_detector._calculate_burstiness", return_value=0.75), \
         patch("src.core.ai_detector._calculate_ngram_repetitiveness", return_value=0.1):
        result = detect_ai_generated_text(HUMAN_TEXT)

    assert result["ai_probability"] < 0.4
    assert result["confidence_tier"] == "low"
    assert result["classification_tier"] == "low"
    assert result["burstiness_score"] > 0.5  # human text is bursty
    assert result["ngram_repetitiveness"] < 0.3  # human text is less repetitive


def test_multi_classifier_synthetic_ai_text():
    """AI-like text (high AI prob, low burstiness, high repetition) should classify as 'high'."""
    with patch("src.core.ai_detector.detect_ai_probability", return_value=0.88), \
         patch("src.core.ai_detector.calculate_text_perplexity", return_value=25.0), \
         patch("src.core.ai_detector._calculate_burstiness", return_value=0.15), \
         patch("src.core.ai_detector._calculate_ngram_repetitiveness", return_value=0.6):
        result = detect_ai_generated_text(AI_TEXT)

    assert result["ai_probability"] >= 0.75
    assert result["confidence_tier"] == "high"
    assert result["classification_tier"] == "high"
    assert result["burstiness_score"] < 0.5  # AI text is uniform
    assert result["ngram_repetitiveness"] > 0.3  # AI text is more repetitive


def test_burstiness_empty_text():
    """Burstiness of empty text must be 0.0."""
    from src.core.ai_detector import _calculate_burstiness
    assert _calculate_burstiness("") == 0.0
    assert _calculate_burstiness(None) == 0.0
    assert _calculate_burstiness("   ") == 0.0


def test_burstiness_single_sentence():
    """Burstiness of a single sentence must be 0.0 (no variation)."""
    from src.core.ai_detector import _calculate_burstiness
    assert _calculate_burstiness("Only one sentence here.") == 0.0


def test_burstiness_varied_sentence_lengths():
    """Burstiness should be higher for text with varied sentence lengths."""
    from src.core.ai_detector import _calculate_burstiness
    varied = "Short. This is a much longer sentence with many words. Medium one here."
    uniform = "This is a sentence. This is a sentence. This is a sentence."
    assert _calculate_burstiness(varied) > _calculate_burstiness(uniform)


def test_ngram_repetitiveness_empty_text():
    """N-gram repetitiveness of empty text must be 0.0."""
    from src.core.ai_detector import _calculate_ngram_repetitiveness
    assert _calculate_ngram_repetitiveness("") == 0.0
    assert _calculate_ngram_repetitiveness(None) == 0.0
    assert _calculate_ngram_repetitiveness("   ") == 0.0


def test_ngram_repetitiveness_short_text():
    """N-gram repetitiveness of text shorter than n must be 0.0."""
    from src.core.ai_detector import _calculate_ngram_repetitiveness
    assert _calculate_ngram_repetitiveness("hi") == 0.0
    assert _calculate_ngram_repetitiveness("one two") == 0.0


def test_ngram_repetitiveness_no_repeats():
    """Text with no repeated n-grams should return ~0.0."""
    from src.core.ai_detector import _calculate_ngram_repetitiveness
    unique = "the quick brown fox jumps over lazy dog runs fast today"
    result = _calculate_ngram_repetitiveness(unique, n=3)
    assert result < 0.1


def test_ngram_repetitiveness_high_repetition():
    """Text with repeated n-grams should return a high score."""
    from src.core.ai_detector import _calculate_ngram_repetitiveness
    repetitive = "the the the the the the the the the the"
    result = _calculate_ngram_repetitiveness(repetitive, n=2)
    assert result > 0.5


# ─── Tests for calculate_text_perplexity (Issue #1154) ──────────────────────────


def test_calculate_text_perplexity_empty_string():
    """Empty string must return the default perplexity score of 0.0."""
    result = calculate_text_perplexity("")
    assert result == 0.0


def test_calculate_text_perplexity_none_input():
    """None input must return the default perplexity score of 0.0."""
    result = calculate_text_perplexity(None)
    assert result == 0.0


def test_calculate_text_perplexity_whitespace_only():
    """Whitespace-only text must return the default perplexity score of 0.0."""
    result = calculate_text_perplexity("   \n\t  ")
    assert result == 0.0


def test_calculate_text_perplexity_returns_float():
    """The return type must always be a float."""
    result = calculate_text_perplexity("This is a valid sentence for testing.")
    assert isinstance(result, float)


def test_calculate_text_perplexity_with_fallback_model():
    """When model is in fallback mode, return default perplexity score."""
    with patch("src.core.ai_detector._get_model_and_tokenizer") as mock_loader:
        mock_loader.return_value = ("fallback", "fallback")
        result = calculate_text_perplexity("Some text to evaluate.")
        assert result == 0.0


def test_calculate_text_perplexity_with_mock_model():
    """Verify perplexity calculation with a mocked transformer model and tokenizer."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Mock the model config to provide a max_length value
    mock_config = MagicMock()
    mock_config.max_position_embeddings = 512
    mock_model.config = mock_config

    # Mock the tokenizer output with tensors that simulate tokenized input
    mock_input_ids = MagicMock()
    mock_input_ids.to = MagicMock(return_value=mock_input_ids)

    mock_attention_mask = MagicMock()
    mock_attention_mask.to = MagicMock(return_value=mock_attention_mask)

    mock_tokenizer.return_value = {
        "input_ids": mock_input_ids,
        "attention_mask": mock_attention_mask,
    }

    # Mock the model output with a specific loss value
    # loss = ln(perplexity), so perplexity = exp(loss)
    # If we want perplexity = 50.0, loss = ln(50) ≈ 3.912
    mock_loss_value = math.log(50.0)
    mock_outputs = MagicMock()
    mock_outputs.loss = MagicMock()
    mock_outputs.loss.item = MagicMock(return_value=mock_loss_value)

    # The float() call on loss will use the item() value
    # We need to mock __float__ on the loss tensor
    type(mock_outputs.loss).__float__ = MagicMock(return_value=mock_loss_value)

    mock_model.return_value = mock_outputs
    mock_model.to = MagicMock(return_value=mock_model)

    with patch(
        "src.core.ai_detector._get_model_and_tokenizer",
        return_value=(mock_model, mock_tokenizer),
    ):
        # Reset the global model/tokenizer to force re-loading
        import src.core.ai_detector as module

        original_model = module._model
        original_tokenizer = module._tokenizer
        module._model = mock_model
        module._tokenizer = mock_tokenizer

        try:
            result = calculate_text_perplexity(
                "The quick brown fox jumps over the lazy dog."
            )
            # Result should be a float value
            assert isinstance(result, float)
            # Perplexity should be >= 0
            assert result >= 0.0
        finally:
            # Restore original globals
            module._model = original_model
            module._tokenizer = original_tokenizer


def test_calculate_text_perplexity_handles_exception_gracefully():
    """If the model throws an unexpected error, return default perplexity score."""
    with patch("src.core.ai_detector._get_model_and_tokenizer") as mock_loader:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.return_value = (mock_model, mock_tokenizer)

        # Make the tokenizer raise an exception to simulate a failure
        mock_tokenizer.side_effect = RuntimeError("Simulated tokenizer failure")

        import src.core.ai_detector as module

        original_model = module._model
        original_tokenizer = module._tokenizer
        module._model = mock_model
        module._tokenizer = mock_tokenizer

        try:
            result = calculate_text_perplexity("Some text that will fail tokenization.")
            assert result == 0.0
        finally:
            module._model = original_model
            module._tokenizer = original_tokenizer


def test_calculate_text_perplexity_value_error_handling():
    """ValueError during perplexity computation must return the default score."""
    with patch("src.core.ai_detector._get_model_and_tokenizer") as mock_loader:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader.return_value = (mock_model, mock_tokenizer)

        # Make the tokenizer raise ValueError
        mock_tokenizer.side_effect = ValueError("Text too short to tokenize")

        import src.core.ai_detector as module

        original_model = module._model
        original_tokenizer = module._tokenizer
        module._model = mock_model
        module._tokenizer = mock_tokenizer

        try:
            result = calculate_text_perplexity("ab")
            assert result == 0.0
        finally:
            module._model = original_model
            module._tokenizer = original_tokenizer


def test_calculate_text_perplexity_non_string_input():
    """Non-string inputs must return the default perplexity score."""
    assert calculate_text_perplexity(123) == 0.0
    assert calculate_text_perplexity([]) == 0.0
    assert calculate_text_perplexity({}) == 0.0
    assert calculate_text_perplexity(True) == 0.0


def test_calculate_text_perplexity_long_text():
    """Very long text should be handled gracefully with truncation."""
    long_text = "This is a sentence. " * 500
    result = calculate_text_perplexity(long_text)
    assert isinstance(result, float)
    assert result >= 0.0


def test_calculate_text_perplexity_returns_non_negative():
    """Perplexity must always be a non-negative value."""
    with patch("src.core.ai_detector._get_model_and_tokenizer") as mock_loader:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_config = MagicMock()
        mock_config.max_position_embeddings = 512
        mock_model.config = mock_config

        mock_input_ids = MagicMock()
        mock_input_ids.to = MagicMock(return_value=mock_input_ids)
        mock_attention_mask = MagicMock()
        mock_attention_mask.to = MagicMock(return_value=mock_attention_mask)

        mock_tokenizer.return_value = {
            "input_ids": mock_input_ids,
            "attention_mask": mock_attention_mask,
        }

        # Create a loss that yields a positive perplexity
        mock_loss_value = 3.0  # exp(3.0) ≈ 20.09
        mock_outputs = MagicMock()
        mock_outputs.loss = MagicMock()
        type(mock_outputs.loss).__float__ = MagicMock(return_value=mock_loss_value)

        mock_model.return_value = mock_outputs
        mock_model.to = MagicMock(return_value=mock_model)

        mock_loader.return_value = (mock_model, mock_tokenizer)

        import src.core.ai_detector as module

        original_model = module._model
        original_tokenizer = module._tokenizer
        module._model = mock_model
        module._tokenizer = mock_tokenizer

        try:
            result = calculate_text_perplexity("A reasonable length text for analysis.")
            assert result >= 0.0
        finally:
            module._model = original_model
            module._tokenizer = original_tokenizer


def test_calculate_text_perplexity_clamps_large_values():
    """Extremely large perplexity values must be clamped to prevent overflow."""
    with patch("src.core.ai_detector._get_model_and_tokenizer") as mock_loader:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_config = MagicMock()
        mock_config.max_position_embeddings = 512
        mock_model.config = mock_config

        mock_input_ids = MagicMock()
        mock_input_ids.to = MagicMock(return_value=mock_input_ids)
        mock_attention_mask = MagicMock()
        mock_attention_mask.to = MagicMock(return_value=mock_attention_mask)

        mock_tokenizer.return_value = {
            "input_ids": mock_input_ids,
            "attention_mask": mock_attention_mask,
        }

        # Create a very large loss value: exp(100) would overflow
        # The function should clamp it to 10000.0
        mock_loss_value = 100.0
        mock_outputs = MagicMock()
        mock_outputs.loss = MagicMock()
        type(mock_outputs.loss).__float__ = MagicMock(return_value=mock_loss_value)

        mock_model.return_value = mock_outputs
        mock_model.to = MagicMock(return_value=mock_model)

        mock_loader.return_value = (mock_model, mock_tokenizer)

        import src.core.ai_detector as module

        original_model = module._model
        original_tokenizer = module._tokenizer
        module._model = mock_model
        module._tokenizer = mock_tokenizer

        try:
            result = calculate_text_perplexity(
                "Text with extreme perplexity potential."
            )
            assert result <= 10000.0
            assert result >= 0.0
        finally:
            module._model = original_model
            module._tokenizer = original_tokenizer


def test_calculate_text_perplexity_none_loss():
    """If model returns None for loss, return default perplexity score."""
    with patch("src.core.ai_detector._get_model_and_tokenizer") as mock_loader:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_config = MagicMock()
        mock_config.max_position_embeddings = 512
        mock_model.config = mock_config

        mock_input_ids = MagicMock()
        mock_input_ids.to = MagicMock(return_value=mock_input_ids)
        mock_attention_mask = MagicMock()
        mock_attention_mask.to = MagicMock(return_value=mock_attention_mask)

        mock_tokenizer.return_value = {
            "input_ids": mock_input_ids,
            "attention_mask": mock_attention_mask,
        }

        # Model returns None loss
        mock_outputs = MagicMock()
        mock_outputs.loss = None

        mock_model.return_value = mock_outputs
        mock_model.to = MagicMock(return_value=mock_model)

        mock_loader.return_value = (mock_model, mock_tokenizer)

        import src.core.ai_detector as module

        original_model = module._model
        original_tokenizer = module._tokenizer
        module._model = mock_model
        module._tokenizer = mock_tokenizer

        try:
            result = calculate_text_perplexity("Text where loss is None.")
            assert result == 0.0
        finally:
            module._model = original_model
            module._tokenizer = original_tokenizer


# ─── Tests for Stylometric Feature Extractor (Issue #1484) ─────────────────────

from src.core.ai_detector import extract_stylometric_features
import pytest

class TestExtractStylometricFeatures:
    """Comprehensive test suite for stylometric feature extraction."""

    def test_empty_string_returns_zeros(self):
        """Empty input must return a dictionary with all values set to 0.0."""
        features = extract_stylometric_features("")
        assert all(v == 0.0 for v in features.values())

    def test_none_input_returns_zeros(self):
        """None input must return a dictionary with all values set to 0.0."""
        features = extract_stylometric_features(None)
        assert all(v == 0.0 for v in features.values())

    def test_whitespace_only_returns_zeros(self):
        """Whitespace-only input must return a dictionary with all values set to 0.0."""
        features = extract_stylometric_features("   \n\t  ")
        assert all(v == 0.0 for v in features.values())

    def test_returns_dict_with_correct_keys(self):
        """The returned dictionary must contain all 5 expected stylometric keys."""
        features = extract_stylometric_features("This is a test sentence.")
        expected_keys = {
            "type_token_ratio",
            "avg_word_length",
            "avg_sentence_length",
            "sentence_length_variance",
            "yules_k",
        }
        assert set(features.keys()) == expected_keys

    def test_all_values_are_floats(self):
        """All returned metric values must be of type float."""
        features = extract_stylometric_features("Hello world. This is a test.")
        for key, value in features.items():
            assert isinstance(value, float), f"{key} is not a float"

    def test_type_token_ratio_perfect_diversity(self):
        """If all words are unique, TTR should be exactly 1.0."""
        text = "one two three four five"
        features = extract_stylometric_features(text)
        assert features["type_token_ratio"] == 1.0

    def test_type_token_ratio_repetition(self):
        """Repeated words should lower the TTR."""
        text = "the the the the the"
        features = extract_stylometric_features(text)
        assert features["type_token_ratio"] == 0.2  # 1 unique / 5 total

    def test_average_word_length_calculation(self):
        """Verify average word length is calculated correctly."""
        # "cat" (3), "dog" (3), "bird" (4) -> avg = 10/3 = 3.3333
        text = "cat dog bird"
        features = extract_stylometric_features(text)
        assert features["avg_word_length"] == pytest.approx(3.3333, rel=1e-3)

    def test_sentence_length_variance_single_sentence(self):
        """A single sentence should have 0.0 variance."""
        text = "This is a single sentence without terminal punctuation"
        features = extract_stylometric_features(text)
        assert features["sentence_length_variance"] == 0.0

    def test_sentence_length_variance_multiple_sentences(self):
        """Multiple sentences with varying lengths should produce >0 variance."""
        text = "Short. This is a medium length sentence. And here is a significantly longer sentence for testing."
        features = extract_stylometric_features(text)
        assert features["sentence_length_variance"] > 0.0

    def test_yules_k_positive_for_real_text(self):
        """Yule's K should be > 0 for typical text with repeated words."""
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "The dog barked at the fox, but the fox ran away."
        )
        features = extract_stylometric_features(text)
        assert features["yules_k"] > 0.0

    def test_yules_k_zero_for_all_unique_words(self):
        """If all words appear exactly once, Yule's K should be 0.0."""
        # 5 unique words, each appearing once. f_1 = 5, all other f_i = 0
        # sum(f_i * i^2) = 5 * 1^2 = 5. N = 5.
        # K = 10000 * (5/25 - 1/5) = 10000 * (0.2 - 0.2) = 0.0
        text = "one two three four five"
        features = extract_stylometric_features(text)
        assert features["yules_k"] == 0.0

    def test_case_insensitivity_in_word_counting(self):
        """Stylometric features should be case-insensitive for word counting."""
        text_a = "The the THE"
        text_b = "the the the"
        feat_a = extract_stylometric_features(text_a)
        feat_b = extract_stylometric_features(text_b)
        assert feat_a["type_token_ratio"] == feat_b["type_token_ratio"]
        assert feat_a["yules_k"] == feat_b["yules_k"]

    def test_punctuation_does_not_affect_word_count(self):
        """Punctuation should be stripped and not counted as words."""
        text = "Hello, world! How are you?"
        features = extract_stylometric_features(text)
        # Words: hello, world, how, are, you (5 words)
        assert features["avg_sentence_length"] == 5.0

    def test_handles_text_without_punctuation(self):
        """Text without sentence-ending punctuation should be treated as one sentence."""
        text = "this is a long fragment without any periods or exclamation marks"
        features = extract_stylometric_features(text)
        # Should be treated as 1 sentence
        assert features["avg_sentence_length"] == len(text.split())
        assert features["sentence_length_variance"] == 0.0

    @pytest.mark.parametrize(
        "text,expected_ttr_range",
        [
            ("a b c d e", (0.99, 1.01)),  # All unique
            ("a a a a a", (0.19, 0.21)),  # All same
            ("a b a b a", (0.39, 0.41)),  # 2 unique out of 5
        ],
    )
    def test_parametrized_ttr_bounds(self, text, expected_ttr_range):
        """Verify TTR falls within expected mathematical bounds for known patterns."""
        features = extract_stylometric_features(text)
        assert expected_ttr_range[0] <= features["type_token_ratio"] <= expected_ttr_range[1]

    def test_long_academic_text_extraction(self):
        """Verify extraction works on a realistic paragraph of academic text."""
        text = (
            "Stylometry is the linguistic analysis of a person's distinctive writing style. "
            "It has been used for authorship attribution, identifying the authors of disputed "
            "or anonymous texts, since the 19th century. Modern computational stylometry uses "
            "machine learning techniques to analyze large corpora of texts."
        )
        features = extract_stylometric_features(text)
        
        assert features["type_token_ratio"] > 0.5  # Academic text has rich vocabulary
        assert features["avg_word_length"] > 4.0   # Academic words tend to be longer
        assert features["yules_k"] > 0.0           # Some repetition expected
