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
test_cross_encoder_rescoring.py
--------------------------------
Comprehensive unit tests for the Cross-Encoder re-ranking stage (#1355).
Tests re-scoring precision, fallback mechanics, cache management, edge cases,
and multi-tuple metadata preservation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.similarity import (
    _CROSS_ENCODER_FAILED_MODELS,
    _CROSS_ENCODER_MODELS,
    _get_cross_encoder,
    clear_cross_encoder_cache,
    get_cross_encoder_info,
    rerank_candidates_with_cross_encoder,
)


class DummyModel:
    """Mock CrossEncoder model for testing inference."""

    def __init__(self, scores=None):
        self.scores = scores or [0.5, 0.9, 0.1]

    def predict(self, sentence_pairs, batch_size=32):
        if len(sentence_pairs) <= len(self.scores):
            return np.array(self.scores[: len(sentence_pairs)])
        return np.array(self.scores + [0.5] * (len(sentence_pairs) - len(self.scores)))


def setup_function():
    clear_cross_encoder_cache()


def teardown_function():
    clear_cross_encoder_cache()


def test_clear_cross_encoder_cache():
    """Verify clearing the model cache resets loaded and failed sets."""
    _CROSS_ENCODER_MODELS["dummy_model"] = DummyModel()
    _CROSS_ENCODER_FAILED_MODELS.add("broken_model")

    info1 = get_cross_encoder_info("dummy_model")
    assert info1["is_loaded"] is True

    clear_cross_encoder_cache()

    info2 = get_cross_encoder_info("dummy_model")
    assert info2["is_loaded"] is False
    assert info2["is_failed"] is False


def test_get_cross_encoder_success():
    """Verify _get_cross_encoder loads and caches the model when available."""
    clear_cross_encoder_cache()

    mock_cross_encoder_cls = MagicMock()
    fake_instance = DummyModel()
    mock_cross_encoder_cls.return_value = fake_instance

    with patch.dict(
        "sys.modules",
        {"sentence_transformers": MagicMock(CrossEncoder=mock_cross_encoder_cls)},
    ):
        model = _get_cross_encoder("custom-cross-encoder")
        assert model is fake_instance
        assert get_cross_encoder_info("custom-cross-encoder")["is_loaded"] is True


def test_get_cross_encoder_failure_logging_and_caching():
    """Verify model loading failures are logged and recorded in failure cache."""
    clear_cross_encoder_cache()

    bad_module = MagicMock()
    bad_module.CrossEncoder.side_effect = RuntimeError("Failed to load model weights")

    with patch.dict("sys.modules", {"sentence_transformers": bad_module}):
        model = _get_cross_encoder("invalid-model-name")
        assert model is None
        assert get_cross_encoder_info("invalid-model-name")["is_failed"] is True

    # Second call should instantly return None from failure cache without retrying
    model_again = _get_cross_encoder("invalid-model-name")
    assert model_again is None


def test_rerank_candidates_with_cross_encoder_default_model_name():
    """Verify default model name parameter works as expected."""
    clear_cross_encoder_cache()

    pairs = [("Sentence 1", "Sentence 2", 0.70)]

    with patch(
        "src.core.similarity._get_cross_encoder", return_value=DummyModel([2.0])
    ):
        rescored = rerank_candidates_with_cross_encoder(pairs)
        assert len(rescored) == 1
        assert rescored[0][0] == "Sentence 1"
        assert rescored[0][1] == "Sentence 2"
        # Sigmoid of 2.0 is ~0.8808
        assert rescored[0][2] == pytest.approx(0.8808, abs=1e-3)


def test_rerank_candidates_with_cross_encoder_without_sigmoid():
    """Verify apply_sigmoid=False clips raw scores directly to [0.0, 1.0]."""
    clear_cross_encoder_cache()

    pairs = [("Text 1", "Text 2", 0.50)]
    mock_model = DummyModel([0.75])

    _CROSS_ENCODER_MODELS["test-no-sigmoid"] = mock_model

    rescored = rerank_candidates_with_cross_encoder(
        pairs, model_name="test-no-sigmoid", apply_sigmoid=False
    )
    assert rescored[0][2] == 0.75


def test_rerank_candidates_with_cross_encoder_preserves_additional_tuple_elements():
    """Verify extra elements in tuples (e.g. metadata, IDs) are preserved after re-ranking."""
    clear_cross_encoder_cache()

    pairs = [
        ("Doc A", "Doc B", 0.60, "pair_id_101", {"severity": "High"}),
        ("Doc C", "Doc D", 0.90, "pair_id_102", {"severity": "Medium"}),
    ]

    mock_model = DummyModel(
        [3.0, -1.0]
    )  # Pair 0 gets high score, Pair 1 gets low score
    _CROSS_ENCODER_MODELS["test-meta"] = mock_model

    rescored = rerank_candidates_with_cross_encoder(pairs, model_name="test-meta")

    assert len(rescored) == 2
    # Pair 0 should now be ranked #1
    top_pair = rescored[0]
    assert top_pair[0] == "Doc A"
    assert top_pair[1] == "Doc B"
    assert top_pair[3] == "pair_id_101"
    assert top_pair[4] == {"severity": "High"}


def test_rerank_candidates_with_cross_encoder_two_element_tuples():
    """Verify pairs of length 2 (text_a, text_b) are converted to (text_a, text_b, score)."""
    clear_cross_encoder_cache()

    pairs = [("Text Alpha", "Text Beta")]
    mock_model = DummyModel([1.0])
    _CROSS_ENCODER_MODELS["test-2elem"] = mock_model

    rescored = rerank_candidates_with_cross_encoder(pairs, model_name="test-2elem")

    assert len(rescored) == 1
    assert len(rescored[0]) == 3
    assert rescored[0][0] == "Text Alpha"
    assert rescored[0][1] == "Text Beta"
    assert isinstance(rescored[0][2], float)


def test_rerank_candidates_with_cross_encoder_prediction_exception_fallback():
    """Verify prediction runtime exceptions fall back gracefully to bi-encoder pairs."""
    clear_cross_encoder_cache()

    pairs = [("Text 1", "Text 2", 0.82)]

    class ErrorModel:
        def predict(self, sentence_pairs, batch_size=32):
            raise RuntimeError("CUDA out of memory during cross-encoder evaluation")

    _CROSS_ENCODER_MODELS["test-error"] = ErrorModel()

    rescored = rerank_candidates_with_cross_encoder(pairs, model_name="test-error")

    # Should safely return original pairs
    assert rescored == pairs


def test_rerank_candidates_with_cross_encoder_malformed_pairs():
    """Verify non-tuple or single-element items in pairs list are handled safely."""
    clear_cross_encoder_cache()

    invalid_pairs = ["invalid_str", 123, None, ("SingleElement",)]

    mock_model = DummyModel()
    _CROSS_ENCODER_MODELS["test-invalid"] = mock_model

    rescored = rerank_candidates_with_cross_encoder(
        invalid_pairs, model_name="test-invalid"
    )

    assert rescored == invalid_pairs
