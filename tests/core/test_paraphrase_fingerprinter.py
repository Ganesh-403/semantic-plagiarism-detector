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
tests/core/test_paraphrase_fingerprinter.py
-------------------------------------------
Unit tests for the Paraphrase Tool Fingerprinting and Attribution engine.
"""

import pytest

from src.core.paraphrase_fingerprinter import (
    attribute_paraphrase_tool,
    compute_sentence_length_variance,
    compute_synonym_entropy,
    compute_transition_anomaly,
    extract_paraphrase_fingerprint,
)


class TestFeatureExtraction:
    """Test suite for statistical feature extraction."""

    def test_synonym_entropy_uniform(self):
        """Verify entropy is high for uniformly distributed words."""
        text = "apple banana cherry date elderberry fig grape honeydew"
        entropy = compute_synonym_entropy(text)
        assert entropy > 0.8  # High entropy for unique words

    def test_synonym_entropy_repetitive(self):
        """Verify entropy is low for highly repetitive text."""
        text = "the the the the the the"
        entropy = compute_synonym_entropy(text)
        assert entropy == 0.0

    def test_sentence_length_variance_uniform(self):
        """Verify variance is low for uniform sentence lengths."""
        text = "One two. Three four. Five six. Seven eight."
        variance = compute_sentence_length_variance(text)
        assert variance < 1.0  # All sentences are 2 words long

    def test_sentence_length_variance_bursty(self):
        """Verify variance is high for bursty human-like writing."""
        text = "Short. This is a much longer sentence that contains many words. Mid."
        variance = compute_sentence_length_variance(text)
        assert variance > 10.0

    def test_transition_anomaly_passive_voice(self):
        """Verify anomaly score increases with passive voice indicators."""
        text = "The ball was kicked by the boy. The cake was eaten by the dog."
        anomaly = compute_transition_anomaly(text)
        assert anomaly > 0.0


class TestToolAttribution:
    """Test suite for tool attribution logic."""

    def test_extract_fingerprint_empty(self):
        """Verify empty text returns zeroed fingerprint."""
        fp = extract_paraphrase_fingerprint("")
        assert fp["synonym_entropy"] == 0.0
        assert fp["sentence_length_variance"] == 0.0

    def test_attribute_tool_returns_scores(self):
        """Verify attribution returns scores for known tools."""
        fp = {
            "synonym_entropy": 0.65,
            "sentence_length_variance": 5.0,
            "transition_anomaly": 0.80,
        }
        result = attribute_paraphrase_tool(fp)

        assert "attributed_tool" in result
        assert "confidence" in result
        assert "scores" in result
        assert "quillbot" in result["scores"]
        assert "spinbot" in result["scores"]

    def test_attribute_tool_confidence_range(self):
        """Verify confidence score is between 0 and 100."""
        fp = {"synonym_entropy": 0.5, "transition_anomaly": 0.5}
        result = attribute_paraphrase_tool(fp)
        assert 0.0 <= result["confidence"] <= 100.0
