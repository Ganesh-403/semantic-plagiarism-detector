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
tests/core/test_ai_watermark.py
-------------------------------
Unit tests for AI-Generated Text Watermark Extraction and Verification.
"""

import math

import pytest

from src.core.ai_watermark_extractor import (
    compute_ngram_frequencies,
    extract_token_distribution,
)
from src.core.watermark_statistical_test import (
    compute_p_value,
    compute_z_score,
    verify_watermark_presence,
)


class TestAIWatermarkExtractor:
    """Test suite for watermark feature extraction."""

    def test_extract_token_distribution(self):
        """Verify token distribution metrics are computed correctly."""
        text = "the the the cat"
        metrics = extract_token_distribution(text)
        assert metrics["total_tokens"] == 4
        assert metrics["green_list_count"] == 3  # 'the' is in green list

    def test_compute_ngram_frequencies(self):
        """Verify n-gram frequencies are computed correctly."""
        text = "a b c a b"
        freqs = compute_ngram_frequencies(text, n=2)
        assert freqs["a b"] == 2


class TestWatermarkStatisticalTest:
    """Test suite for statistical hypothesis testing."""

    def test_compute_z_score(self):
        """Verify z-score calculation."""
        z = compute_z_score(0.6, 0.5, 100)
        assert z > 0

    def test_compute_p_value(self):
        """Verify p-value calculation using erf."""
        p = compute_p_value(1.96)
        assert 0.02 < p < 0.03  # Approx 0.025 for z=1.96

    def test_verify_watermark_presence_positive(self):
        """Verify watermarked text is flagged correctly."""
        metrics = {"total_tokens": 100, "green_list_ratio": 0.8}
        result = verify_watermark_presence(metrics, expected_green_ratio=0.5)
        assert result["is_watermarked"] is True
        assert result["p_value"] < 0.05

    def test_verify_watermark_presence_negative(self):
        """Verify non-watermarked text is not flagged."""
        metrics = {"total_tokens": 100, "green_list_ratio": 0.51}
        result = verify_watermark_presence(metrics, expected_green_ratio=0.5)
        assert result["is_watermarked"] is False
