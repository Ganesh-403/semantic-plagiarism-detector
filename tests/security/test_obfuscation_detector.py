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
tests/security/test_obfuscation_detector.py
-------------------------------------------
Comprehensive unit tests for the adversarial obfuscation detection module.

Verifies detection accuracy for zero-width spaces, Cyrillic homoglyphs,
and control characters, as well as the scoring algorithm and false-positive rates.
"""

import pytest

from src.core.adversarial_analysis import (
    analyze_document_for_obfuscation,
    strip_obfuscation_chars,
)
from src.security.obfuscation_detector import (
    ObfuscationReport,
    analyze_text_obfuscation,
    calculate_obfuscation_score,
    detect_control_chars,
    detect_homoglyphs,
    detect_zero_width_chars,
)


class TestZeroWidthDetection:
    """Test suite for zero-width and invisible character detection."""

    def test_detects_zero_width_space(self):
        """Verify U+200B (Zero Width Space) is detected."""
        text = "Hello\u200bWorld"
        count, indices = detect_zero_width_chars(text)
        assert count == 1
        assert indices == [5]

    def test_detects_zero_width_joiner(self):
        """Verify U+200D (Zero Width Joiner) is detected."""
        text = "A\u200dB"
        count, indices = detect_zero_width_chars(text)
        assert count == 1

    def test_detects_bom_in_text(self):
        """Verify U+FEFF (BOM) in the middle of text is detected."""
        text = "Test\ufeffString"
        count, indices = detect_zero_width_chars(text)
        assert count == 1
        assert indices == [4]

    def test_clean_text_returns_zero(self):
        """Verify standard ASCII text returns 0 count."""
        text = "This is a completely normal sentence."
        count, indices = detect_zero_width_chars(text)
        assert count == 0
        assert indices == []

    def test_multiple_zero_width_chars(self):
        """Verify multiple invisible chars are all counted and indexed."""
        text = "\u200bHello\u200cWorld\u200d"
        count, indices = detect_zero_width_chars(text)
        assert count == 3
        assert indices == [0, 6, 12]


class TestHomoglyphDetection:
    """Test suite for Cyrillic homoglyph detection."""

    def test_detects_cyrillic_a(self):
        """Verify Cyrillic 'а' (U+0430) is detected as a homoglyph for 'a'."""
        text = "cаt"  # The 'a' is Cyrillic
        count, indices = detect_homoglyphs(text)
        assert count == 1
        assert indices == [1]

    def test_detects_cyrillic_o(self):
        """Verify Cyrillic 'о' (U+043E) is detected."""
        text = "dоg"  # The 'o' is Cyrillic
        count, indices = detect_homoglyphs(text)
        assert count == 1

    def test_mixed_latin_and_cyrillic(self):
        """Verify only the Cyrillic chars are flagged in a mixed string."""
        text = "hellо wоrld"  # Both 'o's are Cyrillic
        count, indices = detect_homoglyphs(text)
        assert count == 2
        assert indices == [4, 7]

    def test_pure_latin_returns_zero(self):
        """Verify pure Latin text returns 0 count."""
        text = "The quick brown fox"
        count, indices = detect_homoglyphs(text)
        assert count == 0


class TestObfuscationScoring:
    """Test suite for the obfuscation scoring algorithm."""

    def test_clean_text_score_is_zero(self):
        """Verify clean text produces an obfuscation score of 0.0."""
        report = ObfuscationReport(total_characters=100)
        score = calculate_obfuscation_score(report)
        assert score == 0.0

    def test_high_density_zero_width_scores_high(self):
        """Verify high density of zero-width chars produces a high score."""
        report = ObfuscationReport(total_characters=50, zero_width_count=10)
        score = calculate_obfuscation_score(report)
        assert score > 0.5

    def test_absolute_threshold_penalty(self):
        """Verify >10 zero-width chars triggers the absolute penalty."""
        report = ObfuscationReport(
            total_characters=1000,  # Low density
            zero_width_count=15,  # But high absolute count
        )
        score = calculate_obfuscation_score(report)
        assert score >= 0.3  # Absolute penalty is 0.3

    def test_empty_text_score_is_zero(self):
        """Verify empty text doesn't cause division by zero."""
        report = ObfuscationReport(total_characters=0)
        score = calculate_obfuscation_score(report)
        assert score == 0.0


class TestAdversarialAnalysisIntegration:
    """Test suite for the core integration layer."""

    def test_analyze_document_flags_suspicious(self):
        """Verify analyze_document_for_obfuscation correctly flags suspicious text."""
        # Inject 20 zero-width spaces
        text = "Normal text " + ("\u200b" * 20) + " more text."
        result = analyze_document_for_obfuscation(text, "doc_123")

        assert result["is_suspicious"] is True
        assert result["obfuscation_score"] > 0.15
        assert result["report"]["zero_width_count"] == 20

    def test_strict_mode_raises_value_error(self):
        """Verify strict_mode raises ValueError on suspicious documents."""
        text = "Test\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b\u200b"

        with pytest.raises(ValueError, match="rejected in strict mode"):
            analyze_document_for_obfuscation(
                text, "doc_123", strict_mode=True, threshold=0.1
            )

    def test_strip_obfuscation_chars_cleans_text(self):
        """Verify strip_obfuscation_chars removes invisible chars and fixes homoglyphs."""
        text = "Hеllo\u200bWоrld"  # Cyrillic е, о, and ZW space
        cleaned = strip_obfuscation_chars(text)

        assert "\u200b" not in cleaned
        assert "е" not in cleaned  # Cyrillic e removed
        assert "о" not in cleaned  # Cyrillic o removed
        assert cleaned == "HelloWorld"
