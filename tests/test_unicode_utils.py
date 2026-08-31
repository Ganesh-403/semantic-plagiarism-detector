"""
Comprehensive Unit Tests for Unicode Zero-Width Space Handling
Issue: #4004
"""

import pytest
from src.utils.unicode_utils import remove_zero_width_spaces, has_zero_width_spaces


class TestRemoveZeroWidthSpaces:
    def test_removes_zero_width_space(self):
        """Should remove U+200B zero-width spaces."""
        text = "Hello\u200bWorld"
        assert remove_zero_width_spaces(text) == "HelloWorld"

    def test_removes_zero_width_non_joiner(self):
        """Should remove U+200C zero-width non-joiners."""
        text = "Hello\u200cWorld"
        assert remove_zero_width_spaces(text) == "HelloWorld"

    def test_removes_multiple_zero_width_spaces(self):
        """Should remove multiple zero-width spaces."""
        text = "A\u200bB\u200bC\u200bD"
        assert remove_zero_width_spaces(text) == "ABCD"

    def test_does_not_remove_normal_text(self):
        """Should preserve normal text without zero-width spaces."""
        text = "Hello World"
        assert remove_zero_width_spaces(text) == "Hello World"

    def test_handles_empty_string(self):
        """Should handle empty strings."""
        assert remove_zero_width_spaces("") == ""

    def test_handles_none_input(self):
        """Should handle None input."""
        assert remove_zero_width_spaces(None) == ""


class TestHasZeroWidthSpaces:
    def test_detects_zero_width_space(self):
        """Should detect U+200B zero-width spaces."""
        assert has_zero_width_spaces("Hello\u200bWorld") is True

    def test_detects_zero_width_non_joiner(self):
        """Should detect U+200C zero-width non-joiners."""
        assert has_zero_width_spaces("Hello\u200cWorld") is True

    def test_returns_false_for_normal_text(self):
        """Should return false for normal text."""
        assert has_zero_width_spaces("Hello World") is False

    def test_returns_false_for_empty_string(self):
        """Should return false for empty strings."""
        assert has_zero_width_spaces("") is False

    def test_handles_none_input(self):
        """Should handle None input."""
        assert has_zero_width_spaces(None) is False


class TestEdgeCases:
    def test_mixed_zero_width_characters(self):
        """Should handle mixed zero-width characters."""
        text = "Hello\u200b\u200cWorld"
        assert remove_zero_width_spaces(text) == "HelloWorld"

    def test_text_with_only_zero_width_spaces(self):
        """Should handle text with only zero-width spaces."""
        text = "\u200b\u200b\u200b"
        assert remove_zero_width_spaces(text) == ""

    def test_text_with_regular_unicode(self):
        """Should handle regular unicode characters."""
        text = "Café"
        assert remove_zero_width_spaces(text) == "Café"