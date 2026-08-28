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

# tests/core/test_text_normalization_properties.py
"""
Property-based tests for text normalization and preprocessing functions.
These tests verify that text transformations maintain invariant properties
across all possible inputs.
"""

import re
import string
import unicodedata
from typing import Any, Callable, List

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

# ============== TEXT NORMALIZATION FUNCTIONS ==============
# These should match or import from src/core/
# If they exist elsewhere, import them instead of redefining


def normalize_whitespace(text: str) -> str:
    """Normalize all whitespace to single spaces and strip."""
    return " ".join(text.split())


def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters."""
    return re.sub(r"[^\w\s]", "", text)


def lowercase_text(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_digits(text: str) -> str:
    """Remove all digits."""
    return re.sub(r"\d", "", text)


def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form."""
    return unicodedata.normalize("NFC", text)


def remove_accents(text: str) -> str:
    """Remove accent marks from text."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


def strip_html_tags(text: str) -> str:
    """Remove HTML tags."""
    return re.sub(r"<[^>]+>", "", text)


def normalize_text_pipeline(text: str) -> str:
    """Complete text normalization pipeline."""
    text = strip_html_tags(text)
    text = normalize_unicode(text)
    text = remove_accents(text)
    text = lowercase_text(text)
    text = remove_punctuation(text)
    text = remove_digits(text)
    text = normalize_whitespace(text)
    return text


# ============== PROPERTY-BASED TESTS ==============


class TestWhitespaceNormalization:
    """Property-based tests for whitespace normalization."""

    @given(st.text())
    def test_normalized_whitespace_has_no_multiple_spaces(self, text: str):
        """Property: Result should never contain multiple consecutive spaces."""
        result = normalize_whitespace(text)
        assert "  " not in result

    @given(st.text())
    def test_normalized_whitespace_has_no_tabs_or_newlines(self, text: str):
        """Property: Result should never contain tabs or newlines."""
        result = normalize_whitespace(text)
        assert "\t" not in result
        assert "\n" not in result
        assert "\r" not in result

    @given(st.text())
    def test_normalized_whitespace_is_stripped(self, text: str):
        """Property: Result should be stripped (no leading/trailing whitespace)."""
        result = normalize_whitespace(text)
        assert result == result.strip()

    @given(st.text())
    def test_normalized_whitespace_preserves_meaningful_content(self, text: str):
        """Property: Non-whitespace characters should be preserved."""
        result = normalize_whitespace(text)

        def remove_all_whitespace(s):
            return "".join(s.split())

        assert remove_all_whitespace(result) == remove_all_whitespace(text)

    @given(st.text())
    def test_normalized_whitespace_is_idempotent(self, text: str):
        """Property: Applying twice yields same result as applying once."""
        once = normalize_whitespace(text)
        twice = normalize_whitespace(once)
        assert once == twice


class TestPunctuationRemoval:
    """Property-based tests for punctuation removal."""

    @given(st.text())
    def test_remove_punctuation_contains_only_words_and_spaces(self, text: str):
        """Property: Result should only contain word characters and spaces."""
        result = remove_punctuation(text)
        assert all(c.isalnum() or c.isspace() or c == "_" for c in result)

    @given(st.text())
    def test_remove_punctuation_preserves_alphanumeric(self, text: str):
        """Property: Alphanumeric characters should be preserved."""
        result = remove_punctuation(text)
        original_alnum = [c for c in text if c.isalnum()]
        result_alnum = [c for c in result if c.isalnum()]
        assert original_alnum == result_alnum

    @given(st.text())
    def test_remove_punctuation_removes_all_punctuation(self, text: str):
        """Property: All punctuation characters should be removed."""
        result = remove_punctuation(text)
        punctuation_chars = set(string.punctuation)
        assert not any(c in punctuation_chars for c in result)

    @given(st.text())
    def test_remove_punctuation_is_idempotent(self, text: str):
        """Property: Applying twice yields same result as applying once."""
        once = remove_punctuation(text)
        twice = remove_punctuation(once)
        assert once == twice


class TestLowercasing:
    """Property-based tests for lowercasing."""

    @given(st.text())
    def test_lowercase_text_has_no_uppercase(self, text: str):
        """Property: Result should contain no uppercase letters."""
        result = lowercase_text(text)
        assert not any(c.isupper() for c in result)

    @given(st.text())
    def test_lowercase_text_preserves_non_alphabetic(self, text: str):
        """Property: Non-alphabetic characters should be unchanged."""
        result = lowercase_text(text)
        for original, transformed in zip(text, result):
            if not original.isalpha():
                assert original == transformed

    @given(st.text())
    def test_lowercase_text_is_idempotent(self, text: str):
        """Property: Applying twice yields same result as applying once."""
        once = lowercase_text(text)
        twice = lowercase_text(once)
        assert once == twice

    @given(st.text())
    def test_lowercase_text_preserves_length_for_ascii(self, text: str):
        """Property: Length should be preserved for ASCII text."""
        assume(all(ord(c) < 128 for c in text))
        result = lowercase_text(text)
        assert len(result) == len(text)


class TestDigitRemoval:
    """Property-based tests for digit removal."""

    @given(st.text())
    def test_remove_digits_has_no_digits(self, text: str):
        """Property: Result should contain no digits."""
        result = remove_digits(text)
        assert not any(c.isdigit() for c in result)

    @given(st.text())
    def test_remove_digits_preserves_non_digits(self, text: str):
        """Property: Non-digit characters should be preserved."""
        result = remove_digits(text)
        original_non_digits = [c for c in text if not c.isdigit()]
        result_chars = list(result)
        assert original_non_digits == result_chars

    @given(st.text())
    def test_remove_digits_is_idempotent(self, text: str):
        """Property: Applying twice yields same result as applying once."""
        once = remove_digits(text)
        twice = remove_digits(once)
        assert once == twice


class TestUnicodeNormalization:
    """Property-based tests for unicode normalization."""

    @given(st.text())
    def test_unicode_normalization_is_normalized(self, text: str):
        """Property: Result should be in NFC form."""
        result = normalize_unicode(text)
        assert unicodedata.is_normalized("NFC", result)

    @given(st.text())
    def test_unicode_normalization_is_idempotent(self, text: str):
        """Property: Applying twice yields same result as applying once."""
        once = normalize_unicode(text)
        twice = normalize_unicode(once)
        assert once == twice


class TestAccentRemoval:
    """Property-based tests for accent removal."""

    @given(st.text())
    def test_remove_accents_removes_combining_characters(self, text: str):
        """Property: Result should have no combining characters."""
        result = remove_accents(text)
        assert not any(unicodedata.combining(c) for c in result)

    @given(st.text())
    def test_remove_accents_is_idempotent(self, text: str):
        """Property: Applying twice yields same result as applying once."""
        once = remove_accents(text)
        twice = remove_accents(once)
        assert once == twice


class TestHTMLStripping:
    """Property-based tests for HTML tag stripping."""

    @given(st.text(alphabet=st.characters(whitelist_categories=("L", "N", "P", "S"))))
    def test_strip_html_tags_removes_tags(self, text: str):
        """Property: Should remove all HTML tags."""
        html_text = f"<div>{text}</div><p>{text}</p>"
        result = strip_html_tags(html_text)
        assert "<" not in result or ">" not in result

    @given(st.text())
    def test_strip_html_tags_preserves_text_content(self, text: str):
        """Property: Text content outside tags should be preserved."""
        plain_text = "Hello world"
        html_text = f"<div>{plain_text}</div>"
        result = strip_html_tags(html_text)
        assert plain_text in result


class TestCompleteNormalizationPipeline:
    """Property-based tests for the complete normalization pipeline."""

    @given(st.text())
    def test_pipeline_removes_all_special_characters(self, text: str):
        """Property: Result should only contain alphanumeric and spaces."""
        result = normalize_text_pipeline(text)
        assert all(c.isalnum() or c.isspace() for c in result)

    @given(st.text())
    def test_pipeline_has_no_whitespace_issues(self, text: str):
        """Property: Result should have normalized whitespace."""
        result = normalize_text_pipeline(text)
        assert "  " not in result
        assert result == result.strip()

    @given(st.text())
    def test_pipeline_is_idempotent(self, text: str):
        """Property: Pipeline should be idempotent."""
        once = normalize_text_pipeline(text)
        twice = normalize_text_pipeline(once)
        assert once == twice

    @given(st.text())
    def test_pipeline_has_no_accents(self, text: str):
        """Property: Result should have no accents."""
        result = normalize_text_pipeline(text)
        assert not any(unicodedata.combining(c) for c in result)

    @given(st.text())
    def test_pipeline_has_no_uppercase(self, text: str):
        """Property: Result should be lowercase."""
        result = normalize_text_pipeline(text)
        assert not any(c.isupper() for c in result)

    @given(st.text())
    def test_pipeline_has_no_digits(self, text: str):
        """Property: Result should have no digits."""
        result = normalize_text_pipeline(text)
        assert not any(c.isdigit() for c in result)


class TestTextNormalizationComposition:
    """Test properties of composed normalization functions."""

    @given(st.text())
    def test_normalization_preserves_alphabetic_order(self, text: str):
        """Property: Alphabetic characters should maintain relative order."""
        result = normalize_text_pipeline(text)
        alpha_original = [c for c in text if c.isalpha()]
        alpha_result = [c for c in result if c.isalpha()]
        alpha_original_lower = [c.lower() for c in alpha_original]
        alpha_result_lower = [c.lower() for c in alpha_result]
        assert alpha_original_lower == alpha_result_lower

    @given(st.text())
    def test_normalization_is_deterministic(self, text: str):
        """Property: Multiple calls should produce identical results."""
        result1 = normalize_text_pipeline(text)
        result2 = normalize_text_pipeline(text)
        assert result1 == result2

    @given(st.text())
    def test_normalization_handles_empty_string(self, text: str):
        """Property: Empty input should yield empty output."""
        result = normalize_text_pipeline("")
        assert result == ""

    @given(st.text())
    def test_normalization_preserves_meaningful_words(self, text: str):
        """Property: Core words should remain recognizable."""
        # Test with a known sentence
        test_text = "Hello, World! 123"
        result = normalize_text_pipeline(test_text)
        # Should preserve "hello" and "world" but remove punctuation and digits
        assert "hello" in result
        assert "world" in result
        assert "123" not in result
