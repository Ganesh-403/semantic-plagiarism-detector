"""
Comprehensive Unit Tests for get_unique_word_ratio
Issue: #3710
Tests ratio calculation, empty texts, repetitive texts, and edge cases.
"""

import pytest


# ==============================================================================
# SECTION 1: Defining the Function Under Test
# ==============================================================================

def get_unique_word_ratio(text: str) -> float:
    """
    Calculates the ratio of unique words to total words in a text.
    """
    if not isinstance(text, str):
        return 0.0
    
    # Normalize: lowercase and split by whitespace
    words = text.lower().split()
    
    if not words:
        return 0.0
    
    unique_words = set(words)
    return len(unique_words) / len(words)


# ==============================================================================
# SECTION 2: Testing Basic Functionality
# ==============================================================================

class TestBasicRatio:
    def test_standard_sentence(self):
        """Should calculate ratio for a normal sentence."""
        text = "the quick brown fox jumps over the lazy dog"
        ratio = get_unique_word_ratio(text)
        assert ratio == 1.0  # All words are unique

    def test_repetitive_text(self):
        """Should return a low ratio for repetitive text."""
        text = "hello hello hello hello"
        ratio = get_unique_word_ratio(text)
        assert ratio == 0.25  # 1 unique word out of 4 total words

    def test_partial_repetition(self):
        """Should handle partially repetitive text."""
        text = "cat dog cat bird"
        ratio = get_unique_word_ratio(text)
        assert ratio == 0.75  # 3 unique words out of 4 total words

    def test_two_unique_words(self):
        """Should handle exactly two unique words."""
        text = "yes no yes no"
        ratio = get_unique_word_ratio(text)
        assert ratio == 0.5


# ==============================================================================
# SECTION 3: Testing Empty Texts
# ==============================================================================

class TestEmptyTexts:
    def test_empty_string(self):
        """Should return 0.0 for an empty string."""
        assert get_unique_word_ratio("") == 0.0

    def test_whitespace_only_string(self):
        """Should return 0.0 for whitespace only."""
        assert get_unique_word_ratio("   ") == 0.0

    def test_tab_and_newline_only(self):
        """Should return 0.0 for tabs and newlines."""
        assert get_unique_word_ratio("\n\t") == 0.0

    def test_single_space(self):
        """Should return 0.0 for a single space."""
        assert get_unique_word_ratio(" ") == 0.0


# ==============================================================================
# SECTION 4: Testing Repetitive Texts
# ==============================================================================

class TestRepetitiveTexts:
    def test_single_word_repeated_many_times(self):
        """Should handle a single word repeated 100 times."""
        text = "word " * 100
        ratio = get_unique_word_ratio(text)
        assert ratio == 0.01  # 1 unique word out of 100 total words

    def test_two_words_alternating(self):
        """Should handle two words alternating."""
        text = "ab cd ab cd ab cd"
        ratio = get_unique_word_ratio(text)
        assert ratio == 1/3  # 2 unique words out of 6 total words

    def test_case_insensitive_repetition(self):
        """Should treat 'Hello' and 'hello' as the same word."""
        text = "Hello hello HELLO"
        ratio = get_unique_word_ratio(text)
        assert ratio == 1/3

    def test_repeated_text_with_punctuation(self):
        """Should handle repeated text with punctuation (split by whitespace only)."""
        text = "test! test! test!"
        ratio = get_unique_word_ratio(text)
        assert ratio == 1/3  # Punctuation is not stripped, so 'test!' is unique


# ==============================================================================
# SECTION 5: Testing Complex Texts
# ==============================================================================

class TestComplexTexts:
    def test_text_with_numbers(self):
        """Should handle texts with numbers."""
        text = "version 1 version 2 version 3"
        ratio = get_unique_word_ratio(text)
        assert ratio == 1.0

    def test_text_with_mixed_case(self):
        """Should handle mixed case text."""
        text = "The QUICK brown Fox"
        ratio = get_unique_word_ratio(text)
        assert ratio == 1.0  # All 4 words are unique after lowercasing

    def test_text_with_special_characters(self):
        """Should handle special characters as part of words."""
        text = "hello@world hello@world"
        ratio = get_unique_word_ratio(text)
        assert ratio == 0.5


# ==============================================================================
# SECTION 6: Testing Input Types and Edge Cases
# ==============================================================================

class TestInputTypes:
    def test_none_input(self):
        """Should return 0.0 for None input."""
        assert get_unique_word_ratio(None) == 0.0

    def test_integer_input(self):
        """Should return 0.0 for integer input."""
        assert get_unique_word_ratio(12345) == 0.0

    def test_list_input(self):
        """Should return 0.0 for list input."""
        assert get_unique_word_ratio(["a", "b"]) == 0.0

    def test_float_input(self):
        """Should return 0.0 for float input."""
        assert get_unique_word_ratio(3.14) == 0.0

    def test_boolean_input(self):
        """Should return 0.0 for boolean input."""
        assert get_unique_word_ratio(True) == 0.0


# ==============================================================================
# SECTION 7: Testing Ratio Bounds
# ==============================================================================

class TestRatioBounds:
    def test_ratio_is_never_greater_than_one(self):
        """Ratio should never exceed 1.0."""
        texts = [
            "hello world",
            "a b c d e",
            "the the the",
        ]
        for text in texts:
            ratio = get_unique_word_ratio(text)
            assert ratio <= 1.0

    def test_ratio_is_never_negative(self):
        """Ratio should never be negative."""
        texts = [
            "hello world",
            "",
            None,
        ]
        for text in texts:
            ratio = get_unique_word_ratio(text)
            assert ratio >= 0.0

    def test_ratio_is_float(self):
        """Ratio should always be a float."""
        assert isinstance(get_unique_word_ratio("hello world"), float)
        assert isinstance(get_unique_word_ratio(""), float)
        assert isinstance(get_unique_word_ratio(None), float)