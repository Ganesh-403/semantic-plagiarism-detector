"""
Comprehensive Unit Tests for Stop Word Filtering Toggle
Issue: #4017
"""

import pytest
from src.utils.stop_word_utils import filter_stop_words, get_stop_words


class TestDefaultStopWords:
    def test_default_stop_words_exists(self):
        """Should have a default set of stop words."""
        stop_words = get_stop_words()
        assert "the" in stop_words
        assert "and" in stop_words
        assert "is" in stop_words

    def test_default_stop_words_is_set(self):
        """Should return a set."""
        stop_words = get_stop_words()
        assert isinstance(stop_words, set)


class TestFilterStopWordsToggle:
    def test_toggle_off_returns_original(self):
        """Should return the original list if toggle is off (default)."""
        tokens = ["the", "cat", "is", "sleeping"]
        result = filter_stop_words(tokens)
        assert result == tokens

    def test_toggle_on_filters_words(self):
        """Should filter out stop words if toggle is on."""
        tokens = ["the", "cat", "is", "sleeping"]
        result = filter_stop_words(tokens, remove_stop_words=True)
        assert "the" not in result
        assert "is" not in result
        assert "cat" in result
        assert "sleeping" in result

    def test_toggle_on_empty_list(self):
        """Should handle an empty list."""
        result = filter_stop_words([], remove_stop_words=True)
        assert result == []

    def test_toggle_off_empty_list(self):
        """Should handle an empty list when off."""
        result = filter_stop_words([])
        assert result == []


class TestFilterStopWordsEdgeCases:
    def test_case_insensitive_filtering(self):
        """Should filter stop words regardless of case."""
        tokens = ["The", "CAT", "IS", "sleeping"]
        result = filter_stop_words(tokens, remove_stop_words=True)
        assert "The" not in result
        assert "IS" not in result
        assert "CAT" in result
        assert "sleeping" in result

    def test_none_input(self):
        """Should handle None input gracefully."""
        with pytest.raises(TypeError):
            filter_stop_words(None, remove_stop_words=True)

    def test_single_stop_word(self):
        """Should remove a single stop word."""
        result = filter_stop_words(["the"], remove_stop_words=True)
        assert result == []

    def test_no_stop_words(self):
        """Should return all tokens if none are stop words."""
        tokens = ["apple", "banana", "cat"]
        result = filter_stop_words(tokens, remove_stop_words=True)
        assert result == tokens


class TestFilterStopWordsTypes:
    def test_remove_stop_words_type(self):
        """Should accept boolean toggle."""
        with pytest.raises(TypeError):
            filter_stop_words(["the"], remove_stop_words="yes")

    def test_list_input(self):
        """Should accept a list of strings."""
        tokens = ["run", "fast"]
        result = filter_stop_words(tokens, remove_stop_words=True)
        assert result == tokens