"""
Comprehensive Unit Tests for Case-Insensitive Search in PDF Highlighting
Issue: #3973
"""

import pytest
from src.utils.search_utils import find_case_insensitive, is_case_insensitive_match


class TestCaseInsensitiveSearch:
    def test_basic_match(self):
        """Should find a match when case is the same."""
        assert find_case_insensitive("Hello World", "World") == [6]

    def test_uppercase_search_term(self):
        """Should find a match when search term is uppercase."""
        assert find_case_insensitive("Hello World", "WORLD") == [6]

    def test_lowercase_search_term(self):
        """Should find a match when search term is lowercase."""
        assert find_case_insensitive("HELLO WORLD", "world") == [6]

    def test_multiple_matches(self):
        """Should find all matches in the text."""
        assert find_case_insensitive("cat dog cat", "cat") == [0, 8]

    def test_no_match(self):
        """Should return an empty list when no match is found."""
        assert find_case_insensitive("Hello World", "xyz") == []


class TestIsCaseInsensitiveMatch:
    def test_basic_match(self):
        """Should return True for a basic match."""
        assert is_case_insensitive_match("Hello World", "World") is True

    def test_uppercase_search(self):
        """Should return True for uppercase search term."""
        assert is_case_insensitive_match("Hello World", "WORLD") is True

    def test_lowercase_search(self):
        """Should return True for lowercase search term."""
        assert is_case_insensitive_match("HELLO WORLD", "world") is True

    def test_no_match(self):
        """Should return False when no match is found."""
        assert is_case_insensitive_match("Hello World", "xyz") is False

    def test_empty_text(self):
        """Should return False for empty text."""
        assert is_case_insensitive_match("", "World") is False

    def test_empty_search_term(self):
        """Should return False for empty search term."""
        assert is_case_insensitive_match("Hello", "") is False


class TestSearchEdgeCases:
    def test_find_empty_text(self):
        """Should return an empty list for empty text."""
        assert find_case_insensitive("", "World") == []

    def test_find_empty_search_term(self):
        """Should return an empty list for empty search term."""
        assert find_case_insensitive("Hello", "") == []

    def test_find_text_smaller_than_search(self):
        """Should return an empty list if text is smaller than search term."""
        assert find_case_insensitive("Hi", "Hello") == []

    def test_find_special_characters(self):
        """Should handle special characters in search."""
        assert find_case_insensitive("Hello @World", "@world") == [6]

    def test_none_input(self):
        """Should handle None input."""
        assert find_case_insensitive(None, "World") == []
        assert is_case_insensitive_match(None, "World") is False