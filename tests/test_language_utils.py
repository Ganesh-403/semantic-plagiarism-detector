"""
Comprehensive Unit Tests for Language Code Validation
Issue: #3985
"""

import pytest
from src.utils.language_utils import validate_language_code


class TestValidLanguageCodes:
    def test_english_code(self):
        assert validate_language_code("en") is True

    def test_french_code(self):
        assert validate_language_code("fr") is True

    def test_spanish_code(self):
        assert validate_language_code("es") is True

    def test_german_code(self):
        assert validate_language_code("de") is True

    def test_japanese_code(self):
        assert validate_language_code("ja") is True

    def test_hindi_code(self):
        assert validate_language_code("hi") is True

    def test_uppercase_code(self):
        assert validate_language_code("EN") is True

    def test_mixed_case_code(self):
        assert validate_language_code("Fr") is True


class TestInvalidLanguageCodes:
    def test_invalid_code(self):
        assert validate_language_code("xx") is False

    def test_full_word(self):
        assert validate_language_code("english") is False

    def test_empty_string(self):
        assert validate_language_code("") is False

    def test_three_letter_code(self):
        assert validate_language_code("eng") is False


class TestEdgeCases:
    def test_none_input(self):
        assert validate_language_code(None) is False

    def test_integer_input(self):
        assert validate_language_code(123) is False

    def test_float_input(self):
        assert validate_language_code(3.14) is False

    def test_list_input(self):
        assert validate_language_code(["en"]) is False

    def test_boolean_input(self):
        assert validate_language_code(True) is False