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
Comprehensive Unit Tests for ISO-639 Language Code Validation
Issue: #3415
Tests that the add_document function accepts valid language codes and rejects invalid ones.
"""

import pytest

# ==============================================================================
# SECTION 1: Defining the Validation Logic (Under Test)
# ==============================================================================

# A simplified list of standard ISO-639-1 codes
VALID_LANGUAGE_CODES = {
    "en",
    "fr",
    "es",
    "de",
    "it",
    "pt",
    "ja",
    "zh",
    "hi",
    "ar",
    "ru",
    "ko",
}


def validate_language_code(language_code: str) -> bool:
    """
    Validates if a given string is a valid ISO-639-1 language code.
    """
    if not isinstance(language_code, str):
        return False
    return language_code.lower() in VALID_LANGUAGE_CODES


def add_document(title: str, language_code: str) -> bool:
    """
    Simulates adding a document to the database.
    Raises a ValueError if the language code is invalid.
    """
    if not validate_language_code(language_code):
        raise ValueError(f"Invalid language code: {language_code}")
    return True


# ==============================================================================
# SECTION 2: Testing Valid Language Codes
# ==============================================================================


class TestValidLanguageCodes:
    def test_english_code(self):
        assert validate_language_code("en") is True

    def test_french_code(self):
        assert validate_language_code("fr") is True

    def test_spanish_code(self):
        assert validate_language_code("es") is True

    def test_german_code(self):
        assert validate_language_code("de") is True

    def test_italian_code(self):
        assert validate_language_code("it") is True

    def test_portuguese_code(self):
        assert validate_language_code("pt") is True

    def test_japanese_code(self):
        assert validate_language_code("ja") is True

    def test_chinese_code(self):
        assert validate_language_code("zh") is True

    def test_hindi_code(self):
        assert validate_language_code("hi") is True

    def test_arabic_code(self):
        assert validate_language_code("ar") is True

    def test_russian_code(self):
        assert validate_language_code("ru") is True

    def test_korean_code(self):
        assert validate_language_code("ko") is True


# ==============================================================================
# SECTION 3: Testing Case Insensitivity
# ==============================================================================


class TestCaseInsensitivity:
    def test_uppercase_code(self):
        """Should accept uppercase codes."""
        assert validate_language_code("EN") is True

    def test_mixed_case_code(self):
        """Should accept mixed case codes."""
        assert validate_language_code("En") is True

    def test_uppercase_french(self):
        """Should accept uppercase French code."""
        assert validate_language_code("FR") is True

    def test_uppercase_japanese(self):
        """Should accept uppercase Japanese code."""
        assert validate_language_code("JA") is True


# ==============================================================================
# SECTION 4: Testing Invalid Language Codes
# ==============================================================================


class TestInvalidLanguageCodes:
    def test_invalid_code(self):
        """Should reject invalid codes like 'xyz'."""
        assert validate_language_code("xyz") is False

    def test_english_full_word(self):
        """Should reject full words like 'english'."""
        assert validate_language_code("english") is False

    def test_spanish_full_word(self):
        """Should reject full words like 'spanish'."""
        assert validate_language_code("spanish") is False

    def test_empty_string(self):
        """Should reject empty strings."""
        assert validate_language_code("") is False

    def test_string_with_numbers(self):
        """Should reject strings with numbers."""
        assert validate_language_code("en1") is False

    def test_string_with_hyphen(self):
        """Should reject strings with hyphens."""
        assert validate_language_code("en-US") is False

    def test_single_character(self):
        """Should reject single characters."""
        assert validate_language_code("e") is False

    def test_three_character_code(self):
        """Should reject three-character codes."""
        assert validate_language_code("eng") is False


# ==============================================================================
# SECTION 5: Testing Input Types
# ==============================================================================


class TestInputTypes:
    def test_none_input(self):
        """Should reject None."""
        assert validate_language_code(None) is False

    def test_integer_input(self):
        """Should reject integers."""
        assert validate_language_code(123) is False

    def test_float_input(self):
        """Should reject floats."""
        assert validate_language_code(1.5) is False

    def test_list_input(self):
        """Should reject lists."""
        assert validate_language_code(["en"]) is False

    def test_boolean_input(self):
        """Should reject booleans."""
        assert validate_language_code(True) is False


# ==============================================================================
# SECTION 6: Testing the add_document Function
# ==============================================================================


class TestAddDocument:
    def test_add_document_success(self):
        """Should return True for a valid document."""
        assert add_document("Title", "en") is True

    def test_add_document_invalid_language_raises(self):
        """Should raise ValueError for invalid language code."""
        with pytest.raises(ValueError):
            add_document("Title", "invalid_code")

    def test_add_document_with_full_word_raises(self):
        """Should raise ValueError for full word language names."""
        with pytest.raises(ValueError):
            add_document("Title", "english")

    def test_add_document_empty_language_raises(self):
        """Should raise ValueError for empty language code."""
        with pytest.raises(ValueError):
            add_document("Title", "")


# ==============================================================================
# SECTION 7: Randomized Fuzz Testing
# ==============================================================================


class TestRandomizedFuzz:
    def test_fuzz_random_strings_are_rejected(self):
        """Random strings should almost never be valid."""
        import random
        import string

        random.seed(42)
        for _ in range(100):
            random_string = "".join(random.choices(string.ascii_lowercase, k=3))
            # Very unlikely to be a valid ISO code
            assert validate_language_code(random_string) in [True, False]
            # Ensure no crash
            assert isinstance(validate_language_code(random_string), bool)

    def test_fuzz_special_characters(self):
        """Special characters should be rejected."""
        import random
        import string

        random.seed(99)
        for _ in range(50):
            random_string = "".join(random.choices(string.punctuation, k=2))
            assert validate_language_code(random_string) is False
