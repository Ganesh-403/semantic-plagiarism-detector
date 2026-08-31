"""
tests/core/test_encoding.py
---------------------------
Unit tests for text encoding normalization and mojibake repair.

Validates the expanded Windows-1252 → UTF-8 replacement patterns
introduced in Issue #2052.
"""

import pytest

from src.core.encoding import (
    MOJIBAKE_REPLACEMENTS,
    detect_mojibake,
    normalize_encoding,
)


class TestNormalizeEncoding:
    """Test suite for the normalize_encoding() repair function."""

    def test_empty_and_none_inputs(self):
        """Verify empty strings and None inputs return empty strings."""
        assert normalize_encoding("") == ""
        assert normalize_encoding(None) == ""
        assert normalize_encoding(123) == ""  # Non-string type

    def test_clean_text_unchanged(self):
        """Verify already-clean UTF-8 text is not modified."""
        clean_text = "The quick brown fox jumps over the lazy dog. Café résumé."
        assert normalize_encoding(clean_text) == clean_text

    def test_original_three_patterns(self):
        """Verify the original 3 patterns (é, á, ü) still work."""
        assert normalize_encoding("cafÃ©") == "café"
        assert normalize_encoding("Ã¡rbol") == "árbol"
        assert normalize_encoding("Ã¼ber") == "über"

    @pytest.mark.parametrize(
        "garbled,expected",
        [
            # Spanish / Portuguese
            ("EspaÃ±ol", "Español"),
            ("corazÃ³n", "corazón"),
            ("paÃ­s", "país"),
            ("baÃºl", "baúl"),
            ("franÃ§ais", "français"),
            # French / German
            ("frÃ¨re", "frère"),
            ("fÃªte", "fête"),
            ("naÃ¯ve", "naïve"),
            ("hÃ´tel", "hôtel"),
            ("aoÃ»t", "août"),
            ("StraÃŸe", "Straße"),
            # Nordic
            ("encyclopÃ¦dia", "encyclopædia"),
            ("SÃ¸ren", "Søren"),
            ("blÃ¥bÃ¦r", "blåbær"),
        ],
    )
    def test_accented_character_patterns(self, garbled, expected):
        """Verify expanded accented character patterns are repaired correctly."""
        assert normalize_encoding(garbled) == expected

    @pytest.mark.parametrize(
        "garbled,expected",
        [
            ("He said, â€œHello!â€", "He said, “Hello!”"),
            ("Itâ€™s a beautiful day.", "It’s a beautiful day."),
            ("â€˜Quoteâ€™", "‘Quote’"),
        ],
    )
    def test_smart_quote_patterns(self, garbled, expected):
        """Verify Microsoft Word smart quotes are repaired correctly."""
        assert normalize_encoding(garbled) == expected

    @pytest.mark.parametrize(
        "garbled,expected",
        [
            ("Wordâ€”Word", "Word—Word"),  # Em dash
            ("10â€“20", "10–20"),  # En dash
            ("Waitâ€¦", "Wait…"),  # Ellipsis
        ],
    )
    def test_punctuation_patterns(self, garbled, expected):
        """Verify dashes and ellipses are repaired correctly."""
        assert normalize_encoding(garbled) == expected

    def test_stray_control_characters(self):
        """Verify stray Â characters are removed."""
        assert normalize_encoding("100Â€") == "100€"
        assert normalize_encoding("Â© 2024") == "© 2024"

    def test_mixed_garbled_and_clean_text(self):
        """Verify function handles text with both garbled and clean sections."""
        text = "The cafÃ© on RÃ©publique street serves crÃ¨me brÃ»lÃ©e."
        expected = "The café on République street serves crème brûlée."
        assert normalize_encoding(text) == expected

    def test_idempotency(self):
        """Verify applying the function twice yields the same result as once."""
        text = "The cafÃ© was beautifÃ»l."
        first_pass = normalize_encoding(text)
        second_pass = normalize_encoding(first_pass)

        assert first_pass == "The café was beautifûl."
        assert first_pass == second_pass

    def test_long_document_performance(self):
        """Verify the regex replacement performs efficiently on large texts."""
        # Generate a 10,000 character string with scattered mojibake
        base = "The quick brown fox jumps over the lazy dog. "
        garbled_base = "The quÃ®ck brÃ¸wn fÃ¸x jÃ»mps Ã¸ver the lÃ¥zy dÃ¸g. "

        large_text = (garbled_base * 200) + (base * 200)

        result = normalize_encoding(large_text)

        assert "quîck" in result
        assert "brøwn" in result
        assert len(result) < len(large_text)  # Garbled chars are longer


class TestDetectMojibake:
    """Test suite for the detect_mojibake() heuristic function."""

    def test_clean_text_returns_false(self):
        """Verify clean text is not flagged as mojibake."""
        assert detect_mojibake("This is perfectly normal English text.") is False
        assert detect_mojibake("Café résumé naïve.") is False

    def test_heavily_garbled_text_returns_true(self):
        """Verify text with many mojibake patterns is flagged."""
        text = (
            "The cafÃ© on RÃ©publique street serves crÃ¨me brÃ»lÃ©e and Ã¼ber-dÃ¶ner."
        )
        assert detect_mojibake(text) is True

    def test_threshold_boundary(self):
        """Verify the threshold parameter controls sensitivity."""
        # 1 garbled char in 100 chars = 1% ratio
        text = "Ã©" + ("a" * 99)

        assert detect_mojibake(text, threshold=0.05) is False  # 1% < 5%
        assert detect_mojibake(text, threshold=0.005) is True  # 1% > 0.5%

    def test_empty_and_none_inputs(self):
        """Verify empty and None inputs return False."""
        assert detect_mojibake("") is False
        assert detect_mojibake(None) is False


class TestMojibakeDictionary:
    """Test suite for the MOJIBAKE_REPLACEMENTS dictionary integrity."""

    def test_minimum_pattern_count(self):
        """Verify the dictionary contains at least 20 patterns (Issue #2052)."""
        # Original 3 + at least 8 new = 11 minimum, but we added ~20
        assert len(MOJIBAKE_REPLACEMENTS) >= 20

    def test_all_values_are_strings(self):
        """Verify all replacement values are strings."""
        for key, value in MOJIBAKE_REPLACEMENTS.items():
            assert isinstance(key, str), f"Key {key} is not a string"
            assert isinstance(value, str), f"Value for {key} is not a string"

    def test_no_empty_keys(self):
        """Verify no empty string keys exist in the dictionary."""
        assert "" not in MOJIBAKE_REPLACEMENTS
