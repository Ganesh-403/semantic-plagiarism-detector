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
tests/utils/test_badge_generator_colors.py
------------------------------------------
Comprehensive unit tests for the badge generator color validation logic.

Verifies that standard hex codes, CSS named colors, and invalid inputs
are handled correctly with appropriate fallbacks (Issue #2898).
"""

import logging

from src.utils.badge_generator import (
    CSS_NAMED_COLORS,
    DEFAULT_BADGE_COLOR,
    generate_svg_badge,
    validate_hex_color,
)


class TestValidateHexColorStandard:
    """Test suite for standard hex code validation."""

    def test_valid_6_digit_hex(self):
        """Verify standard 6-digit hex codes are accepted."""
        assert validate_hex_color("#ff0000") == "#ff0000"
        assert validate_hex_color("#00FF00") == "#00ff00"
        assert validate_hex_color("#123abc") == "#123abc"

    def test_valid_3_digit_hex(self):
        """Verify 3-digit hex codes are accepted as-is."""
        assert validate_hex_color("#f00") == "#f00"
        assert validate_hex_color("#0F0") == "#0f0"
        assert validate_hex_color("#123") == "#123"

    def test_valid_8_digit_hex_with_alpha(self):
        """Verify 8-digit hex codes (with alpha channel) are accepted."""
        assert validate_hex_color("#ff000080") == "#ff000080"

    def test_valid_4_digit_hex_with_alpha(self):
        """Verify 4-digit hex codes are accepted as-is."""
        assert validate_hex_color("#f008") == "#f008"

    def test_whitespace_stripped(self):
        """Verify leading/trailing whitespace is stripped before validation."""
        assert validate_hex_color("  #ff0000  ") == "#ff0000"
        assert validate_hex_color("\t#00ff00\n") == "#00ff00"


class TestValidateHexColorNamed:
    """Test suite for CSS named color fallback (Issue #2898)."""

    def test_basic_named_colors(self):
        """Verify basic CSS named colors are resolved to hex."""
        assert validate_hex_color("red") == "#ff0000"
        assert validate_hex_color("green") == "#008000"
        assert validate_hex_color("blue") == "#0000ff"
        assert validate_hex_color("white") == "#ffffff"
        assert validate_hex_color("black") == "#000000"

    def test_case_insensitive_named_colors(self):
        """Verify named color matching is case-insensitive."""
        assert validate_hex_color("RED") == "#ff0000"
        assert validate_hex_color("Red") == "#ff0000"
        assert validate_hex_color("DaRkSlAtEgRaY") == "#2f4f4f"

    def test_transparent_named_color(self):
        """Verify 'transparent' resolves to 8-digit hex with 00 alpha."""
        assert validate_hex_color("transparent") == "#00000000"

    def test_extended_named_colors(self):
        """Verify extended CSS named colors are supported."""
        assert validate_hex_color("coral") == "#ff7f50"
        assert validate_hex_color("tomato") == "#ff6347"
        assert validate_hex_color("skyblue") == "#87ceeb"
        assert validate_hex_color("whitesmoke") == "#f5f5f5"

    def test_all_dictionary_colors_resolve(self):
        """Verify every color in the CSS_NAMED_COLORS dict resolves correctly."""
        for name, expected_hex in CSS_NAMED_COLORS.items():
            assert validate_hex_color(name) == expected_hex


class TestValidateHexColorFallback:
    """Test suite for fallback behavior on invalid inputs."""

    def test_invalid_named_color_falls_back(self):
        """Verify unrecognized named colors fall back to default."""
        assert validate_hex_color("notacolor") == DEFAULT_BADGE_COLOR
        assert validate_hex_color("superred") == DEFAULT_BADGE_COLOR

    def test_invalid_hex_format_falls_back(self):
        """Verify malformed hex codes fall back to default."""
        assert validate_hex_color("#gg0000") == DEFAULT_BADGE_COLOR  # Invalid hex chars
        assert validate_hex_color("#12345") == DEFAULT_BADGE_COLOR  # 5 digits (invalid)
        assert validate_hex_color("ff0000") == DEFAULT_BADGE_COLOR  # Missing #

    def test_empty_string_falls_back(self):
        """Verify empty strings fall back to default."""
        assert validate_hex_color("") == DEFAULT_BADGE_COLOR
        assert validate_hex_color("   ") == DEFAULT_BADGE_COLOR

    def test_none_input_falls_back(self):
        """Verify None input falls back to default."""
        assert validate_hex_color(None) == DEFAULT_BADGE_COLOR

    def test_non_string_input_falls_back(self):
        """Verify non-string inputs fall back to default."""
        assert validate_hex_color(12345) == DEFAULT_BADGE_COLOR
        assert validate_hex_color(["#ff0000"]) == DEFAULT_BADGE_COLOR

    def test_custom_default_color(self):
        """Verify custom default color is used when validation fails."""
        assert validate_hex_color("invalid", default_color="#000000") == "#000000"
        assert validate_hex_color("", default_color="#ffffff") == "#ffffff"

    def test_logs_warning_on_fallback(self, caplog):
        """Verify a warning is logged when falling back to default."""
        with caplog.at_level(logging.WARNING):
            validate_hex_color("invalid_color_name")

        assert any(
            "Unrecognized color format" in record.message for record in caplog.records
        )


class TestGenerateSvgBadge:
    """Test suite for SVG badge generation with color validation."""

    def test_badge_uses_named_colors(self):
        """Verify generate_svg_badge correctly resolves named colors."""
        svg = generate_svg_badge(
            "build", "passing", label_color="gray", message_color="green"
        )

        # Verify the resolved hex codes are in the SVG
        assert "#808080" in svg  # gray
        assert "#008000" in svg  # green

    def test_badge_fallback_on_invalid_colors(self):
        """Verify generate_svg_badge uses defaults for invalid colors."""
        svg = generate_svg_badge(
            "test", "fail", label_color="invalid", message_color="invalid"
        )

        # Should fall back to #555555 and #007ec6 (or the function defaults)
        assert "#555555" in svg or "#555" in svg
        assert "#007ec6" in svg

    def test_badge_contains_text(self):
        """Verify the generated SVG contains the label and message text."""
        svg = generate_svg_badge("coverage", "95%")
        assert "coverage" in svg
        assert "95%" in svg
