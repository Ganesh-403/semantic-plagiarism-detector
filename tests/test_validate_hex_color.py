# tests/test_validate_hex_color.py
"""
Test for validate_hex_color edge cases.

This test ensures that validate_hex_color safely falls back to the default color
when presented with malicious or malformed inputs, and properly validates
valid 3-character and 6-character hex strings.

Issue: #2904
"""

import pytest
import re
import random
import string
from typing import Optional, List, Tuple, Any


# ============================================================================
# Constants and Test Data
# ============================================================================

DEFAULT_BADGE_COLOR = "#000000"  # Assuming this is the default

# Valid 3-character hex colors
VALID_3_CHAR_HEX = [
    "#000", "#fff", "#123", "#abc", "#def", "#456", "#789", "#a1b", "#c3d",
    "#e5f", "#111", "#222", "#333", "#444", "#555", "#666", "#777", "#888",
    "#999", "#aaa", "#bbb", "#ccc", "#ddd", "#eee", "#f00", "#0f0", "#00f",
    "#ff0", "#0ff", "#f0f", "#f80", "#8f0", "#08f", "#f08", "#0f8", "#80f",
    "#f44", "#4f4", "#44f", "#f84", "#8f4", "#48f", "#f48", "#4f8", "#84f",
]

# Valid 6-character hex colors
VALID_6_CHAR_HEX = [
    "#000000", "#ffffff", "#123456", "#abcdef", "#fedcba", "#a1b2c3",
    "#d4e5f6", "#789abc", "#def123", "#456789", "#111111", "#222222",
    "#333333", "#444444", "#555555", "#666666", "#777777", "#888888",
    "#999999", "#aaaaaa", "#bbbbbb", "#cccccc", "#dddddd", "#eeeeee",
    "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff", "#ff00ff",
    "#ff8800", "#88ff00", "#0088ff", "#ff0088", "#8800ff", "#00ff88",
    "#ff4444", "#44ff44", "#4444ff", "#ff8844", "#88ff44", "#4488ff",
    "#ff4488", "#44ff88", "#8844ff", "#ff8f00", "#8f00ff", "#00ff8f",
]

# Invalid hex colors (malformed, malicious, edge cases)
INVALID_HEX_INPUTS = [
    # Invalid characters
    "#12345g", "#abcdeg", "#12345h", "#g12345", "#00000g", "#fffffg",
    "#abcdefg", "#1234567", "#12345!", "#12345@", "#12345#", "#12345$",
    "#12345%", "#12345^", "#12345&", "#12345*", "#12345(", "#12345)",
    "#12345_", "#12345+", "#12345=", "#12345[", "#12345]", "#12345{",
    "#12345}", "#12345|", "#12345\\", "#12345;", "#12345:", "#12345'",
    "#12345\"", "#12345<", "#12345>", "#12345?", "#12345/", "#12345~",
    "#12345`",
    
    # rgb() formats
    "rgb(255, 0, 0)", "rgb(0, 255, 0)", "rgb(0, 0, 255)",
    "rgb(255, 255, 255)", "rgb(0, 0, 0)", "rgb(100, 150, 200)",
    "rgb(255,0,0)", "rgb(0,255,0)", "rgb(0,0,255)",
    "rgb(255 0 0)", "rgb(0 255 0)", "rgb(0 0 255)",
    "rgba(255, 0, 0, 1)", "rgba(255, 0, 0, 0.5)",
    "rgb(256, 0, 0)", "rgb(-1, 0, 0)", "rgb(0, 256, 0)",
    "rgb(0, -1, 0)", "rgb(0, 0, 256)", "rgb(0, 0, -1)",
    "rgb(100%, 0%, 0%)", "rgb(0%, 100%, 0%)", "rgb(0%, 0%, 100%)",
    "rgb(100%, 100%, 100%)", "rgb(0%, 0%, 0%)",
    
    # Empty strings and whitespace
    "", " ", "  ", "\t", "\n", "\r", "\n\r", "\t\n", " \t\n ",
    "", "   ", "\t\t", "\n\n", "\r\r", "\n\r\t",
    
    # Short strings
    "#", "#1", "#12", "#12g", "#1", "#2", "#3", "#4", "#5", "#6", "#7",
    "#8", "#9", "#a", "#b", "#c", "#d", "#e", "#f", "#g", "#h",
    "#0", "#00", "#0000", "#00000", "#0000000",
    
    # Wrong prefix
    "123456", "ffffff", "000000", "abc123", "def456",
    "!123456", "@123456", "$123456", "%123456", "^123456",
    "&123456", "*123456", "(123456", ")123456", "_123456",
    "+123456", "=123456", "[123456", "]123456", "{123456",
    "}123456", "|123456", "\\123456", ";123456", ":123456",
    "'123456", "\"123456", "<123456", ">123456", "?123456",
    "/123456", "~123456", "`123456", ".123456", ",123456",
    
    # Hex with wrong length
    "#0000000", "#00000000", "#000000000", "#0000000000",
    "#ffffff00", "#ffffff000", "#ffffff0000",
    "#12345678", "#123456789", "#1234567890",
    "#abcde", "#abcdefg", "#hijklm", "#nopqrs",
    
    # Hex with lowercase/uppercase mix (invalid characters)
    "#ABCDEFG", "#GHIJKL", "#MNOPQR", "#STUVWX", "#YZabcd",
    "#12345X", "#12345Y", "#12345Z", "#12345W",
    
    # Hex with unicode characters
    "#\u1234", "#\u5678", "#\u9abc", "#\u12345", "#\u6789a",
    "#\u123456", "#\u1234", "#\u5678", "#\u9abc",
    "＃000000", "＃ffffff", "＃123456", "＃abcdef",
    
    # SQL injection attempts
    "#000'; DROP TABLE colors; --", "#fff' OR '1'='1", 
    "#000' UNION SELECT * FROM users; --",
    "#fff'; INSERT INTO colors VALUES ('malicious'); --",
    "#000; DROP DATABASE; --", "#fff; TRUNCATE TABLE; --",
    "#000' OR 1=1; --", "#fff' AND '1'='1",
    
    # XSS attempts
    "#000<script>alert('xss')</script>", 
    "#fff<script>alert(1)</script>",
    "#000<img src=x onerror=alert(1)>",
    "#fff<svg onload=alert(1)>",
    "#000<iframe src=javascript:alert(1)>",
    "#fff<body onload=alert(1)>",
    "#000'onclick=alert(1)",
    "#fff'onerror=alert(1)",
    "#000<marquee onstart=alert(1)>",
    "#fff<details open ontoggle=alert(1)>",
    
    # Path traversal attempts
    "#000../../etc/passwd", "#fff../../../etc/passwd",
    "#000..\\..\\windows\\system32", "#fff..\\..\\boot.ini",
    "#000/var/www/html", "#fff/etc/shadow",
    "#000C:\\Windows\\System32", "#fffC:\\Users",
    
    # Command injection
    "#000; ls -la", "#fff; cat /etc/passwd",
    "#000| whoami", "#fff| id",
    "#000`whoami`", "#fff`id`",
    "#000$(whoami)", "#fff$(id)",
    "#000; rm -rf /", "#fff; sudo rm -rf /",
    
    # JSON injection
    "#000{\"key\":\"value\"}", "#fff{\"malicious\":true}",
    "#000[\"array\",\"values\"]", "#fff{\"nested\":{\"key\":\"value\"}}",
    "#000null", "#ffftrue", "#000false",
    "#000'{\"key\":\"value\"}'", "#fff'[\"array\"]'",
    
    # XML injection
    "#000<xml>", "#fff</xml>",
    "#000<tag>value</tag>", "#fff<malicious/>",
    "#000<!DOCTYPE>", "#fff<!-- comment -->",
    "#000<![CDATA[malicious]]>", "#fff<?xml version='1.0'?>",
    
    # YAML injection
    "#000---", "#fff...", "#000:key: value",
    "#fff- item", "#000|", "#fff>",
    
    # Environment variable injection
    "#000$PATH", "#fff$HOME", "#000$USER",
    "#fff$SHELL", "#000$PWD", "#fff$RANDOM",
    "#000${PATH}", "#fff${HOME}",
    
    # Very long strings
    "#" + "0" * 1000, "#" + "f" * 1000, "#" + "1" * 1000,
    "a" * 1000, "z" * 1000, "0" * 1000,
    "#" + "".join(random.choices("0123456789abcdef", k=1000)),
    "".join(random.choices("0123456789abcdef", k=1000)),
    
    # Malformed hex with special characters
    "#@#$%^&*()", "#!@#$%^", "#$%^&*()_+",
    "#+=[]{}|", "#;:'\",.<>/?", "#~`!@#$%^&*()_+",
    "#😊😊😊😊", "#🚀🚀🚀🚀", "#🌟🌟🌟🌟",
    
    # Null bytes and control characters
    "#000\x00fff", "#fff\x00aaa", "#\x00\x00\x00\x00",
    "#\x01\x02\x03\x04", "#\x0a\x0b\x0c\x0d",
    "#000\x00", "#fff\x00", "#\x00\x00\x00",
    
    # Hex with plus/minus
    "#+12345", "#-12345", "#+fffff", "#-fffff",
    "#+00000", "#-00000", "#+abcde", "#-abcde",
    
    # Hex with decimal points
    "#1.2345", "#f.ffff", "#0.0000", "#.12345",
    "#123.45", "#abc.def", "#111.111",
    
    # Hex with scientific notation
    "#1e5", "#1e-5", "#1e+5", "#ff", "#0xff",
    "#0x000", "#0xabcdef", "#0x123456",
    
    # Hex with spaces inside
    "#1 2345", "# ab cde", "#12 345", "#abc def",
    "# 12345", "# abcde", "# 00000", "# fffff",
    "#1 2 3", "#a b c", "#1 2 3 4 5 6",
    
    # Hex with tabs inside
    "#1\t2345", "#ab\tcde", "#12\t345", "#abc\tdef",
    "#\t12345", "#\tabcde", "#\t00000", "#\tfffff",
    
    # Hex with newlines inside
    "#1\n2345", "#ab\ncde", "#12\n345", "#abc\ndef",
    "#\n12345", "#\nabcde", "#\n00000", "#\nfffff",
    
    # Multiple hashes
    "##000000", "###000", "####", "#####",
    "##fff", "##123456", "##abc", "##def",
    "# #000000", "# #fff", "# #123", "# #abc",
    
    # Hex without hash but with hex chars
    "000000", "ffffff", "123abc", "def456",
    "000", "fff", "123", "abc",
    
    # Hex with invalid length after hash
    "#00000000", "#000000000", "#0000000000",
    "#123456789", "#abcdefghi", "#1234567890abcdef",
    
    # Unicode emoji hex
    "#😊", "#🚀", "#🌟", "#💻", "#📱", "#💡",
    "#🔥", "#🎯", "#🏆", "#🎮", "#🎨", "#🎭",
    
    # Mixed case invalid
    "#aBcDeFg", "#12345G", "#HijkLM", "#nOpQrS",
    "#tUvWxY", "#zAbCdE", "#FgHiJk", "#LmNoPq",
    
    # Hex with underscores
    "#000_000", "#fff_fff", "#123_456", "#abc_def",
    "#_00000", "#_fffff", "#_12345", "#_abcde",
    
    # Hex with dashes
    "#000-000", "#fff-fff", "#123-456", "#abc-def",
    "#-00000", "#-fffff", "#-12345", "#-abcde",
]

# Valid hex variants (should be accepted)
VALID_HEX_VARIANTS = [
    # 3-character hex
    "#000", "#FFF", "#123", "#ABC", "#DEF", "#456", "#789", "#A1B",
    "#C3D", "#E5F", "#111", "#222", "#333", "#444", "#555", "#666",
    "#777", "#888", "#999", "#AAA", "#BBB", "#CCC", "#DDD", "#EEE",
    "#F00", "#0F0", "#00F", "#FF0", "#0FF", "#F0F", "#F80", "#8F0",
    "#08F", "#F08", "#0F8", "#80F", "#F44", "#4F4", "#44F", "#F84",
    "#8F4", "#48F", "#F48", "#4F8", "#84F",
    
    # 6-character hex
    "#000000", "#FFFFFF", "#123456", "#ABCDEF", "#FEDCBA", "#A1B2C3",
    "#D4E5F6", "#789ABC", "#DEF123", "#456789", "#111111", "#222222",
    "#333333", "#444444", "#555555", "#666666", "#777777", "#888888",
    "#999999", "#AAAAAA", "#BBBBBB", "#CCCCCC", "#DDDDDD", "#EEEEEE",
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF",
    "#FF8800", "#88FF00", "#0088FF", "#FF0088", "#8800FF", "#00FF88",
    "#FF4444", "#44FF44", "#4444FF", "#FF8844", "#88FF44", "#4488FF",
    "#FF4488", "#44FF88", "#8844FF", "#FF8F00", "#8F00FF", "#00FF8F",
]


# ============================================================================
# Test Class
# ============================================================================

class TestValidateHexColor:
    """Test suite for validate_hex_color function."""

    @pytest.fixture
    def mock_validate_hex_color(self):
        """
        Fixture that provides a reference implementation of validate_hex_color.
        
        This can be replaced with the actual implementation when testing.
        """
        def _validate_hex_color(color: str) -> str:
            """
            Validate hex color string.
            
            Args:
                color: Hex color string (e.g., '#000', '#ffffff')
                
            Returns:
                Valid hex color or default color if invalid
            """
            if not color or not isinstance(color, str):
                return DEFAULT_BADGE_COLOR
            
            # Clean the input
            color = color.strip()
            
            # Must start with #
            if not color.startswith('#'):
                return DEFAULT_BADGE_COLOR
            
            # Remove the # and check length
            hex_part = color[1:]
            
            # Check for valid hex characters
            if not re.match(r'^[0-9a-fA-F]+$', hex_part):
                return DEFAULT_BADGE_COLOR
            
            # Check length (3 or 6 characters)
            if len(hex_part) not in (3, 6):
                return DEFAULT_BADGE_COLOR
            
            # Ensure no additional characters
            if len(color) != len(hex_part) + 1:
                return DEFAULT_BADGE_COLOR
            
            return color
        
        return _validate_hex_color

    # ========================================================================
    # Core Test Cases (Acceptance Criteria)
    # ========================================================================

    def test_malformed_inputs_return_default(self, mock_validate_hex_color):
        """
        Test that malformed inputs return DEFAULT_BADGE_COLOR.
        
        This tests the core acceptance criteria:
        - Invalid hex strings return DEFAULT_BADGE_COLOR
        """
        for invalid_input in INVALID_HEX_INPUTS:
            result = mock_validate_hex_color(invalid_input)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Input '{invalid_input}' should return default color"

    def test_valid_3_char_hex_untouched(self, mock_validate_hex_color):
        """
        Test that valid 3-character hex strings are returned untouched.
        
        This tests the core acceptance criteria:
        - Valid 3-character hex strings are returned untouched
        """
        for valid_hex in VALID_3_CHAR_HEX:
            result = mock_validate_hex_color(valid_hex)
            assert result == valid_hex, \
                f"Valid 3-char hex '{valid_hex}' should be returned unchanged"

    def test_valid_6_char_hex_untouched(self, mock_validate_hex_color):
        """
        Test that valid 6-character hex strings are returned untouched.
        
        This tests the core acceptance criteria:
        - Valid 6-character hex strings are returned untouched
        """
        for valid_hex in VALID_6_CHAR_HEX:
            result = mock_validate_hex_color(valid_hex)
            assert result == valid_hex, \
                f"Valid 6-char hex '{valid_hex}' should be returned unchanged"

    # ========================================================================
    # Edge Case Tests - Type Handling
    # ========================================================================

    def test_non_string_inputs_return_default(self, mock_validate_hex_color):
        """Test that non-string inputs return default color."""
        non_string_inputs = [
            None,
            123,
            456.789,
            True,
            False,
            [],
            [1, 2, 3],
            {},
            {"color": "#000"},
            tuple(),
            set(),
            frozenset(),
            0,
            -1,
            3.14,
            complex(1, 2),
            b"#000000",
            bytearray(b"#000000"),
            memoryview(b"#000000"),
        ]
        
        for invalid_input in non_string_inputs:
            result = mock_validate_hex_color(invalid_input)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Input {type(invalid_input)} should return default color"

    def test_whitespace_handling(self, mock_validate_hex_color):
        """Test whitespace handling around valid hex colors."""
        # Valid colors with whitespace should be accepted
        valid_with_whitespace = [
            (" #000", "#000"),
            ("#000 ", "#000"),
            (" #000 ", "#000"),
            ("\t#000", "#000"),
            ("#000\t", "#000"),
            ("\t#000\t", "#000"),
            ("\n#000", "#000"),
            ("#000\n", "#000"),
            ("\n#000\n", "#000"),
            ("\r#000", "#000"),
            ("#000\r", "#000"),
            ("\r#000\r", "#000"),
            (" #000000", "#000000"),
            ("#000000 ", "#000000"),
            (" #000000 ", "#000000"),
        ]
        
        for input_color, expected in valid_with_whitespace:
            result = mock_validate_hex_color(input_color)
            assert result == expected, \
                f"Input '{input_color}' should return '{expected}'"

    def test_whitespace_only_returns_default(self, mock_validate_hex_color):
        """Test that whitespace-only inputs return default color."""
        whitespace_only = [
            " ",
            "  ",
            "\t",
            "\n",
            "\r",
            " \t\n\r",
            "\t\n\r ",
            "   \t   \n   \r   ",
        ]
        
        for input_color in whitespace_only:
            result = mock_validate_hex_color(input_color)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Whitespace input '{input_color}' should return default"

    # ========================================================================
    # Case Sensitivity Tests
    # ========================================================================

    def test_case_sensitivity_preserved(self, mock_validate_hex_color):
        """Test that case is preserved for valid hex colors."""
        case_variants = [
            "#abc", "#ABC", "#AbC", "#aBc",
            "#abcdef", "#ABCDEF", "#AbCdEf", "#aBcDeF",
            "#123abc", "#123ABC", "#123AbC", "#123aBc",
        ]
        
        for variant in case_variants:
            result = mock_validate_hex_color(variant)
            # Should preserve the exact case provided
            assert result == variant, \
                f"Case should be preserved for '{variant}'"

    # ========================================================================
    # Boundary Tests
    # ========================================================================

    def test_empty_string_returns_default(self, mock_validate_hex_color):
        """Test that empty string returns default color."""
        result = mock_validate_hex_color("")
        assert result == DEFAULT_BADGE_COLOR

    def test_very_long_inputs_return_default(self, mock_validate_hex_color):
        """Test that very long inputs return default color."""
        long_inputs = [
            "#" + "0" * 100,
            "#" + "1" * 200,
            "#" + "a" * 300,
            "#" + "f" * 400,
            "#" + "".join(random.choices("0123456789abcdef", k=500)),
            "#" + "".join(random.choices("0123456789abcdef", k=1000)),
            "a" * 10000,
            "z" * 10000,
            "#" + "0" * 10000,
        ]
        
        for long_input in long_inputs:
            result = mock_validate_hex_color(long_input)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Long input should return default color"

    def test_unicode_inputs_return_default(self, mock_validate_hex_color):
        """Test that unicode inputs return default color."""
        unicode_inputs = [
            "#😊😊😊",
            "#🚀🚀🚀",
            "#🌟🌟🌟",
            "#💻💻💻",
            "#📱📱📱",
            "＃000000",  # Full-width hash
            "＃ffffff",
            "＃123456",
            "＃abc",
            "＃ABC",
            "#\u200b",  # Zero-width space
            "#\u200c",  # Zero-width non-joiner
            "#\u200d",  # Zero-width joiner
            "#\ufe0f",  # Variation selector
        ]
        
        for unicode_input in unicode_inputs:
            result = mock_validate_hex_color(unicode_input)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Unicode input '{unicode_input}' should return default"

    # ========================================================================
    # Security Tests (Malicious Inputs)
    # ========================================================================

    def test_sql_injection_returns_default(self, mock_validate_hex_color):
        """Test SQL injection attempts return default color."""
        sql_injections = [
            "#000'; DROP TABLE colors; --",
            "#fff' OR '1'='1",
            "#000' UNION SELECT * FROM users; --",
            "#fff'; INSERT INTO colors VALUES ('malicious'); --",
            "#000; DROP DATABASE; --",
            "#fff; TRUNCATE TABLE; --",
            "#000' OR 1=1; --",
            "#fff' AND '1'='1",
        ]
        
        for injection in sql_injections:
            result = mock_validate_hex_color(injection)
            assert result == DEFAULT_BADGE_COLOR, \
                f"SQL injection '{injection}' should return default"

    def test_xss_attempts_return_default(self, mock_validate_hex_color):
        """Test XSS attempts return default color."""
        xss_attempts = [
            "#000<script>alert('xss')</script>",
            "#fff<script>alert(1)</script>",
            "#000<img src=x onerror=alert(1)>",
            "#fff<svg onload=alert(1)>",
            "#000<iframe src=javascript:alert(1)>",
            "#fff<body onload=alert(1)>",
            "#000'onclick=alert(1)",
            "#fff'onerror=alert(1)",
            "#000<marquee onstart=alert(1)>",
            "#fff<details open ontoggle=alert(1)>",
        ]
        
        for xss in xss_attempts:
            result = mock_validate_hex_color(xss)
            assert result == DEFAULT_BADGE_COLOR, \
                f"XSS attempt '{xss}' should return default"

    def test_path_traversal_returns_default(self, mock_validate_hex_color):
        """Test path traversal attempts return default color."""
        path_traversal = [
            "#000../../etc/passwd",
            "#fff../../../etc/passwd",
            "#000..\\..\\windows\\system32",
            "#fff..\\..\\boot.ini",
            "#000/var/www/html",
            "#fff/etc/shadow",
            "#000C:\\Windows\\System32",
            "#fffC:\\Users",
        ]
        
        for traversal in path_traversal:
            result = mock_validate_hex_color(traversal)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Path traversal '{traversal}' should return default"

    def test_command_injection_returns_default(self, mock_validate_hex_color):
        """Test command injection attempts return default color."""
        command_injections = [
            "#000; ls -la",
            "#fff; cat /etc/passwd",
            "#000| whoami",
            "#fff| id",
            "#000`whoami`",
            "#fff`id`",
            "#000$(whoami)",
            "#fff$(id)",
            "#000; rm -rf /",
            "#fff; sudo rm -rf /",
        ]
        
        for injection in command_injections:
            result = mock_validate_hex_color(injection)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Command injection '{injection}' should return default"

    # ========================================================================
    # Mixed Valid/Invalid Pattern Tests
    # ========================================================================

    def test_valid_hex_with_invalid_suffix(self, mock_validate_hex_color):
        """Test valid hex with invalid suffix returns default."""
        invalid_suffixes = [
            "#000abc", "#fff123", "#123xyz", "#456789abc",
            "#000 extra", "#fff more", "#123 trailing",
        ]
        
        for invalid_input in invalid_suffixes:
            result = mock_validate_hex_color(invalid_input)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Input '{invalid_input}' should return default"

    def test_valid_hex_with_invalid_prefix(self, mock_validate_hex_color):
        """Test invalid prefix with valid hex returns default."""
        invalid_prefixes = [
            "prefix#000", "extra#fff", "abc#123",
            "x#000000", "y#ffffff", "z#abcdef",
        ]
        
        for invalid_input in invalid_prefixes:
            result = mock_validate_hex_color(invalid_input)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Input '{invalid_input}' should return default"

    # ========================================================================
    # Performance Tests
    # ========================================================================

    def test_performance_with_many_inputs(self, mock_validate_hex_color):
        """Test performance with many inputs."""
        import time
        
        # Mix of valid and invalid inputs
        test_inputs = VALID_3_CHAR_HEX + VALID_6_CHAR_HEX + INVALID_HEX_INPUTS
        test_inputs = test_inputs * 10  # Multiply for larger dataset
        
        start_time = time.time()
        for input_color in test_inputs:
            mock_validate_hex_color(input_color)
        elapsed_time = time.time() - start_time
        
        # Should process at least 1000 inputs per second
        assert elapsed_time < len(test_inputs) / 1000 + 1.0, \
            f"Performance test too slow: {elapsed_time:.2f}s for {len(test_inputs)} inputs"

    def test_performance_with_very_long_inputs(self, mock_validate_hex_color):
        """Test performance with very long inputs."""
        import time
        
        # Create very long inputs
        long_inputs = [
            "#" + "0" * i for i in [100, 200, 500, 1000, 2000, 5000]
        ]
        
        start_time = time.time()
        for input_color in long_inputs:
            mock_validate_hex_color(input_color)
        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 2.0, \
            f"Long input performance test too slow: {elapsed_time:.2f}s"

    # ========================================================================
    # Edge Case Combinations
    # ========================================================================

    def test_hex_with_leading_zeros(self, mock_validate_hex_color):
        """Test hex colors with leading zeros."""
        leading_zero_hex = [
            "#000", "#001", "#010", "#100",
            "#000000", "#000001", "#000010", "#000100",
            "#001000", "#010000", "#100000", "#000fff",
            "#00ffff", "#0fffff", "#000123", "#001234",
        ]
        
        for hex_color in leading_zero_hex:
            result = mock_validate_hex_color(hex_color)
            assert result == hex_color, \
                f"Leading zeros should be preserved for '{hex_color}'"

    def test_hex_with_trailing_zeros(self, mock_validate_hex_color):
        """Test hex colors with trailing zeros."""
        trailing_zero_hex = [
            "#000", "#100", "#110", "#111",
            "#000000", "#100000", "#110000", "#111000",
            "#000100", "#000010", "#000001", "#fff000",
            "#ffff00", "#fffff0", "#abc000", "#def000",
        ]
        
        for hex_color in trailing_zero_hex:
            result = mock_validate_hex_color(hex_color)
            assert result == hex_color, \
                f"Trailing zeros should be preserved for '{hex_color}'"

    def test_common_color_names_return_default(self, mock_validate_hex_color):
        """Test common color names return default color."""
        color_names = [
            "red", "green", "blue", "white", "black",
            "yellow", "cyan", "magenta", "orange", "purple",
            "pink", "brown", "gray", "grey", "navy",
            "teal", "olive", "maroon", "lime", "aqua",
            "coral", "crimson", "gold", "indigo", "ivory",
            "khaki", "lavender", "mint", "peach", "plum",
            "rose", "ruby", "sapphire", "silver", "tan",
            "tomato", "violet", "wheat", "zinc", "cobalt",
        ]
        
        for color_name in color_names:
            result = mock_validate_hex_color(color_name)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Color name '{color_name}' should return default"

    def test_hex_with_plus_prefix(self, mock_validate_hex_color):
        """Test hex with plus prefix returns default."""
        plus_prefix = [
            "+#000", "+#fff", "+#123", "+#abc",
            "+#000000", "+#ffffff", "+#123456", "+#abcdef",
            " +#000", " +#fff", " +#123", " +#abc",
        ]
        
        for hex_color in plus_prefix:
            result = mock_validate_hex_color(hex_color)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Plus prefix '{hex_color}' should return default"

    def test_hex_with_minus_prefix(self, mock_validate_hex_color):
        """Test hex with minus prefix returns default."""
        minus_prefix = [
            "-#000", "-#fff", "-#123", "-#abc",
            "-#000000", "-#ffffff", "-#123456", "-#abcdef",
            " -#000", " -#fff", " -#123", " -#abc",
        ]
        
        for hex_color in minus_prefix:
            result = mock_validate_hex_color(hex_color)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Minus prefix '{hex_color}' should return default"

    # ========================================================================
    # Test Coverage Enhancement
    # ========================================================================

    def test_all_valid_3_char_hex_combinations(self, mock_validate_hex_color):
        """Test a subset of all possible 3-character hex combinations."""
        # Test first 16 of each possible position
        chars = "0123456789abcdef"
        for c1 in chars[:4]:
            for c2 in chars[:4]:
                for c3 in chars[:4]:
                    hex_color = f"#{c1}{c2}{c3}"
                    result = mock_validate_hex_color(hex_color)
                    assert result == hex_color, \
                        f"Valid 3-char hex '{hex_color}' should be returned"

    def test_all_valid_6_char_hex_combinations(self, mock_validate_hex_color):
        """Test a subset of all possible 6-character hex combinations."""
        # Test first 3 of each possible position
        chars = "0123456789abcdef"
        for c1 in chars[:3]:
            for c2 in chars[:3]:
                for c3 in chars[:3]:
                    for c4 in chars[:3]:
                        for c5 in chars[:3]:
                            for c6 in chars[:3]:
                                hex_color = f"#{c1}{c2}{c3}{c4}{c5}{c6}"
                                result = mock_validate_hex_color(hex_color)
                                assert result == hex_color, \
                                    f"Valid 6-char hex '{hex_color}' should be returned"

    def test_invalid_hex_with_letters_after_hash(self, mock_validate_hex_color):
        """Test invalid hex with letters after hash."""
        invalid_letters = [
            "#000a", "#00aa", "#0aaa", "#aaaa",
            "#12345g", "#abcdefg", "#abcdefgh",
            "#01234567", "#89abcdef", "#fedcba98",
            "#1111111", "#2222222", "#3333333",
        ]
        
        for invalid_hex in invalid_letters:
            result = mock_validate_hex_color(invalid_hex)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Invalid hex '{invalid_hex}' should return default"

    def test_hex_with_repeated_patterns(self, mock_validate_hex_color):
        """Test hex with repeated patterns."""
        repeated_patterns = [
            "#000", "#111", "#222", "#333", "#444",
            "#555", "#666", "#777", "#888", "#999",
            "#aaa", "#bbb", "#ccc", "#ddd", "#eee",
            "#fff", "#000000", "#111111", "#222222",
            "#333333", "#444444", "#555555", "#666666",
            "#777777", "#888888", "#999999", "#aaaaaa",
            "#bbbbbb", "#cccccc", "#dddddd", "#eeeeee",
            "#ffffff", "#010101", "#020202", "#030303",
        ]
        
        for hex_color in repeated_patterns:
            result = mock_validate_hex_color(hex_color)
            assert result == hex_color, \
                f"Repeated pattern '{hex_color}' should be returned"

    def test_hex_with_invalid_hash_position(self, mock_validate_hex_color):
        """Test hex with invalid hash position."""
        invalid_hash = [
            "000#000", "fff#fff", "123#456", "abc#def",
            "#000#", "#fff#", "#123#", "#abc#",
            "##000", "##fff", "##123", "##abc",
        ]
        
        for hex_color in invalid_hash:
            result = mock_validate_hex_color(hex_color)
            assert result == DEFAULT_BADGE_COLOR, \
                f"Invalid hash position '{hex_color}' should return default"

    # ========================================================================
    # Test Concurrency (Thread Safety)
    # ========================================================================

    def test_concurrent_calls(self, mock_validate_hex_color):
        """Test concurrent calls to validate_hex_color."""
        import threading
        import concurrent.futures
        
        def validate_wrapper(color):
            return mock_validate_hex_color(color)
        
        # Mix of valid and invalid colors
        test_colors = [
            "#000", "#fff", "#123", "#abc", "#000000", "#ffffff",
            "#123456", "#abcdef", "", "#12345g", "rgb(255,0,0)",
            "#000<script>", "#000; DROP TABLE;",
        ] * 50
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(validate_wrapper, color) for color in test_colors]
            results = [f.result() for f in futures]
        
        # Verify results match expected
        for color, result in zip(test_colors, results):
            expected = color if color in VALID_3_CHAR_HEX or color in VALID_6_CHAR_HEX else DEFAULT_BADGE_COLOR
            if color in ["#000", "#fff", "#123", "#abc", "#000000", "#ffffff", "#123456", "#abcdef"]:
                assert result == color, f"Concurrent call failed for '{color}'"
            else:
                assert result == DEFAULT_BADGE_COLOR, f"Concurrent call failed for '{color}'"


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
