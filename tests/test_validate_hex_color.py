"""Badge color validation uses the production CSS/hex parser."""

import pytest
from src.utils.badge_generator import DEFAULT_BADGE_COLOR, validate_hex_color


@pytest.mark.parametrize(("value", "expected"), [
    ("#abc", "#aabbcc"), ("#1e5", "#11ee55"), ("#000abc", "#000abc"),
    ("#123456", "#123456"), ("#FFF", "#ffffff"),
    ("#abcd", "#aabbccdd"), ("#12345678", "#12345678"),
    ("  #ABCDEF  ", "#abcdef"), ("red", "#ff0000"),
    ("transparent", "#00000000"),
])
def test_valid_colors(value, expected):
    assert validate_hex_color(value) == expected


@pytest.mark.parametrize("value", [
    None, 123, [], {}, "", " ", "#", "#12", "#12345", "#1234567",
    "#12345g", "#abcdefg", "rgb(1,2,3)", "#abc;display:none",
    "<script>alert(1)</script>", "#abc\x00", "#ab\nc", "#" + "f" * 1000,
    "url(javascript:alert(1))", "#1e+5", "#000$(id)", "#fff' onclick='alert(1)",
])
def test_invalid_colors_fall_back(value):
    assert validate_hex_color(value) == DEFAULT_BADGE_COLOR
    assert validate_hex_color(value, "#123456") == "#123456"
