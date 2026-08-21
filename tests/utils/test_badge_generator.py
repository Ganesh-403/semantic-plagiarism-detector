import pytest

from src.utils.badge_generator import (
    DEFAULT_BADGE_COLOR,
    generate_badge_svg,
    validate_hex_color,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("#4f46e5", "#4f46e5"),
        ("4f46e5", "#4f46e5"),
        ("#fff", "#fff"),
        ("fff", "#fff"),
        ("not-a-color", DEFAULT_BADGE_COLOR),
        ("", DEFAULT_BADGE_COLOR),
        (None, DEFAULT_BADGE_COLOR),
    ],
)
def test_validate_hex_color(raw, expected):
    assert validate_hex_color(raw) == expected


def test_generate_badge_svg_uses_validated_color():
    svg = generate_badge_svg(student_name="Alex", accent_color="not-a-color")
    assert DEFAULT_BADGE_COLOR in svg
    assert "not-a-color" not in svg


def test_generate_badge_svg_escapes_student_name():
    svg = generate_badge_svg(student_name="<script>alert(1)</script>")
    assert "<script>" not in svg


def test_generate_badge_svg_default_font_family():
    svg = generate_badge_svg(student_name="Alex")
    assert 'font-family="Verdana, Geneva, sans-serif"' in svg


def test_generate_badge_svg_custom_font_family():
    svg = generate_badge_svg(student_name="Alex", font_family="Arial, sans-serif")
    assert 'font-family="Arial, sans-serif"' in svg


def test_generate_badge_svg_default_font_size():
    svg = generate_badge_svg(student_name="Alex")
    assert 'font-size="11"' in svg


def test_generate_badge_svg_custom_font_size():
    svg = generate_badge_svg(student_name="Alex", font_size=20)
    assert 'font-size="20"' in svg
