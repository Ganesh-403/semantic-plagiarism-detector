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
