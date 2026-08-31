import pytest

from src.utils.badge_generator import (
    DEFAULT_BADGE_COLOR,
    generate_badge_pdf,
    generate_badge_png,
    generate_badge_svg,
    has_reportlab,
    validate_hex_color,
)
from src.utils.redis_cache import CacheNamespace, RedisCache


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("#4f46e5", "#4f46e5"),
        ("#abc", "#abc"),
        ("#fff", "#fff"),
        ("#1e3a8a", "#1e3a8a"),
        ("#12345g", DEFAULT_BADGE_COLOR),
        ("rgb(255, 0, 0)", DEFAULT_BADGE_COLOR),
        ("", DEFAULT_BADGE_COLOR),
        (None, DEFAULT_BADGE_COLOR),
    ],
)
def test_validate_hex_color(raw, expected):
    assert validate_hex_color(raw, DEFAULT_BADGE_COLOR) == expected


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


def test_has_reportlab():
    from src.utils.badge_generator import SimpleDocTemplate

    assert has_reportlab() == (SimpleDocTemplate is not None)


def test_generate_badge_pdf_raises_when_reportlab_missing(monkeypatch):
    import src.utils.badge_generator as bg

    monkeypatch.setattr(bg, "SimpleDocTemplate", None)
    with pytest.raises(
        ImportError, match="reportlab is required for PDF badge generation"
    ):
        bg.generate_badge_pdf()


def test_generate_badge_png_and_caching():
    """Test generating PNG badge and caching in Redis."""
    cache = RedisCache.get_instance()
    student_id = "test_student_123"
    date_str = "2026-08-20"
    cache_key = CacheNamespace.BADGES.build_key("png", student_id, date_str)

    # Invalidate cache if present
    cache.delete(cache_key)

    buf1 = generate_badge_png(
        student_name="Test Student",
        date=date_str,
        student_id=student_id,
    )
    val1 = buf1.getvalue()
    assert val1.startswith(b"\x89PNG")

    # Verify cached in Redis
    cached_val = cache.get(cache_key)
    assert cached_val is not None
    assert cached_val == val1

    # Second call hits cache
    buf2 = generate_badge_png(
        student_name="Test Student",
        date=date_str,
        student_id=student_id,
    )
    assert buf2.getvalue() == val1


def test_generate_badge_pdf_and_caching():
    """Test generating PDF certificate and caching in Redis."""
    cache = RedisCache.get_instance()
    student_id = "test_student_456"
    date_str = "2026-08-20"
    cache_key = CacheNamespace.BADGES.build_key("pdf", student_id, date_str)

    # Invalidate cache if present
    cache.delete(cache_key)

    buf1 = generate_badge_pdf(
        student_name="Test Student",
        date=date_str,
        student_id=student_id,
    )
    val1 = buf1.getvalue()
    assert val1.startswith(b"%PDF")

    # Verify cached in Redis
    cached_val = cache.get(cache_key)
    assert cached_val is not None
    assert cached_val == val1

    # Second call hits cache
    buf2 = generate_badge_pdf(
        student_name="Test Student",
        date=date_str,
        student_id=student_id,
    )
    assert buf2.getvalue() == val1


def test_generate_badge_png_uses_bundled_ttf_font():
    """Verify generate_badge_png loads bundled TTF font from src/assets/fonts/."""
    from pathlib import Path
    fonts_dir = Path(__file__).parent.parent.parent / "src" / "assets" / "fonts"
    assert (fonts_dir / "Roboto-Regular.ttf").exists() or (fonts_dir / "DejaVuSans.ttf").exists()

    buf = generate_badge_png(student_name="Bundled Font Test", student_id="test_font_123")
    val = buf.getvalue()
    assert val.startswith(b"\x89PNG")
    assert len(val) > 1000

