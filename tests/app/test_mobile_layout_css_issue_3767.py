"""Tests for mobile layout CSS (Issue #3767)."""

import os
import re

from app.css_constants import MOBILE_LAYOUT_CSS

_THEME_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "app", "theme.py")
)


def test_mobile_layout_css_media_query():
    assert "@media (max-width: 768px)" in MOBILE_LAYOUT_CSS
    assert "padding" in MOBILE_LAYOUT_CSS
    assert re.search(r"height:\s*\d+px", MOBILE_LAYOUT_CSS)


def test_theme_injects_mobile_layout_css():
    with open(_THEME_PATH, encoding="utf-8") as f:
        source = f.read()
    assert "MOBILE_LAYOUT_CSS" in source
