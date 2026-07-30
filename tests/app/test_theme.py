from unittest.mock import patch

from app.theme import badge_html, get_colors, inject_css, sanitize_hex_color


def test_get_colors_returns_valid_theme_colors():
    colors = get_colors()

    assert isinstance(colors, dict)
    assert colors
    assert "background" in colors
    assert "accent" in colors
from app.theme import (COLORS, severity_tier, tier_color,
                       tier_from_severity_label)


def test_severity_tier():
    # Test with threshold 0.75
    assert severity_tier(0.95, 0.75) == "high"
    assert severity_tier(0.90, 0.75) == "high"
    assert severity_tier(0.85, 0.75) == "medium"
    assert severity_tier(0.75, 0.75) == "medium"
    assert severity_tier(0.70, 0.75) == "low"
    assert severity_tier(0.00, 0.75) == "low"

    # Test with threshold 0.59
    assert severity_tier(0.65, 0.59) == "medium"
    assert severity_tier(0.59, 0.59) == "medium"
    assert severity_tier(0.58, 0.59) == "low"


def test_tier_from_severity_label():
    assert tier_from_severity_label("🔴 High") == "high"
    assert tier_from_severity_label("🟡 Medium") == "medium"
    assert tier_from_severity_label("HIGH") == "high"
    assert tier_from_severity_label("Warning") == "medium"
    assert tier_from_severity_label("Low") == "low"
    assert tier_from_severity_label("unknown") == "low"


def test_tier_color():
    assert tier_color("high") == COLORS["danger"]
    assert tier_color("medium") == COLORS["warning"]
    assert tier_color("low") == COLORS["success"]
    assert tier_color("unknown") == COLORS["neutral_soft"]


def test_badge_html_default():
    html = badge_html("high")
    assert "background-color: " + COLORS["danger_soft"] in html
    assert "color: " + COLORS["danger"] in html
    assert "🔴 High" in html


def test_inject_css_generates_css_without_errors():
    with patch("app.theme.st.markdown") as mock_markdown:
        inject_css()

    mock_markdown.assert_called_once()

    css = mock_markdown.call_args.args[0]

    assert isinstance(css, str)
    assert len(css.strip()) > 0
    assert "<style>" in css




def test_sanitize_hex_color_valid_and_invalid():
    """Verify regex validation for hex colors."""
    # Valid hex colors (3 and 6 digits)
    assert sanitize_hex_color("#FFF") == "#FFF"
    assert sanitize_hex_color("#123456") == "#123456"
    assert sanitize_hex_color("#aBcDeF") == "#aBcDeF"

    # Invalid hex colors / injection attempts
    assert sanitize_hex_color("red", fallback="#000000") == "#000000"
    assert sanitize_hex_color("#12345", fallback="#000000") == "#000000"
    assert sanitize_hex_color("#1234567", fallback="#000000") == "#000000"
    assert sanitize_hex_color("url('http://evil')", fallback="#000000") == "#000000"
    assert sanitize_hex_color("; background: red;", fallback="#000000") == "#000000"


def test_badge_html_returns_valid_html():
    html = badge_html("high")

    assert isinstance(html, str)
    assert len(html.strip()) > 0
    assert "badge" in html


import streamlit as st
from unittest.mock import patch
from app.theme import initialize_theme, set_theme

def test_initialize_theme_loads_dark_from_query_params():
    st.session_state.clear()
    with patch("app.theme.st.query_params", {"theme": "dark"}):
        initialize_theme()
    assert st.session_state.theme == "Dark"

def test_initialize_theme_loads_light_from_query_params():
    st.session_state.clear()
    with patch("app.theme.st.query_params", {"theme": "light"}):
        initialize_theme()
    assert st.session_state.theme == "Light"

def test_initialize_theme_invalid_query_params_fallback():
    st.session_state.clear()
    with patch("app.theme.st.query_params", {"theme": "invalid_value"}):
        initialize_theme()
    assert st.session_state.theme == "Light"

def test_set_theme_updates_query_params():
    mock_query_params = {}
    st.session_state.clear()
    with patch("app.theme.st.query_params", mock_query_params):
        set_theme("Dark")
    assert mock_query_params["theme"] == "dark"
    assert st.session_state.theme == "Dark"