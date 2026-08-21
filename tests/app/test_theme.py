"""Tests for app/theme.py theming and styling utilities."""

from unittest.mock import patch

from app.theme import (
    COLORS,
    THEMES,
    badge_html,
    empty_state_html,
    format_similarity_html,
    get_chart_colors,
    get_colors,
    inject_css,
    pipeline_progress_html,
    sanitize_hex_color,
    severity_tier,
    sidebar_user_badge_html,
    tier_color,
    tier_from_severity_label,
)


def test_render_notification_badge_with_negative_count():
    from app.theme import render_notification_badge

    assert render_notification_badge(-1) == ""


def test_render_notification_badge_with_count():
    from app.theme import render_notification_badge

    badge = render_notification_badge(5)

    assert "5" in badge
    assert 'class="notification-badge"' in badge


def test_render_notification_badge_with_zero_count():
    from app.theme import render_notification_badge

    assert render_notification_badge(0) == ""


def test_get_colors_returns_valid_theme_colors():
    colors = get_colors()

    assert isinstance(colors, dict)
    assert colors
    assert "background" in colors
    assert "accent" in colors


def test_themes_have_expected_keys():
    """Verify both Light and Dark themes have all expected color keys."""
    required_keys = [
        "background",
        "surface",
        "card",
        "ink",
        "muted",
        "accent",
        "border",
        "input",
        "danger",
        "danger_soft",
        "warning",
        "warning_soft",
        "success",
        "success_soft",
        "neutral_soft",
    ]
    for theme_name, theme in THEMES.items():
        assert theme_name in ["Light", "Dark"]
        for key in required_keys:
            assert key in theme, f"Theme {theme_name} missing key: {key}"


def test_default_colors():
    """Verify default COLORS matches Light theme."""
    assert COLORS == THEMES["Light"]


def test_get_chart_colors_matches_active_theme_when_override_disabled():
    """Without the 'Force Dark Mode Charts' override, chart colors follow the app theme."""
    mock_state: dict = {"force_dark_charts": False}
    with patch("app.theme.st.session_state", mock_state):
        with patch("app.theme.get_colors", return_value=THEMES["Light"]):
            assert get_chart_colors() == THEMES["Light"]


def test_get_chart_colors_forces_dark_when_override_enabled():
    """With 'Force Dark Mode Charts' enabled, chart colors are Dark regardless of app theme."""
    mock_state: dict = {"force_dark_charts": True}
    with patch("app.theme.st.session_state", mock_state):
        with patch("app.theme.get_colors", return_value=THEMES["Light"]):
            assert get_chart_colors() == THEMES["Dark"]


def test_get_chart_colors_defaults_to_active_theme_when_key_absent():
    """If the override key was never set (widget not yet rendered), fall back to get_colors()."""
    mock_state: dict = {}
    with patch("app.theme.st.session_state", mock_state):
        with patch("app.theme.get_colors", return_value=THEMES["Dark"]):
            assert get_chart_colors() == THEMES["Dark"]


def test_get_chart_colors_defaults_to_light_active_theme_when_key_absent():
    """Missing override key follows an active Light theme."""
    mock_state: dict = {}
    with patch("app.theme.st.session_state", mock_state):
        with patch("app.theme.get_colors", return_value=THEMES["Light"]):
            assert get_chart_colors() == THEMES["Light"]


def test_get_chart_colors_does_not_mutate_session_state_when_key_absent():
    """Reading chart colors without the override does not create session state."""
    mock_state: dict = {}
    with patch("app.theme.st.session_state", mock_state):
        with patch("app.theme.get_colors", return_value=THEMES["Dark"]):
            get_chart_colors()

    assert mock_state == {}


def test_get_chart_colors_explicit_false_still_uses_active_theme():
    """An explicit disabled override follows the active theme rather than forcing Dark."""
    mock_state: dict = {"force_dark_charts": False}
    with patch("app.theme.st.session_state", mock_state):
        with patch("app.theme.get_colors", return_value=THEMES["Dark"]):
            assert get_chart_colors() == THEMES["Dark"]


def test_severity_tier_high():
    """Test high severity tier detection."""
    assert severity_tier(0.95, 0.59) == "high"
    assert severity_tier(0.90, 0.59) == "high"
    assert severity_tier(1.0, 0.59) == "high"


def test_severity_tier_medium():
    """Test medium severity tier detection."""
    assert severity_tier(0.85, 0.59) == "medium"
    assert severity_tier(0.75, 0.59) == "medium"
    assert severity_tier(0.59, 0.59) == "medium"


def test_severity_tier_low():
    """Test low severity tier detection."""
    assert severity_tier(0.50, 0.59) == "low"
    assert severity_tier(0.00, 0.59) == "low"
    assert severity_tier(0.58, 0.59) == "low"


def test_severity_tier_with_higher_threshold():
    """Test severity with different threshold."""
    assert severity_tier(0.76, 0.75) == "medium"
    assert severity_tier(0.75, 0.75) == "medium"
    assert severity_tier(0.74, 0.75) == "low"
    assert severity_tier(0.50, 0.75) == "low"


def test_tier_from_severity_label():
    """Test mapping severity labels to tier keys."""
    assert tier_from_severity_label("🔴 High") == "high"
    assert tier_from_severity_label("🟡 Medium") == "medium"
    assert tier_from_severity_label("HIGH") == "high"
    assert tier_from_severity_label("Warning") == "medium"
    assert tier_from_severity_label("Low") == "low"
    assert tier_from_severity_label("unknown") == "low"
    assert tier_from_severity_label("low") == "low"


def test_tier_color():
    """Test color mapping for severity tiers."""
    assert tier_color("high") == COLORS["danger"]
    assert tier_color("medium") == COLORS["warning"]
    assert tier_color("low") == COLORS["success"]
    assert tier_color("unknown") == COLORS["neutral_soft"]

    # Case-insensitive checks
    assert tier_color("High") == COLORS["danger"]
    assert tier_color("Low") == COLORS["success"]

    # Fallback checks for invalid strings / None
    assert tier_color("invalid") == COLORS["neutral_soft"]
    assert tier_color("") == COLORS["neutral_soft"]
    assert tier_color(None) == COLORS["neutral_soft"]


def test_badge_html_default():
    """Test badge HTML generation with default label."""
    html = badge_html("high")
    assert 'class="badge"' in html
    assert f"background-color: {COLORS['danger_soft']}" in html
    assert f"color: {COLORS['danger']}" in html
    assert "border: 1px solid" in html
    assert COLORS["danger"] in html
    assert "🔴 High" in html

    html_med = badge_html("medium")
    assert f"background-color: {COLORS['warning_soft']}" in html_med
    assert f"color: {COLORS['warning']}" in html_med

    html_low = badge_html("low")
    assert f"background-color: {COLORS['success_soft']}" in html_low
    assert f"color: {COLORS['success']}" in html_low
    assert "🟢 Low" in html_low


def test_inject_css_generates_css_without_errors():
    with patch("app.theme.st.markdown") as mock_markdown:
        inject_css()

    assert mock_markdown.call_count == 3

    css = mock_markdown.call_args_list[0].args[0]

    assert isinstance(css, str)
    assert len(css.strip()) > 0
    assert "block-container" in css
    assert "stAlert" in css


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


def test_badge_html_custom_label():
    """Test badge HTML with custom label."""
    custom_label = "Custom Label"
    html = badge_html("high", custom_label)
    assert custom_label in html
    assert "🔴 High" not in html


def test_format_similarity_html():
    """Test similarity pill HTML generation."""
    high_html = format_similarity_html(0.95)
    assert 'class="sim-pill"' in high_html
    assert f"background-color: {COLORS['danger']};" in high_html
    assert "Similarity: 95.0%" in high_html

    med_html = format_similarity_html(0.80)
    assert f"background-color: {COLORS['warning']};" in med_html
    assert "Similarity: 80.0%" in med_html

    low_html = format_similarity_html(0.50)
    assert f"background-color: {COLORS['success']};" in low_html
    assert "Similarity: 50.0%" in low_html


def test_format_similarity_html_custom_threshold():
    """Test similarity pill with custom threshold."""
    html = format_similarity_html(0.70, threshold=0.75)
    assert f"background-color: {COLORS['success']};" in html


def test_empty_state_html():
    """Test empty state HTML generation."""
    html = empty_state_html("📁", "No Files", "Please upload files to continue.")
    assert 'class="empty-state"' in html
    assert 'class="empty-icon"' in html
    assert "📁" in html
    assert 'class="empty-title"' in html
    assert "No Files" in html
    assert 'class="empty-desc"' in html
    assert "Please upload files to continue." in html


def test_sidebar_user_badge_html():
    """Test sidebar user badge HTML generation."""
    html = sidebar_user_badge_html("testuser", "admin")
    assert 'class="sidebar-user-badge"' in html
    assert 'class="avatar"' in html
    assert "T" in html
    assert "testuser" in html
    assert "ADMIN" in html


def test_sidebar_user_badge_html_empty_username():
    """Test sidebar user badge with empty username."""
    html = sidebar_user_badge_html("", "user")
    assert "?" in html


def test_pipeline_progress_html_all_pending():
    """Test pipeline progress with all steps pending."""
    steps = ["Extract", "Chunk", "Embed"]
    html = pipeline_progress_html(steps)
    assert 'class="pipeline-steps"' in html
    assert 'class="pipeline-step"' in html
    assert "Extract" in html
    assert "Chunk" in html
    assert "Embed" in html
    assert "→" in html


def test_pipeline_progress_html_with_done_steps():
    """Test pipeline progress with completed steps."""
    steps = ["Extract", "Chunk", "Embed", "Flag"]
    html = pipeline_progress_html(steps, active_index=1)
    assert 'class="pipeline-step done"' in html
    assert 'class="pipeline-step active"' in html
    assert "✓ Extract" in html
    assert "Chunk" in html
    assert "→" in html


def test_pipeline_progress_html_with_active_and_done():
    """Test pipeline progress with active and completed steps."""
    steps = ["Extract", "Chunk", "Embed"]
    html = pipeline_progress_html(steps, active_index=1)
    assert 'class="pipeline-step active"' in html
    assert "✓ Extract" in html
    assert "Chunk" in html


def test_set_theme_updates_query_params():
    mock_query_params = {}
    st.session_state.clear()
    with patch("app.theme.st.query_params", mock_query_params):
        set_theme("Dark")
    assert mock_query_params["theme"] == "dark"
    assert st.session_state.theme == "Dark"


import matplotlib as mpl

from app.theme import (
    apply_matplotlib_theme,
)


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


def test_apply_matplotlib_theme():
    """Verify apply_matplotlib_theme updates Matplotlib rcParams correctly."""
    custom_theme = {
        "background": "#112233",
        "surface": "#223344",
        "border": "#334455",
        "ink": "#ffffff",
    }
    apply_matplotlib_theme(custom_theme)

    assert mpl.rcParams["figure.facecolor"] == "#112233"
    assert mpl.rcParams["axes.facecolor"] == "#223344"
    assert mpl.rcParams["axes.edgecolor"] == "#334455"
    assert mpl.rcParams["axes.labelcolor"] == "#ffffff"
    assert mpl.rcParams["xtick.color"] == "#ffffff"
    assert mpl.rcParams["ytick.color"] == "#ffffff"
    assert mpl.rcParams["text.color"] == "#ffffff"


# ── Issue #644: CSP Nonce Tests ───────────────────────────────────────────────
from app.theme import back_to_top_html, generate_csp_nonce, get_csp_nonce


def test_generate_csp_nonce_returns_unique_hex_strings():
    """generate_csp_nonce() should return a non-empty, unique 32-char hex string."""
    nonce1 = generate_csp_nonce()
    nonce2 = generate_csp_nonce()
    assert isinstance(nonce1, str)
    assert len(nonce1) == 32  # 16 bytes -> 32 hex chars
    assert nonce1 != nonce2  # cryptographically unique


def test_get_csp_nonce_persists_in_session_state():
    """get_csp_nonce() stores and reuses the nonce from st.session_state."""
    mock_state: dict = {}
    with patch("app.theme.st.session_state", mock_state):
        nonce = get_csp_nonce()
        assert "csp_nonce" in mock_state
        assert mock_state["csp_nonce"] == nonce
        # second call returns the cached nonce
        assert get_csp_nonce() == nonce


def test_inject_css_includes_csp_nonce():
    """inject_css() must attach nonce="..." to both the <style> and <script> blocks."""
    mock_state: dict = {}
    with patch("app.theme.st.session_state", mock_state):
        nonce = get_csp_nonce()  # prime the nonce in mock state
        with patch("app.theme.st.markdown") as mock_md:
            inject_css()

        assert mock_md.call_count == 3
        style_html = mock_md.call_args_list[0].args[0]
        script_html = mock_md.call_args_list[1].args[0]

        assert f'<style nonce="{nonce}">' in style_html
        assert f'<script nonce="{nonce}">' in script_html


def test_back_to_top_html_includes_csp_nonce():
    """back_to_top_html() must attach nonce="..." to its <script> block."""
    mock_state: dict = {}
    with patch("app.theme.st.session_state", mock_state):
        nonce = get_csp_nonce()
        html = back_to_top_html()
        assert f'<script nonce="{nonce}">' in html


def test_inject_css_contains_active_sidebar_tab_selector():
    """inject_css() must output CSS rules for active sidebar button tabs."""
    with patch("app.theme.st.markdown") as mock_md:
        inject_css()

    style_html = mock_md.call_args_list[0].args[0]
    assert '.stButton button[data-selected="true"]' in style_html
    assert 'section[data-testid="stSidebar"]' in style_html


def test_inject_css_contains_accent_border_left():
    """inject_css() must specify border-left: 4px solid #4f46e5 for active state (Issue #1028)."""
    with patch("app.theme.st.markdown") as mock_md:
        inject_css()

    style_html = mock_md.call_args_list[0].args[0]
    assert "border-left: 4px solid #4f46e5" in style_html


def test_inject_css_contains_high_severity_row_styling():
    """inject_css() must output CSS rules for .high-severity-row (Issue #1569)."""
    with patch("app.theme.st.markdown") as mock_md:
        inject_css()

    style_html = mock_md.call_args_list[0].args[0]
    assert ".high-severity-row" in style_html
    assert "border-left: 4px solid #ef4444" in style_html
    assert "background-color: rgba(239, 68, 68, 0.05)" in style_html


def test_inject_css_contains_low_confidence_card_styling():
    """inject_css() must output CSS rules for .low-confidence-card (Issue #1726)."""
    with patch("app.theme.st.markdown") as mock_md:
        inject_css()

    style_html = mock_md.call_args_list[0].args[0]
    assert ".low-confidence-card" in style_html
    assert "border-left: 4px solid #f59e0b" in style_html


def test_active_tab_border_style_default():
    from app.theme import active_tab_border_style

    css_decl = active_tab_border_style()
    assert css_decl == "border-left: 4px solid #4f46e5;"


def test_active_tab_border_style_custom_color_and_width():
    from app.theme import active_tab_border_style

    css_decl = active_tab_border_style(color="#0D9488", width=6)
    assert css_decl == "border-left: 6px solid #0D9488;"


def test_active_tab_border_style_invalid_color_fallback():
    from app.theme import active_tab_border_style

    css_decl = active_tab_border_style(color="invalid-color-value")
    assert css_decl == "border-left: 4px solid #4f46e5;"


def test_get_active_sidebar_tab_css():
    from app.theme import get_active_sidebar_tab_css

    css_block = get_active_sidebar_tab_css("#4f46e5")
    assert isinstance(css_block, str)
    assert "border-left: 4px solid #4f46e5" in css_block
    assert 'button[data-selected="true"]' in css_block


def test_get_sidebar_tab_style_selected():
    from app.theme import get_sidebar_tab_style

    selected_style = get_sidebar_tab_style(
        is_selected=True, accent_border_color="#4f46e5"
    )
    assert selected_style["border-left"] == "4px solid #4f46e5"
    assert selected_style["font-weight"] == "700"


def test_get_sidebar_tab_style_unselected():
    from app.theme import get_sidebar_tab_style

    unselected_style = get_sidebar_tab_style(is_selected=False)
    assert unselected_style["border-left"] == "4px solid transparent"
    assert unselected_style["font-weight"] == "400"


def test_get_theme_accent_color():
    from app.theme import get_theme_accent_color

    assert get_theme_accent_color("Indigo") == "#4f46e5"
    assert get_theme_accent_color("Teal") == "#0d9488"
    assert get_theme_accent_color("Light") == "#0D9488"
    assert get_theme_accent_color("Dark") == "#2DD4BF"


def test_build_active_tab_custom_css():
    from app.theme import build_active_tab_custom_css

    css = build_active_tab_custom_css(accent_hex="#4f46e5", border_width=4)
    assert "border-left: 4px solid #4f46e5 !important" in css
    assert 'button[data-selected="true"]' in css


def test_generate_active_tab_theme_tokens():
    from app.theme import generate_active_tab_theme_tokens

    tokens = generate_active_tab_theme_tokens("Light")
    assert isinstance(tokens, dict)
    assert tokens["active_border_color"] == "#4f46e5"
    assert tokens["active_border_width"] == "4px"
    assert tokens["active_font_weight"] == "700"


def test_get_sidebar_navigation_config():
    from app.theme import get_sidebar_navigation_config

    config = get_sidebar_navigation_config()
    assert isinstance(config, dict)
    assert config["accent_border_color"] == "#4f46e5"
    assert config["accent_border_width_px"] == 4
    assert len(config["supported_selectors"]) >= 4


def test_render_active_tab_badge_html():
    from app.theme import render_active_tab_badge_html

    active_html = render_active_tab_badge_html("Dashboard", is_active=True)
    assert "border-left: 4px solid #4f46e5" in active_html
    assert "Dashboard" in active_html

    inactive_html = render_active_tab_badge_html("Settings", is_active=False)
    assert "border-left: 4px solid transparent" in inactive_html
    assert "Settings" in inactive_html


def test_generate_sidebar_theme_stylesheet():
    from app.theme import generate_sidebar_theme_stylesheet

    stylesheet = generate_sidebar_theme_stylesheet("Modern", "#4f46e5")
    assert "border-left: 4px solid #4f46e5" in stylesheet
    assert 'button[data-selected="true"]' in stylesheet


def test_get_active_tab_accessibility_attributes():
    from app.theme import get_active_tab_accessibility_attributes

    active_attrs = get_active_tab_accessibility_attributes(is_active=True)
    assert active_attrs["aria-selected"] == "true"
    assert active_attrs["data-selected"] == "true"
    assert active_attrs["role"] == "tab"

    inactive_attrs = get_active_tab_accessibility_attributes(is_active=False)
    assert inactive_attrs["aria-selected"] == "false"
    assert inactive_attrs["data-selected"] == "false"


def test_render_sidebar_navigation_menu():
    from app.theme import render_sidebar_navigation_menu

    tabs = [("home", "Home"), ("dashboard", "Dashboard"), ("settings", "Settings")]
    menu_html = render_sidebar_navigation_menu(tabs, active_tab_id="dashboard")
    assert 'class="sidebar-nav-menu"' in menu_html
    assert 'data-tab-id="dashboard"' in menu_html
    assert "border-left: 4px solid #4f46e5" in menu_html


def test_css_variables_injected():
    """Test that :root and CSS variables are correctly defined and referenced."""
    with patch("app.theme.st.markdown") as mock_markdown:
        inject_css()

    css = mock_markdown.call_args_list[0].args[0]

    # Check :root existence
    assert ":root" in css, "Missing :root block"

    # Check variable declarations
    assert "--primary-bg:" in css, "Missing --primary-bg declaration"
    assert "--text-color:" in css, "Missing --text-color declaration"

    # Verify values are valid hex colors
    import re

    assert re.search(r"--primary-bg:\s*#[0-9a-fA-F]+", css), (
        "--primary-bg does not have a valid hex value"
    )
    assert re.search(r"--text-color:\s*#[0-9a-fA-F]+", css), (
        "--text-color does not have a valid hex value"
    )

    # Verify component CSS uses var() instead of hardcoded
    assert "background-color: var(--primary-bg)" in css
    assert "color: var(--text-color)" in css
    assert "background-color: var(--secondary-bg)" in css
    assert "border: 1px solid var(--border-color)" in css
    assert "var(--accent-color)" in css


def test_render_session_status_banner():
    """Verify that render_session_status_banner sets session start time and displays banner."""
    import time

    from app.theme import render_session_status_banner

    mock_state = {}

    with (
        patch("app.theme.st.session_state", mock_state),
        patch("app.theme.st.caption") as mock_caption,
    ):
        # First call: should initialize session_start_time and render 0 mins
        render_session_status_banner()
        assert "session_start_time" in mock_state
        mock_caption.assert_called_once_with("Active Session: 0 mins")

    # Second test: with established session start time in the past
    mock_state_past = {"session_start_time": time.time() - 45.2 * 60}
    with (
        patch("app.theme.st.session_state", mock_state_past),
        patch("app.theme.st.caption") as mock_caption_past,
    ):
        render_session_status_banner()
        mock_caption_past.assert_called_once_with("Active Session: 45 mins")


# ==============================================================================
# Issue #2353: severity_tier threshold boundary tests
# ==============================================================================


def test_severity_tier_boundary_just_below_medium():
    """A score of 0.49 is classified as low."""
    assert severity_tier(0.49, 0.50) == "low"


def test_severity_tier_boundary_at_medium():
    """A score of 0.50 is classified as medium."""
    assert severity_tier(0.50, 0.50) == "medium"


def test_severity_tier_boundary_just_below_high():
    """A score of 0.79 remains medium."""
    assert severity_tier(0.79, 0.50) == "medium"


def test_severity_tier_boundary_at_high():
    """A score of 0.80 is classified as high."""
    assert severity_tier(0.80, 0.50) == "high"


# sanitize_hex_color edge-case tests (Issue #2352)
# ==============================================================================


def test_sanitize_hex_color_valid():
    """Valid six-digit uppercase hex colors are returned unchanged."""
    assert sanitize_hex_color("#FF0000") == "#FF0000"


def test_sanitize_hex_color_missing_hash():
    """Hex values without the leading hash fall back to the default."""
    assert sanitize_hex_color("FF0000") == "#000000"


def test_sanitize_hex_color_invalid():
    """Invalid/non-hex values use the configured fallback."""
    assert sanitize_hex_color("not-a-color", fallback="#FFFFFF") == "#FFFFFF"
