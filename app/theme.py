from __future__ import annotations

# -*- coding: utf-8 -*-

from app.css_constants import (
    CLASS_AVATAR,
    CLASS_BADGE,
    CLASS_EMPTY_DESC,
    CLASS_EMPTY_ICON,
    CLASS_EMPTY_STATE,
    CLASS_EMPTY_TITLE,
    CLASS_PIPELINE_ACTIVE,
    CLASS_PIPELINE_ARROW,
    CLASS_PIPELINE_DONE,
    CLASS_PIPELINE_ETA,
    CLASS_PIPELINE_STEP,
    CLASS_PIPELINE_STEPS,
    CLASS_SIDEBAR_USER_BADGE,
    CLASS_SIM_PILL,
    CLASS_WELCOME_BANNER,
)

"""
theme.py
--------
Centralized theme management and CSS injection for the Semantic Plagiarism Detector.

This module defines the color palettes for Light and Dark modes, provides
utilities for sanitizing hex colors, and injects global CSS to ensure a
cohesive, theme-aware user experience across all Streamlit components.

Recent Additions (Issue #572):
- Added comprehensive CSS rules targeting Streamlit's `.stFileUploader`
  dropzone borders, background, and hover states to match the active theme tokens.
"""

import re
import secrets
from datetime import datetime, timezone
import streamlit as st
from typing import Any
from src.core.config import DEFAULT_THRESHOLDS, normalize_severity_label, severity_key


# ── CSP Nonce Generation (Issue #644) ──────────────────────────────────────────
def generate_csp_nonce(length: int = 16) -> str:
    """Generate a cryptographically secure random hex nonce for use in CSP headers."""
    return secrets.token_hex(length)


def get_csp_nonce() -> str:
    """
    Retrieve or create a per-session CSP nonce stored in st.session_state.

    Generates a new nonce on the first call each session and returns the
    cached value on subsequent calls, ensuring a consistent nonce is used
    across all inline <style> and <script> blocks rendered in one page load.
    """
    try:
        if isinstance(st.session_state, dict):
            # Dict-like mock used in unit tests
            if not st.session_state.get("csp_nonce"):
                st.session_state["csp_nonce"] = generate_csp_nonce()
            return st.session_state["csp_nonce"]
        if "csp_nonce" not in st.session_state or not st.session_state.csp_nonce:
            st.session_state.csp_nonce = generate_csp_nonce()
        return st.session_state.csp_nonce
    except Exception:
        return generate_csp_nonce()


# ── Matplotlib Theme Helper ────────────────────────────────────────────────────
def apply_matplotlib_theme(theme_colors: dict | None = None) -> None:
    """Apply the active theme colours to Matplotlib's global rcParams."""
    try:
        import matplotlib as mpl

        colors = theme_colors if theme_colors is not None else get_colors()
        mpl.rcParams["figure.facecolor"] = colors.get("background", "#FFFFFF")
        mpl.rcParams["axes.facecolor"] = colors.get("surface", "#F8FAFC")
        mpl.rcParams["axes.edgecolor"] = colors.get("border", "#E2E8F0")
        mpl.rcParams["axes.labelcolor"] = colors.get("ink", "#0F172A")
        mpl.rcParams["xtick.color"] = colors.get("ink", "#0F172A")
        mpl.rcParams["ytick.color"] = colors.get("ink", "#0F172A")
        mpl.rcParams["text.color"] = colors.get("ink", "#0F172A")
    except Exception:
        pass


# ── Validation Patterns ────────────────────────────────────────────────────────
HEX_COLOR_PATTERN = re.compile(r"^#(?:[0-9a-fA-F]{3}){1,2}$")


def sanitize_hex_color(color_val: str, fallback: str = "#000000") -> str:
    """
    Validates and sanitizes a hex color string against ^#(?:[0-9a-fA-F]{3}){1,2}$.
    Returns fallback if invalid.
    """
    if isinstance(color_val, str) and HEX_COLOR_PATTERN.match(color_val.strip()):
        return color_val.strip()
    return fallback


def sanitize_theme_colors(colors: dict) -> dict:
    """Sanitize all color values in a theme dictionary to ensure CSS safety."""
    sanitized = {}
    fallback_map = {
        "background": "#FFFFFF",
        "surface": "#F8FAFC",
        "card": "#FFFFFF",
        "ink": "#0F172A",
        "muted": "#64748B",
        "accent": "#0D9488",
        "border": "#E2E8F0",
        "input": "#FFFFFF",
        "neutral_soft": "#F1F5F9",
        "danger": "#FF4B4B",
        "danger_soft": "#FEE2E2",
        "warning": "#FFA500",
        "warning_soft": "#FEF3C7",
        "success": "#21C55D",
        "success_soft": "#DCFCE7",
    }
    for k, v in colors.items():
        fallback = fallback_map.get(k, "#000000")
        sanitized[k] = sanitize_hex_color(str(v), fallback=fallback)
    return sanitized


# ── Theme Definitions ──────────────────────────────────────────────────────────
THEMES = {
    "Light": {
        "background": "#FFFFFF",
        "surface": "#F8FAFC",
        "card": "#FFFFFF",
        "ink": "#0F172A",
        "muted": "#64748B",
        "accent": "#0D9488",
        "border": "#E2E8F0",
        "input": "#FFFFFF",
        "danger": "#FF4B4B",
        "danger_soft": "#FEE2E2",
        "warning": "#FFA500",
        "warning_soft": "#FEF3C7",
        "success": "#21C55D",
        "success_soft": "#DCFCE7",
        "neutral_soft": "#F1F5F9",
    },
    "Dark": {
        "background": "#0E1117",
        "surface": "#161B22",
        "card": "#1F2937",
        "ink": "#F8FAFC",
        "muted": "#CBD5E1",
        "accent": "#2DD4BF",
        "border": "#374151",
        "input": "#111827",
        "danger": "#F87171",
        "danger_soft": "#450A0A",
        "warning": "#FBBF24",
        "warning_soft": "#422006",
        "success": "#4ADE80",
        "success_soft": "#052E16",
        "neutral_soft": "#1E293B",
    },
}

# Backward-compatible default palette used by existing tests and callers.
COLORS = THEMES["Light"]

# ── Colormap Mappings & Constants ──────────────────────────────────────────────
UI_COLORMAP_OPTIONS = [
    "Viridis",
    "Cividis",
    "Plasma",
    "Coolwarm",
    "YlOrRd",
]

MATPLOTLIB_CMAP_MAPPING: dict[str, str] = {
    "Viridis": "viridis",
    "Cividis": "cividis",
    "Plasma": "plasma",
    "Coolwarm": "coolwarm",
    "YlOrRd": "YlOrRd",
    "Legacy Red/Green": "RdYlGn_r",
}

PLOTLY_CMAP_MAPPING = {
    "Viridis": "Viridis",
    "Cividis": "Cividis",
    "Plasma": "Plasma",
    "Coolwarm": "RdBu_r",
    "YlOrRd": "YlOrRd",
    "Legacy Red/Green": "RdYlGn_r",
}

DEFAULT_UI_COLORMAP: str = "Viridis"


def initialize_theme() -> None:
    """Initialize the active theme for the current session."""
    try:
        if "theme" not in st.session_state:
            query_theme = st.query_params.get("theme")
            if query_theme and query_theme.lower() == "dark":
                st.session_state.theme = "Dark"
            elif query_theme and query_theme.lower() == "light":
                st.session_state.theme = "Light"
            else:
                st.session_state.theme = "Light"

        if "theme_colors" not in st.session_state:
            st.session_state.theme_colors = THEMES[st.session_state.theme]
    except Exception:
        pass


def get_theme_name() -> str:
    """Return the active theme name."""
    initialize_theme()
    try:
        return st.session_state.theme
    except Exception:
        return "Light"


def set_theme(theme_name: str) -> None:
    """Set the active theme."""
    if theme_name in THEMES:
        try:
            st.session_state.theme = theme_name
            st.session_state.theme_colors = THEMES[theme_name]
            st.query_params["theme"] = theme_name.lower()
        except Exception:
            pass


def get_colors() -> dict:
    """Return the colors for the active theme."""
    initialize_theme()
    try:
        return st.session_state.theme_colors
    except Exception:
        return THEMES["Light"]


def get_chart_colors() -> dict:
    """Return the color palette Plotly chart builders should use.

    Normally mirrors the app's active Light/Dark theme (via get_colors()).
    If the user has enabled "Force Dark Mode Charts" in Settings, this
    returns the Dark palette regardless of the app's overall theme, so
    charts can be forced dark independently of the Streamlit UI theme.

    Note: "force_dark_charts" must match SessionKeys.FORCE_DARK_CHARTS
    (app/session_keys.py) — not imported directly here to avoid a
    circular import between app.theme and app.session_keys.
    """
    try:
        if st.session_state.get("force_dark_charts", False):
            return THEMES["Dark"]
    except Exception:
        pass
    return get_colors()


def inject_css() -> None:
    """
    Inject CSS for the currently selected Light or Dark theme.

    Includes comprehensive styling for file uploaders, empty states,
    pipeline indicators, and severity badges to ensure a cohesive UI.
    """
    colors = sanitize_theme_colors(get_colors())

    main_css = f"""
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

        :root {{
            --primary-bg: {colors["background"]};
            --secondary-bg: {colors["surface"]};
            --text-color: {colors["ink"]};
            --secondary-text-color: {colors["muted"]};
            --border-color: {colors["border"]};
            --accent-color: {colors["accent"]};
            --background: {colors["background"]};
            --surface: {colors["surface"]};
            --card: {colors["card"]};
            --ink: {colors["ink"]};
            --muted: {colors["muted"]};
            --accent: {colors["accent"]};
            --border: {colors["border"]};
            --input: {colors["input"]};
            --neutral-soft: {colors["neutral_soft"]};
            --danger: {colors["danger"]};
            --danger-soft: {colors["danger_soft"]};
            --warning: {colors["warning"]};
            --warning-soft: {colors["warning_soft"]};
            --success: {colors["success"]};
            --success-soft: {colors["success_soft"]};
        }}

        html,
        body,
        [class*="css"] {{
            font-family: 'Inter', sans-serif !important;
        }}

        .stApp {{
            background-color: var(--primary-bg) !important;
            color: var(--text-color) !important;
        }}

        [data-testid="stHeader"] {{
            background-color: var(--primary-bg) !important;
        }}

        [data-testid="stToolbar"] {{
            color: var(--text-color) !important;
        }}

        h1,
        h2,
        h3,
        h4,
        h5,
        h6 {{
            font-family: 'Newsreader', Georgia, serif !important;
            color: var(--text-color) !important;
            font-weight: 700 !important;
        }}

        p,
        label,
        span,
        li,
        [data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"] {{
            color: var(--text-color);
        }}

        [data-testid="stCaptionContainer"],
        .stCaption {{
            color: var(--secondary-text-color) !important;
        }}

        .hero-kicker {{
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--accent-color);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.25rem;
        }}

        /* ── Sidebar ────────────────────────────────────────────────── */

        [data-testid="stSidebar"] {{
            background-color: var(--secondary-bg) !important;
            border-right: 1px solid var(--border-color) !important;
        }}

        [data-testid="stSidebar"] * {{
            color: var(--text-color);
        }}

        .sidebar-brand-title {{
            font-family: 'Newsreader', serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-color);
            text-align: center;
            line-height: 1.2;
            margin-top: 0.25rem;
            margin-bottom: 0;
        }}

        .sidebar-brand-kicker {{
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--accent-color);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            text-align: center;
            margin-bottom: 1.25rem;
        }}

        .sidebar-section-label {{
            font-family: 'Inter', sans-serif;
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--secondary-text-color);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 2px;
        }}

        .sidebar-user-badge {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 8px;
            background-color: var(--neutral-soft);
            border: 1px solid var(--border-color);
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-color);
            margin-bottom: 0.75rem;
        }}

        .sidebar-user-badge .avatar {{
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background-color: var(--accent-color);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 700;
            flex-shrink: 0;
        }}

        /* ── Document row (sidebar) ─────────────────────────────────── */

        .doc-row {{
            border-radius: 8px;
            padding: 4px 8px;
            margin-bottom: 2px;
            transition: background-color 0.18s ease, box-shadow 0.18s ease;
            cursor: default;
        }}

        .doc-row:hover {{
            background-color: var(--neutral-soft);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
        }}

        /* ── Metric cards ───────────────────────────────────────────── */

        div[data-testid="stMetric"] {{
            background-color: var(--card) !important;
            border: 1px solid var(--border-color) !important;
            border-top: 4px solid var(--accent-color) !important;
            border-radius: 8px !important;
            padding: 14px 16px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12) !important;
        }}

        div[data-testid="stMetricLabel"] > div {{
            font-family: 'Inter', sans-serif !important;
            font-size: 0.75rem !important;
            font-weight: 700 !important;
            color: var(--secondary-text-color) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }}

        div[data-testid="stMetricValue"] > div {{
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            color: var(--text-color) !important;
        }}

        div[data-testid="stMetricDelta"] > div {{
            font-family: 'Inter', sans-serif !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
        }}

        /* ── Badge ──────────────────────────────────────────────────── */

        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 700;
            font-family: 'IBM Plex Mono', monospace;
            text-align: center;
        }}

        .meta-chip {{
            background-color: var(--neutral-soft);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-color);
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .meta-chip code {{
            font-family: 'IBM Plex Mono', monospace !important;
            background: none !important;
            padding: 0 !important;
            color: var(--accent-color) !important;
            font-weight: 700 !important;
        }}

        /* ── Login container ────────────────────────────────────────── */

        .login-container {{
            background-color: var(--card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 2.5rem !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.18) !important;
            max-width: 480px;
            margin: 2rem auto;
            animation: loginSlideIn 0.4s ease-out;
        }}

        .login-container .login-header {{
            text-align: center;
            margin-bottom: 1.5rem;
        }}

        .login-container .login-icon {{
            font-size: 3rem;
            line-height: 1;
            margin-bottom: 0.5rem;
        }}

        .login-container .login-title {{
            font-family: 'Newsreader', serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 0.25rem;
        }}

        .login-container .login-subtitle {{
            font-size: 0.85rem;
            color: var(--secondary-text-color);
        }}

        .login-accent-bar {{
            height: 4px;
            background: linear-gradient(90deg, var(--accent-color), transparent);
            border-radius: 2px;
            margin-bottom: 1.5rem;
        }}

        @keyframes loginSlideIn {{
            from {{
                opacity: 0;
                transform: translateY(12px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        /* ── Warning card accent borders ────────────────────────────── */

        .warning-card-high {{
            border-left: 4px solid var(--danger) !important;
        }}

        .warning-card-medium {{
            border-left: 4px solid var(--warning) !important;
        }}

        .warning-card-low {{
            border-left: 4px solid var(--success) !important;
        }}

        /* ── High severity row accent border (Issue #1569) ───────────── */

        .high-severity-row {{
            border-left: 4px solid #ef4444 !important;
            background-color: rgba(239, 68, 68, 0.05) !important;
        }}

        /* ── Warning list container animation (#369) ─────────────────
           The threshold slider re-filters the warning list on every
           change. This transition smooths out the resulting layout /
           opacity shifts on the container instead of snapping instantly. */

        .st-key-warning_list_container {{
            transition: all 0.3s ease;
        }}

        /* ── Similarity score pill ──────────────────────────────────── */

        .sim-pill {{
            display: inline-block;
            padding: 3px 12px;
            border-radius: 10px;
            font-size: 0.85rem;
            font-weight: 700;
            font-family: 'IBM Plex Mono', monospace;
            color: white;
        }}

        /* ── Mono text ──────────────────────────────────────────────── */

        .mono-text {{
            font-family: 'IBM Plex Mono', monospace !important;
        }}

        /* ── Legend ─────────────────────────────────────────────────── */

        .legend-container {{
            display: flex;
            gap: 16px;
            align-items: center;
            margin-bottom: 1rem;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--secondary-text-color);
        }}

        .legend-item {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
            display: inline-block;
        }}

        /* ── Form inputs ────────────────────────────────────────────── */

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        [data-baseweb="select"] > div {{
            background-color: var(--input) !important;
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
        }}

        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [role="listbox"] {{
            background-color: var(--card) !important;
            color: var(--text-color) !important;
        }}

        .stButton button,
        .stDownloadButton button,
        .stFormSubmitButton button {{
            border-color: var(--border-color) !important;
        }}

        .clear-all-container button {{
            background-color: var(--danger) !important;
            color: white !important;
            border-color: var(--danger) !important;
            font-weight: 600 !important;
        }}

        .clear-all-container button:hover {{
            background-color: #ff3333 !important;
            color: white !important;
            border-color: #ff3333 !important;
        }}

        .{CLASS_WELCOME_BANNER} {{
            background-color: var(--secondary-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
            color: var(--text-color);
            font-size: 0.95rem;
        }}

        [data-testid="stExpander"],
        [data-testid="stForm"] {{
            background-color: var(--card) !important;
            border-color: var(--border-color) !important;
        }}

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {{
            border-color: var(--border-color) !important;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background-color: var(--secondary-bg) !important;
            border-color: var(--border-color) !important;
        }}

        /* ── Tabs ───────────────────────────────────────────────────── */

        [data-testid="stTabs"] button {{
            color: var(--secondary-text-color) !important;
        }}

        [data-testid="stTabs"] button[aria-selected="true"] {{
            color: var(--accent-color) !important;
            border-bottom-color: var(--accent-color) !important;
        }}

        hr {{
            border-color: var(--border-color) !important;
        }}

        /* ── Enhanced footer ────────────────────────────────────────── */

        .app-footer {{
            text-align: center;
            padding: 1rem 0 0.5rem;
            font-size: 0.78rem;
            color: var(--secondary-text-color);
        }}

        .app-footer a {{
            color: var(--accent-color);
            text-decoration: none;
        }}

        .app-footer a:hover {{
            text-decoration: underline;
        }}

        /* ── Empty state ────────────────────────────────────────────── */

        .empty-state {{
            text-align: center;
            padding: 2.5rem 1rem;
            color: var(--secondary-text-color);
        }}

        .empty-state .empty-icon {{
            font-size: 3rem;
            line-height: 1;
            margin-bottom: 0.75rem;
        }}

        .empty-state .empty-title {{
            font-family: 'Newsreader', serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 0.25rem;
        }}

        .empty-state .empty-desc {{
            font-size: 0.85rem;
            max-width: 400px;
            margin: 0 auto;
        }}

        /* ── Pipeline progress ──────────────────────────────────────── */

        .pipeline-steps {{
            display: flex;
            gap: 4px;
            align-items: center;
            justify-content: center;
            margin: 1rem 0;
            flex-wrap: wrap;
        }}

        .pipeline-step {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            background-color: var(--neutral-soft);
            color: var(--secondary-text-color);
            border: 1px solid var(--border-color);
        }}

        .pipeline-step.active {{
            background-color: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
            animation: pipelinePulse 1.2s ease-in-out infinite;
        }}

        .pipeline-step.done {{
            background-color: var(--success-soft);
            color: var(--success);
            border-color: var(--success);
        }}

        .pipeline-arrow {{
            color: var(--secondary-text-color);
            font-size: 0.7rem;
        }}

        @keyframes pipelinePulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}

        /* ── Back to Top Button ─────────────────────────────────────── */

        .sr-only {{
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }}

        #back-to-top-btn {{
            position: fixed;
            bottom: max(2rem, env(safe-area-inset-bottom, 2rem));
            right: max(2rem, env(safe-area-inset-right, 2rem));
            z-index: 9999;
            background-color: var(--accent-color);
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            opacity: 0;
            visibility: hidden;
            transform: translateY(12px);
            transition: opacity 0.3s ease, visibility 0.3s ease,
                        transform 0.3s ease, box-shadow 0.2s ease;
            display: flex;
            align-items: center;
            gap: 4px;
        }}

        #back-to-top-btn.visible {{
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
        }}

        #back-to-top-btn:hover {{
            filter: brightness(0.85);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
            transform: translateY(-2px);
        }}

        #back-to-top-btn:focus-visible {{
            outline: 2px solid var(--accent-color);
            outline-offset: 2px;
        }}

        @media (prefers-reduced-motion: reduce) {{
            #back-to-top-btn {{
                transition: opacity 0.15s ease, visibility 0.15s ease;
            }}

            #back-to-top-btn.visible,
            #back-to-top-btn:hover {{
                transform: none;
            }}
        }}

        /* ── Responsive: mobile / tablet ────────────────────────────── */

        @media (max-width: 768px) {{
            .login-container {{
                padding: 1.5rem !important;
                margin: 1rem auto;
            }}

            .sidebar-brand-title {{
                font-size: 1.25rem;
            }}

            div[data-testid="stMetricValue"] > div {{
                font-size: 1.3rem !important;
            }}

            /* Issue #258: when the sidebar is opened on a phone/small
               tablet, keep it from covering the whole screen so the
               similarity matrix / heatmap stay legible behind it. */
            [data-testid="stSidebar"] {{
                min-width: 85vw !important;
                max-width: 85vw !important;
            }}
        }}
    """
    # Issue #572: File Uploader Drag-Zone Customization
    file_uploader_css = f"""
    /* File Uploader Drag-Zone Customization */
    .stFileUploader [data-testid="stFileUploaderDropzone"] {{
        border: 2px dashed var(--border-color) !important;
        border-radius: 8px !important;
        background-color: var(--secondary-bg) !important;
        transition: all 0.2s ease-in-out !important;
        padding: 1.5rem !important;
    }}

    .stFileUploader [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: var(--accent-color) !important;
        background-color: {colors['neutral_soft']} !important;
        cursor: pointer !important;
    }}

    .stFileUploader [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderInstruction"] {{
        color: var(--secondary-text-color) !important;
        font-weight: 500 !important;
    }}

    .stFileUploader [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderBrowseFiles"] {{
        background-color: var(--accent-color) !important;
        color: #FFFFFF !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        transition: background-color 0.2s ease !important;
    }}

    .stFileUploader [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderBrowseFiles"]:hover {{
        background-color: var(--text-color) !important;
    }}
    """

    # Issue #1028: Active Sidebar Tab Accent Border Styling
    sidebar_active_tab_css = """
    /* Active Sidebar Navigation Tab Highlight (Issue #1028) */
    section[data-testid="stSidebar"] .stButton button[data-selected="true"],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"][aria-selected="true"],
    section[data-testid="stSidebar"] button[aria-selected="true"],
    section[data-testid="stSidebar"] .stButton button.st-active,
    .stButton button[data-selected="true"] {
        border-left: 4px solid #4f46e5 !important;
        background-color: var(--neutral-soft) !important;
        color: var(--accent-color) !important;
        font-weight: 700 !important;
        border-top-left-radius: 0 !important;
        border-bottom-left-radius: 0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        transition: border-left-color 0.2s ease, background-color 0.2s ease, color 0.2s ease !important;
    }

    section[data-testid="stSidebar"] .stButton button[data-selected="true"]:hover,
    .stButton button[data-selected="true"]:hover {
        border-left: 4px solid #4f46e5 !important;
        background-color: var(--secondary-bg) !important;
    }

    section[data-testid="stSidebar"] .stButton button:hover {
        border-left: 4px solid #4f46e5;
        transition: border-left 0.2s ease !important;
    }
    """

    base_css = f"""
    /* Global Theme Overrides */
    .stApp {{
        background-color: var(--primary-bg) !important;
        color: var(--text-color) !important;
    }}

    .block-container {{
        padding-top: 2rem !important;
    }}

    .stAlert {{
        border-radius: 8px !important;
    }}

    .stCard {{
        background-color: var(--card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }}

    /* Empty State Styling */
    .{CLASS_EMPTY_STATE} {{
        text-align: center;
        padding: 2rem;
        background-color: var(--secondary-bg);
        border-radius: 8px;
        border: 1px dashed var(--border-color);
    }}

    .{CLASS_EMPTY_ICON} {{
        font-size: 3rem;
        margin-bottom: 1rem;
        color: var(--secondary-text-color);
    }}

    .{CLASS_EMPTY_TITLE} {{
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 0.5rem;
    }}

    .{CLASS_EMPTY_DESC} {{
        color: var(--secondary-text-color);
        font-size: 0.95rem;
    }}

    /* Pipeline Progress Styling */
    .{CLASS_PIPELINE_STEPS} {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 1.5rem 0;
    }}

    .{CLASS_PIPELINE_STEP} {{
        color: var(--secondary-text-color);
        font-weight: 500;
        font-size: 0.9rem;
    }}

    .{CLASS_PIPELINE_ACTIVE} {{
        color: var(--accent-color);
        font-weight: 700;
    }}

    .{CLASS_PIPELINE_DONE} {{
        color: var(--success);
    }}

    .{CLASS_PIPELINE_ARROW} {{
        color: var(--border-color);
        margin: 0 0.5rem;
    }}

    .{CLASS_PIPELINE_ETA} {{
        font-size: 0.8rem;
        color: var(--secondary-text-color);
        margin-top: 0.5rem;
        font-style: italic;
    }}

    /* Sidebar User Badge */
    .{CLASS_SIDEBAR_USER_BADGE} {{
        display: flex;
        align-items: center;
        padding: 0.75rem;
        background-color: var(--secondary-bg);
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }}

    .{CLASS_AVATAR} {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: var(--accent-color);
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin-right: 0.75rem;
    }}

    /* Severity Badges */
    .{CLASS_BADGE} {{
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }}

    .{CLASS_SIM_PILL} {{
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }}

    .{CLASS_WELCOME_BANNER} {{
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--success) 100%);
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }}

    /* High Severity Row Styling (Issue #1569) */
    .high-severity-row {{
        border-left: 4px solid #ef4444 !important;
        background-color: rgba(239, 68, 68, 0.05) !important;
    }}
    """

    css = main_css + base_css + file_uploader_css + sidebar_active_tab_css

    if st.session_state.get("privacy_mode", False):
        css += """
        /* Privacy Mode: Blur student name labels */
        [class*="st-key-student_"] {
            filter: blur(4px) !important;
            transition: filter 0.3s ease;
        }
        [class*="st-key-student_"]:hover {
            filter: none !important;
        }
        """

    # Issue #644: wrap CSS in a nonced <style> block
    nonce = get_csp_nonce()
    css_html = f'<style nonce="{nonce}">\n{css}\n</style>'

    # ── Search Hotkey: press "/" to focus the warning search bar ──────────
    hotkey_js = f"""
    <script nonce="{nonce}">
    (function() {{
        // Prevent duplicate listeners (Streamlit re-runs on rerender)
        if (window.__chalu_hotkey_installed) return;
        window.__chalu_hotkey_installed = true;

        document.addEventListener('keydown', function(e) {{
            // Only trigger on "/" key
            if (e.key !== '/') return;
            // Don't intercept if user is already typing in an input/textarea
            var active = document.activeElement;
            if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) {{
                return;
            }}
            // Don't intercept modifier combos (Cmd+/, Ctrl+/)
            if (e.metaKey || e.ctrlKey || e.altKey) return;

            e.preventDefault();

            // Find the warning search input by its Streamlit widget key
            // Streamlit renders st.text_input(key="warning_search") with a
            // data attribute or aria-label matching the label text.
            var searchInputs = document.querySelectorAll('input[type="text"]');
            for (var i = 0; i < searchInputs.length; i++) {{
                var input = searchInputs[i];
                // Match by the placeholder or aria-label containing "search"
                var label = (input.getAttribute('placeholder') || '') +
                            (input.getAttribute('aria-label') || '');
                if (label.toLowerCase().indexOf('search') !== -1) {{
                    input.focus();
                    input.select();
                    return;
                }}
            }}
            // Fallback: try the .stTextInput class
            var textInputs = document.querySelectorAll('.stTextInput input[type="text"]');
            if (textInputs.length > 0) {{
                textInputs[0].focus();
                textInputs[0].select();
            }}
        }});
    }})();
    </script>
    """

    st.markdown(css_html, unsafe_allow_html=True)
    st.markdown(hotkey_js, unsafe_allow_html=True)
    st.markdown(back_to_top_html(), unsafe_allow_html=True)


# ── Severity Helpers ───────────────────────────────────────────────────────────
try:
    from src.core.config import (
        DEFAULT_THRESHOLDS,
        normalize_severity_label,
        severity_key,
    )
except ImportError:
    # Fallbacks for testing
    class DefaultThresholds:
        plagiarism = 0.59

    DEFAULT_THRESHOLDS = DefaultThresholds()

    def normalize_severity_label(label: str) -> str:
        return label.lower()

    def severity_key(score: float) -> str:
        if score >= 0.90:
            return "high"
        if score >= 0.59:
            return "medium"
        return "low"
def severity_tier(
    score: float, threshold: float = DEFAULT_THRESHOLDS.plagiarism
) -> str:
    """Return the severity tier based on score and threshold."""
    if score >= 0.90:
        return "high"
    elif score >= threshold:
        return "medium"
    else:
        return "low"


def tier_from_severity_label(label: str) -> str:
    """Map canonical or legacy severity labels to a lowercase tier."""
    try:
        return normalize_severity_label(label).lower()
    except ValueError:
        return "low"


def tier_color(tier: str) -> str:
    """Returns color hex associated with a tier."""
    colors = get_colors()
    if tier == "high":
        return colors["danger"]
    elif tier == "medium":
        return colors["warning"]
    elif tier == "low":
        return colors["success"]
    return colors["neutral_soft"]


def badge_html(tier: str, label: str = None) -> str:
    """Generates standard HTML badge chip for severity."""
    colors = get_colors()
    if tier == "high":
        text_color = colors["danger"]
        bg_color = colors["danger_soft"]
        default_label = "🔴 High"
    elif tier == "medium":
        text_color = colors["warning"]
        bg_color = colors["warning_soft"]
        default_label = "🟡 Medium"
    else:
        text_color = colors["success"]
        bg_color = colors["success_soft"]
        default_label = "🟢 Low"

    display_label = label if label is not None else default_label

    tooltip_map = {
        "high": "Similarity >= 80%",
        "medium": "Similarity between 50% and 79%",
        "low": "Similarity < 50%",
    }

    tooltip = tooltip_map.get(tier, "Similarity score")

    return (
        f'<span class="{CLASS_BADGE}" '
        f'title="{tooltip}" '
        f'style="background-color: {bg_color}; '
        f"color: {text_color}; "
        f'border: 1px solid {text_color};">'
        f"{display_label}</span>"
    )


# ── UI helpers ────────────────────────────────────────────────────────────────
def format_similarity_html(
    score: float,
    threshold: float = DEFAULT_THRESHOLDS.plagiarism,
) -> str:
    """Return a themed similarity pill using central severity boundaries."""
    colors = get_colors()
    tier = severity_key(score)

    if tier == "high":
        bg = colors["danger"]
        text = "#FFFFFF"
    elif tier == "medium":
        bg = colors["warning"]
        text = "#000000"
    else:
        bg = colors["success"]
        text = "#FFFFFF"

    return f'<span class="{CLASS_SIM_PILL}" style="background-color: {bg}; color: {text};">Similarity: {score * 100:.1f}%</span>'


def empty_state_html(icon: str, title: str, description: str) -> str:
    """Return styled empty-state HTML block."""
    return (
        f'<div class="{CLASS_EMPTY_STATE}">'
        f'<div class="{CLASS_EMPTY_ICON}">{icon}</div>'
        f'<div class="{CLASS_EMPTY_TITLE}">{title}</div>'
        f'<div class="{CLASS_EMPTY_DESC}">{description}</div>'
        f"</div>"
    )
def sidebar_user_badge_html(username: str, role: str) -> str:
    """Return the sidebar user badge with avatar circle."""
    initial = username[0].upper() if username else "?"
    return (
        f'<div class="{CLASS_SIDEBAR_USER_BADGE}">'
        f'<div class="{CLASS_AVATAR}">{initial}</div>'
        f"<div>"
        f'<div style="font-weight: 600;">{username}</div>'
        f'<div style="font-size: 0.8rem; color: {get_colors()["muted"]};">{role.upper()}</div>'
        f"</div>"
        f"</div>"
    )


def pipeline_progress_html(
    steps: list[str], active_index: int = -1, estimated_seconds: int | None = None
) -> str:
    """Return a horizontal pipeline progress indicator with optional ETA."""
    parts = []
    for i, step in enumerate(steps):
        if active_index < 0:
            cls = CLASS_PIPELINE_STEP
        elif i < active_index:
            cls = f"{CLASS_PIPELINE_STEP} {CLASS_PIPELINE_DONE}"
        elif i == active_index:
            cls = f"{CLASS_PIPELINE_STEP} {CLASS_PIPELINE_ACTIVE}"
        else:
            cls = CLASS_PIPELINE_STEP

        prefix = "✓ " if active_index >= 0 and i < active_index else ""
        parts.append(f'<span class="{cls}">{prefix}{step}</span>')

        if i < len(steps) - 1:
            parts.append(f'<span class="{CLASS_PIPELINE_ARROW}">→</span>')

    progress = f'<div class="{CLASS_PIPELINE_STEPS}">{"".join(parts)}</div>'

    if estimated_seconds is None:
        return progress

    try:
        from src.utils.processing_time import format_processing_duration

        duration = format_processing_duration(estimated_seconds)
    except ImportError:
        duration = f"{estimated_seconds}s"

    eta = f'<div class="{CLASS_PIPELINE_ETA}">Estimated processing time: about {duration}</div>'
    return f"{progress}{eta}"


def back_to_top_html(scroll_threshold: int = 250) -> str:
    """Return HTML and JavaScript for a floating back-to-top button.
    The button is hidden by default and fades in once the user scrolls past
    the configured threshold.  Clicking it smoothly scrolls the page to the top.

    Streamlit (>= 1.28) scrolls inside a container whose parent holds
    ``[data-testid="block-container"]``, not the window viewport.

    The IIFE guards against duplicate listener registration across Streamlit
    reruns.  The click handler uses event delegation and the scroll handler
    re-queries the button on each event so that Streamlit reruns (which
    recreate the DOM) do not break the feature.
    """
    nonce = get_csp_nonce()
    return f"""
title="Back to top">
        ⬆️ Top
    </button>
    <div id="back-to-top-status" class="sr-only" role="status" aria-live="polite"></div>    <script nonce="{nonce}">
    (function () {{
        if (window.__backToTopInitialized) return;
        window.__backToTopInitialized = true;

        var SCROLL_THRESHOLD = {scroll_threshold};
        /* Streamlit >= 1.28 scrolls inside the parent of
           [data-testid="block-container"], not the window. */
        var scrollContainer =
            document.querySelector('[data-testid="block-container"]')
                ?.parentElement
            || document.querySelector('section.main > div')
            || window;

        /* Event delegation — works even after Streamlit recreates the
           button element on a rerun. */
        scrollContainer.addEventListener('click', function (e) {{
            if (e.target.closest('#back-to-top-btn')) {{
                scrollContainer.scrollTo({{ top: 0, behavior: 'smooth' }});
            }}
        }});

        /* Re-query the button every scroll tick so the .visible class
           is always applied to the live element, not a detached one. */
        scrollContainer.addEventListener('scroll', function () {{
            var btn = document.getElementById('back-to-top-btn');
            var status = document.getElementById('back-to-top-status');
            if (!btn) return;
            var scrollTop = scrollContainer === window
                ? window.scrollY
                : scrollContainer.scrollTop;
            var shouldShow = scrollTop > SCROLL_THRESHOLD;
            var wasVisible = btn.classList.contains('visible');
            btn.classList.toggle('visible', shouldShow);
            if (status && shouldShow && !wasVisible) {{
                status.textContent = 'Back to top button available';
            }} else if (status && !shouldShow && wasVisible) {{
                status.textContent = '';
            }}
        }}, {{ passive: true }});    }})();
    </script>
    """




def active_tab_border_style(color: str = "#4f46e5", width: int = 4) -> str:
    """Return inline CSS string for an active navigation tab accent border (Issue #1028).

    Args:
        color: Hex or CSS color string for the accent border.
        width: Border width in pixels.

    Returns:
        CSS declaration string, e.g. "border-left: 4px solid #4f46e5;".
    """
    valid_color = sanitize_hex_color(color, fallback="#4f46e5")
    return f"border-left: {width}px solid {valid_color};"


def get_active_sidebar_tab_css(accent_border_color: str = "#4f46e5") -> str:
    """Generate standalone CSS snippet for active sidebar tab buttons.

    Args:
        accent_border_color: Primary border color for active state.

    Returns:
        CSS style block string.
    """
    colors = get_colors()
    border = sanitize_hex_color(accent_border_color, fallback="#4f46e5")
    bg = colors.get("neutral_soft", "#F1F5F9")
    accent = colors.get("accent", "#0D9488")
    return f"""
    section[data-testid="stSidebar"] .stButton button[data-selected="true"],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"][aria-selected="true"],
    .stButton button[data-selected="true"] {{
        border-left: 4px solid {border} !important;
        background-color: {bg} !important;
        color: {accent} !important;
        font-weight: 700 !important;
    }}
    """


def get_sidebar_tab_style(
    is_selected: bool = False,
    accent_border_color: str = "#4f46e5",
) -> dict[str, str]:
    """Return a dictionary of inline CSS properties for sidebar tab rendering.

    Args:
        is_selected: Whether the tab is currently active/selected.
        accent_border_color: Border accent color for the active state.

    Returns:
        Dictionary of CSS property names to values.
    """
    colors = get_colors()
    border = sanitize_hex_color(accent_border_color, fallback="#4f46e5")
    if is_selected:
        return {
            "border-left": f"4px solid {border}",
            "background-color": colors.get("neutral_soft", "#F1F5F9"),
            "color": colors.get("accent", "#0D9488"),
            "font-weight": "700",
        }
    return {
        "border-left": "4px solid transparent",
        "background-color": "transparent",
        "color": colors.get("ink", "#0F172A"),
        "font-weight": "400",
    }


# Theme Accent Palettes for custom sidebar highlight customization
THEME_ACCENT_PALETTES: dict[str, dict[str, str]] = {
    "Indigo": {"primary": "#4f46e5", "hover": "#4338ca", "light": "#e0e7ff"},
    "Teal": {"primary": "#0d9488", "hover": "#0f766e", "light": "#ccfbf1"},
    "Emerald": {"primary": "#059669", "hover": "#047857", "light": "#d1fae5"},
    "Rose": {"primary": "#e11d48", "hover": "#be123c", "light": "#ffe4e6"},
    "Violet": {"primary": "#7c3aed", "hover": "#6d28d9", "light": "#ede9fe"},
    "Amber": {"primary": "#d97706", "hover": "#b45309", "light": "#fef3c7"},
}


def get_theme_accent_color(theme_name: str | None = None) -> str:
    """Retrieve the primary accent color for a specified theme or active theme.

    Args:
        theme_name: Optional theme name ('Light', 'Dark', or palette name).

    Returns:
        Hex color string for active accent.
    """
    if theme_name in THEME_ACCENT_PALETTES:
        return THEME_ACCENT_PALETTES[theme_name]["primary"]
    if theme_name in THEMES:
        return THEMES[theme_name].get("accent", "#4f46e5")
    colors = get_colors()
    return colors.get("accent", "#4f46e5")


def build_active_tab_custom_css(
    accent_hex: str = "#4f46e5",
    border_width: int = 4,
    bg_hover: str | None = None,
) -> str:
    """Build dynamic custom CSS for active sidebar tab navigation.

    Args:
        accent_hex: Hex code for the left border accent.
        border_width: Width of the active border in pixels.
        bg_hover: Optional background color on hover.

    Returns:
        CSS text block with rules targeting active tab selectors.
    """
    border_color = sanitize_hex_color(accent_hex, fallback="#4f46e5")
    hover_bg = (
        sanitize_hex_color(bg_hover, fallback="#F1F5F9") if bg_hover else "#F1F5F9"
    )
    return f"""
    /* Custom Active Sidebar Tab Highlight */
    section[data-testid="stSidebar"] .stButton button[data-selected="true"],
    section[data-testid="stSidebar"] [aria-selected="true"],
    .stButton button[data-selected="true"] {{
        border-left: {border_width}px solid {border_color} !important;
        background-color: {hover_bg} !important;
        font-weight: 700 !important;
        transition: border-left 0.2s ease, background-color 0.2s ease !important;
    }}
    section[data-testid="stSidebar"] .stButton button[data-selected="true"]:hover {{
        border-left-color: {border_color} !important;
    }}
    """


def generate_active_tab_theme_tokens(theme_name: str | None = None) -> dict[str, str]:
    """Generate design system tokens for sidebar tab navigation states.

    Args:
        theme_name: Active theme ('Light' or 'Dark').

    Returns:
        Dictionary mapping tab state token keys to CSS color/dimension values.
    """
    selected_theme = theme_name if theme_name in THEMES else get_theme_name()
    palette = THEMES.get(selected_theme, THEMES["Light"])
    return {
        "active_border_color": "#4f46e5",
        "active_border_width": "4px",
        "active_bg_color": palette.get("neutral_soft", "#F1F5F9"),
        "active_text_color": palette.get("accent", "#0D9488"),
        "active_font_weight": "700",
        "inactive_bg_color": "transparent",
        "inactive_text_color": palette.get("ink", "#0F172A"),
        "inactive_font_weight": "400",
        "hover_border_color": "#4f46e5",
        "hover_bg_color": palette.get("surface", "#F8FAFC"),
    }


def get_sidebar_navigation_config() -> dict[str, Any]: # type: ignore
    """Return central configuration parameters for sidebar active tab rendering.

    Returns:
        Dictionary containing active tab style settings.
    """
    colors = get_colors()
    return {
        "accent_border_color": "#4f46e5",
        "accent_border_width_px": 4,
        "active_background": colors.get("neutral_soft", "#F1F5F9"),
        "active_text_color": colors.get("accent", "#0D9488"),
        "transition_duration_ms": 200,
        "border_position": "left",
        "supported_selectors": [
            'section[data-testid="stSidebar"] .stButton button[data-selected="true"]',
            'section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"][aria-selected="true"]',
            'section[data-testid="stSidebar"] button[aria-selected="true"]',
            'section[data-testid="stSidebar"] .stButton button.st-active',
            '.stButton button[data-selected="true"]',
        ],
    }


def render_active_tab_badge_html(tab_name: str, is_active: bool = False) -> str:
    """Render an HTML badge snippet representing an active/inactive tab indicator.

    Args:
        tab_name: Name of the navigation tab.
        is_active: Whether the tab is currently selected.

    Returns:
        HTML string representation.
    """
    colors = get_colors()
    if is_active:
        style = (
            f"border-left: 4px solid #4f46e5; "
            f"background-color: {colors.get('neutral_soft', '#F1F5F9')}; "
            f"color: {colors.get('accent', '#0D9488')}; "
            f"font-weight: 700; padding: 6px 12px; border-radius: 0 4px 4px 0;"
        )
    else:
        style = (
            f"border-left: 4px solid transparent; "
            f"background-color: transparent; "
            f"color: {colors.get('muted', '#64748B')}; "
            f"font-weight: 400; padding: 6px 12px;"
        )
    return f'<div class="sidebar-tab-badge" style="{style}">{tab_name}</div>'


SIDEBAR_TAB_THEME_TEMPLATES: dict[str, dict[str, str]] = {
    "Default": {
        "border_width": "4px",
        "border_style": "solid",
        "border_color": "#4f46e5",
        "border_radius": "0 6px 6px 0",
        "shadow": "0 1px 3px rgba(0,0,0,0.05)",
    },
    "Modern": {
        "border_width": "4px",
        "border_style": "solid",
        "border_color": "#4f46e5",
        "border_radius": "0 8px 8px 0",
        "shadow": "0 2px 4px rgba(79,70,229,0.15)",
    },
    "Glassmorphism": {
        "border_width": "4px",
        "border_style": "solid",
        "border_color": "#4f46e5",
        "border_radius": "0 10px 10px 0",
        "shadow": "0 4px 12px rgba(0,0,0,0.1)",
    },
    "Minimal": {
        "border_width": "3px",
        "border_style": "solid",
        "border_color": "#4f46e5",
        "border_radius": "0",
        "shadow": "none",
    },
    "High Contrast": {
        "border_width": "5px",
        "border_style": "solid",
        "border_color": "#4f46e5",
        "border_radius": "0 4px 4px 0",
        "shadow": "0 0 0 2px #000000",
    },
}


def generate_sidebar_theme_stylesheet(
    template_name: str = "Modern",
    accent_color: str = "#4f46e5",
) -> str:
    """Generate complete CSS stylesheet rules for sidebar navigation tabs.

    Args:
        template_name: Name of sidebar tab theme template.
        accent_color: Primary border accent color.

    Returns:
        Formatted CSS stylesheet block string.
    """
    template = SIDEBAR_TAB_THEME_TEMPLATES.get(
        template_name, SIDEBAR_TAB_THEME_TEMPLATES["Default"]
    )
    border = sanitize_hex_color(accent_color, fallback="#4f46e5")
    colors = get_colors()
    return f"""
    /* Sidebar Navigation Stylesheet ({template_name}) */
    section[data-testid="stSidebar"] .stButton button[data-selected="true"],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"][aria-selected="true"],
    section[data-testid="stSidebar"] button[aria-selected="true"],
    .stButton button[data-selected="true"] {{
        border-left: {template['border_width']} {template['border_style']} {border} !important;
        border-radius: {template['border_radius']} !important;
        box-shadow: {template['shadow']} !important;
        background-color: {colors.get('neutral_soft', '#F1F5F9')} !important;
        color: {colors.get('accent', '#0D9488')} !important;
        font-weight: 700 !important;
    }}
    """


def get_active_tab_accessibility_attributes(is_active: bool = True) -> dict[str, str]:
    """Return WAI-ARIA accessibility attributes for tab navigation buttons.

    Args:
        is_active: Whether the tab button is currently active.

    Returns:
        Dictionary of HTML attribute key-value pairs.
    """
    if is_active:
        return {
            "aria-selected": "true",
            "data-selected": "true",
            "tabindex": "0",
            "role": "tab",
        }
    return {
        "aria-selected": "false",
        "data-selected": "false",
        "tabindex": "-1",
        "role": "tab",
    }


def render_sidebar_navigation_menu(
    tabs: list[tuple[str, str]],
    active_tab_id: str,
) -> str:
    """Render an HTML string representing a complete sidebar navigation menu with active indicator.

    Args:
        tabs: List of tuples (tab_id, tab_label).
        active_tab_id: ID of the currently selected tab.

    Returns:
        HTML string containing menu container and tab elements.
    """
    html_items = []
    for tab_id, label in tabs:
        is_active = tab_id == active_tab_id
        badge = render_active_tab_badge_html(label, is_active=is_active)
        html_items.append(f'<li data-tab-id="{tab_id}">{badge}</li>')

    return f'<ul class="sidebar-nav-menu" style="list-style: none; padding: 0; margin: 0;">{"".join(html_items)}</ul>'


def render_timezone_footer() -> str:
    """Render current UTC server time and timezone label caption in the dashboard sidebar footer.

    Returns:
        Formatted server timezone caption string.
    """
    now_utc = datetime.now(timezone.utc)
    time_str = now_utc.strftime("%H:%M")
    caption_text = f"Server Time: {time_str} UTC"
    st.sidebar.caption(f"🕒 {caption_text}")
    return caption_text

