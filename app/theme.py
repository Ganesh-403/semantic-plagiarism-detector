from __future__ import annotations
"""
app/theme.py
------------
Theme management and CSS utility functions for the Semantic Plagiarism Detector.

Provides:
- Light and Dark theme color definitions
- CSS class name constants for consistent styling
- HTML generation helpers for UI components
- Dynamic theme injection for Streamlit
"""
# -*- coding: utf-8 -*-

from app.css_constants import (
    BADGE,
    EMPTY_STATE,
    EMPTY_ICON,
    EMPTY_TITLE,
    EMPTY_DESC,
    SIDEBAR_USER_BADGE,
    AVATAR,
    PIPELINE_STEPS,
    PIPELINE_STEP,
    PIPELINE_STEP_ACTIVE,
    PIPELINE_STEP_DONE,
    PIPELINE_ARROW,
    SIM_PILL,
)
import base64
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
import streamlit as st

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


# ── CSS Class Constants ────────────────────────────────────────────────────────
try:
    from app.css_constants import (
        CLASS_AVATAR, CLASS_BADGE, CLASS_EMPTY_DESC, CLASS_EMPTY_ICON,
        CLASS_EMPTY_STATE, CLASS_EMPTY_TITLE, CLASS_PIPELINE_ACTIVE,
        CLASS_PIPELINE_ARROW, CLASS_PIPELINE_DONE, CLASS_PIPELINE_ETA,
        CLASS_PIPELINE_STEP, CLASS_PIPELINE_STEPS, CLASS_SIDEBAR_USER_BADGE,
        CLASS_SIM_PILL, CLASS_WELCOME_BANNER
    )
except ImportError:
    # Fallbacks for isolated testing
    CLASS_AVATAR = "avatar-circle"
    CLASS_BADGE = "severity-badge"
    CLASS_EMPTY_DESC = "empty-desc"
    CLASS_EMPTY_ICON = "empty-icon"
    CLASS_EMPTY_STATE = "empty-state"
    CLASS_EMPTY_TITLE = "empty-title"
    CLASS_PIPELINE_ACTIVE = "pipeline-active"
    CLASS_PIPELINE_ARROW = "pipeline-arrow"
    CLASS_PIPELINE_DONE = "pipeline-done"
    CLASS_PIPELINE_ETA = "pipeline-eta"
    CLASS_PIPELINE_STEP = "pipeline-step"
    CLASS_PIPELINE_STEPS = "pipeline-steps"
    CLASS_SIDEBAR_USER_BADGE = "sidebar-user-badge"
    CLASS_SIM_PILL = "sim-pill"
    CLASS_WELCOME_BANNER = "welcome-banner"


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
UI_COLORMAP_OPTIONS: list[str] = ["Viridis", "Plasma", "Coolwarm", "YlOrRd"]

MATPLOTLIB_CMAP_MAPPING: dict[str, str] = {
    "Viridis": "viridis",
    "Plasma": "plasma",
    "Coolwarm": "coolwarm",
    "YlOrRd": "YlOrRd",
    "Legacy Red/Green": "RdYlGn_r",
}

PLOTLY_CMAP_MAPPING: dict[str, str] = {
    "Viridis": "Viridis",
    "Plasma": "Plasma",
    "Coolwarm": "RdBu_r",
    "YlOrRd": "YlOrRd",
    "Legacy Red/Green": "RdYlGn_r",
}

DEFAULT_UI_COLORMAP: str = "Viridis"


def apply_matplotlib_theme(theme_colors: dict | None = None) -> None:
    """
    Applies active theme colors and default styling rules to Matplotlib rcParams.

    Args:
        theme_colors: Optional dictionary containing theme colors. If None, uses get_colors().
    """
    import matplotlib as mpl

    colors = sanitize_theme_colors(theme_colors if theme_colors is not None else get_colors())

    mpl.rcParams.update({
        "figure.facecolor": colors.get("background", "#FFFFFF"),
        "axes.facecolor": colors.get("surface", "#F8FAFC"),
        "axes.edgecolor": colors.get("border", "#E2E8F0"),
        "axes.labelcolor": colors.get("ink", "#0F172A"),
        "xtick.color": colors.get("ink", "#0F172A"),
        "ytick.color": colors.get("ink", "#0F172A"),
        "text.color": colors.get("ink", "#0F172A"),
        "grid.color": colors.get("border", "#E2E8F0"),
        "figure.edgecolor": colors.get("border", "#E2E8F0"),
        "savefig.facecolor": colors.get("background", "#FFFFFF"),
        "savefig.edgecolor": colors.get("background", "#FFFFFF"),
    })


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


def inject_css() -> None:
    """
    Inject CSS for the currently selected Light or Dark theme.
    
    Includes comprehensive styling for file uploaders, empty states, 
    pipeline indicators, and severity badges to ensure a cohesive UI.
    """
    colors = sanitize_theme_colors(get_colors())

    # Issue #572: File Uploader Drag-Zone Customization
    file_uploader_css = f"""
    /* File Uploader Drag-Zone Customization */
    .stFileUploader [data-testid="stFileUploaderDropzone"] {{
        border: 2px dashed {colors['border']} !important;
        border-radius: 8px !important;
        background-color: {colors['surface']} !important;
        transition: all 0.2s ease-in-out !important;
        padding: 1.5rem !important;
    }}
    
    .stFileUploader [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {colors['accent']} !important;
        background-color: {colors['neutral_soft']} !important;
        cursor: pointer !important;
    }}
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderInstruction"] {{
        color: {colors['muted']} !important;
        font-weight: 500 !important;
    }}
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderBrowseFiles"] {{
        background-color: {colors['accent']} !important;
        color: #FFFFFF !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        transition: background-color 0.2s ease !important;
    }}
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderBrowseFiles"]:hover {{
        background-color: {colors['ink']} !important;
    }}
    """

    base_css = f"""
    /* Global Theme Overrides */
    .stApp {{
        background-color: {colors['background']} !important;
        color: {colors['ink']} !important;
    }}
    
    .block-container {{
        padding-top: 2rem !important;
    }}
    
    .stAlert {{
        border-radius: 8px !important;
    }}
    
    .stCard {{
        background-color: {colors['card']} !important;
        border: 1px solid {colors['border']} !important;
        border-radius: 8px !important;
    }}
    
    /* Empty State Styling */
    .{CLASS_EMPTY_STATE} {{
        text-align: center;
        padding: 2rem;
        background-color: {colors['surface']};
        border-radius: 8px;
        border: 1px dashed {colors['border']};
    }}
    
    .{CLASS_EMPTY_ICON} {{
        font-size: 3rem;
        margin-bottom: 1rem;
        color: {colors['muted']};
    }}
    
    .{CLASS_EMPTY_TITLE} {{
        font-size: 1.25rem;
        font-weight: 600;
        color: {colors['ink']};
        margin-bottom: 0.5rem;
    }}
    
    .{CLASS_EMPTY_DESC} {{
        color: {colors['muted']};
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
        color: {colors['muted']};
        font-weight: 500;
        font-size: 0.9rem;
    }}
    
    .{CLASS_PIPELINE_ACTIVE} {{
        color: {colors['accent']};
        font-weight: 700;
    }}
    
    .{CLASS_PIPELINE_DONE} {{
        color: {colors['success']};
    }}
    
    .{CLASS_PIPELINE_ARROW} {{
        color: {colors['border']};
        margin: 0 0.5rem;
    }}
    
    .{CLASS_PIPELINE_ETA} {{
        font-size: 0.8rem;
        color: {colors['muted']};
        margin-top: 0.5rem;
        font-style: italic;
    }}
    
    /* Sidebar User Badge */
    .{CLASS_SIDEBAR_USER_BADGE} {{
        display: flex;
        align-items: center;
        padding: 0.75rem;
        background-color: {colors['surface']};
        border-radius: 8px;
        border: 1px solid {colors['border']};
        margin-bottom: 1rem;
    }}
    
    .{CLASS_AVATAR} {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: {colors['accent']};
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
        background: linear-gradient(135deg, {colors['accent']} 0%, {colors['success']} 100%);
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }}
    """

    css = base_css + file_uploader_css

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

    st.markdown(css, unsafe_allow_html=True)


# ── Severity Helpers ───────────────────────────────────────────────────────────
try:
    from src.core.config import DEFAULT_THRESHOLDS, normalize_severity_label, severity_key
except ImportError:
    # Fallbacks for testing
    class DefaultThresholds:
        plagiarism = 0.59
    DEFAULT_THRESHOLDS = DefaultThresholds()
    def normalize_severity_label(label: str) -> str: return label.lower()
    def severity_key(score: float) -> str:
        if score >= 0.90:
            return "high"
        if score >= 0.59:
            return "medium"
        return "low"


def severity_tier(score: float, threshold: float = DEFAULT_THRESHOLDS.plagiarism) -> str:
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
    return f'<span class="{BADGE}" style="background-color: {bg_color}; color: {text_color}; border: 1px solid {text_color};">{display_label}</span>'
    return f'<span class="{CLASS_BADGE}" style="background-color: {bg_color}; color: {text_color}; border: 1px solid {text_color};">{display_label}</span>'
    return f'<span class="{CLASS_BADGE}" style="color: {text_color}; background-color: {bg_color};">{display_label}</span>'


def format_similarity_html(score: float, threshold: float = DEFAULT_THRESHOLDS.plagiarism) -> str:
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

    return (
        f'<span class="{SIM_PILL}" style="background:{bg};">'
        f'<span class="{CLASS_SIM_PILL}" style="background:{bg};">'
        f"Similarity: {score * 100:.1f}%</span>"
    )
    return f'<span class="{CLASS_SIM_PILL}" style="background-color: {bg}; color: {text};">Similarity: {score * 100:.1f}%</span>'


def empty_state_html(icon: str, title: str, description: str) -> str:
    """Return styled empty-state HTML block."""
    return (
        f'<div class="{EMPTY_STATE}">'
        f'<div class="{EMPTY_ICON}">{icon}</div>'
        f'<div class="{EMPTY_TITLE}">{title}</div>'
        f'<div class="{EMPTY_DESC}">{description}</div>'
        f'<div class="{CLASS_EMPTY_STATE}">'
        f'<div class="{CLASS_EMPTY_ICON}">{icon}</div>'
        f'<div class="{CLASS_EMPTY_TITLE}">{title}</div>'
        f'<div class="{CLASS_EMPTY_DESC}">{description}</div>'
        f'</div>'
    )


def sidebar_user_badge_html(username: str, role: str) -> str:
    """Return the sidebar user badge with avatar circle."""
    initial = username[0].upper() if username else "?"
    return (
        f'<div class="{SIDEBAR_USER_BADGE}">'
        f'<div class="{AVATAR}">{initial}</div>'
        f'<div><strong>{username}</strong><br>'
        f'<div class="{CLASS_SIDEBAR_USER_BADGE}">'
        f'<div class="{CLASS_AVATAR}">{initial}</div>'
        f'<div>'
        f'<div style="font-weight: 600;">{username}</div>'
        f'<div style="font-size: 0.8rem; color: {get_colors()["muted"]};">{role.upper()}</div>'
        f'</div>'
        f'</div>'
    )


def pipeline_progress_html(steps: list[str], active_index: int = -1, estimated_seconds: int | None = None) -> str:
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
    """Return HTML for a floating back-to-top button."""
    return """
    <style>
        .back-to-top {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #0D9488;
            color: white;
            padding: 10px 15px;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            z-index: 999;
        }
    </style>
    <a href="#top" class="back-to-top">⬆️</a>
    """


def version_check_widget_html(local_version: str, latest_tag: str, repo_url: str = "https://github.com/Ganesh-403/semantic-plagiarism-detector/releases/latest") -> str:
    """Return an HTML snippet that renders an update-available notification banner."""
    colors = get_colors()
    warning_color = colors["warning"]
    warning_soft = colors["warning_soft"]
    ink = colors["ink"]

    return f"""
<div id="spd-update-banner" style="
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    margin-top: 8px;
    background: {warning_soft};
    border: 1px solid {warning_color};
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: {ink};
">
    <span style="font-size: 1.1rem;">🔔</span>
    <span>
        <strong>Update available:</strong>
        v{local_version} &rarr; <strong>{latest_tag}</strong>.
        &nbsp;
        <a href="{repo_url}" target="_blank" rel="noopener noreferrer"
           style="color: {warning_color}; font-weight: 600; text-decoration: underline;">
            View release &rarr;
        </a>
    </span>
</div>

  def version_check_widget_html(local_version: str, latest_tag: str, repo_url: str = "https://github.com/Ganesh-403/semantic-plagiarism-detector/releases") -> str:
    warning_soft = "rgba(255, 165, 0, 0.1)"
    warning_color = "#ffa500"
    ink = "#333333"

    return f
    <div style="background-color: {warning_soft}; border-left: 4px solid {warning_color}; padding: 1rem; border-radius: 4px; color: {ink};">
        <strong>&#128276; Update available:</strong> v{local_version} &rarr; <strong>{latest_tag}</strong>. 
        <a href="{repo_url}" target="_blank" style="color: {warning_color}; font-weight: 600; margin-left: 0.5rem;">View release &rarr;</a>
    </div>
"""