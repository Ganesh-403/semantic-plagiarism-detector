"""
css_constants.py
----------------
Centralized CSS class name constants for consistent styling across the application.

All CSS class names used in the application should be defined here to avoid
hardcoded strings and ensure consistency when styling changes are needed.
"""

# ── Hero kicker ────────────────────────────────────────────────────────────────

HERO_KICKER = "hero-kicker"
"""Hero kicker element for section introductions."""

# ── Badge and chip classes ─────────────────────────────────────────────────────

BADGE = "badge"
"""Standard badge class for severity indicators and status chips."""

SIM_PILL = "sim-pill"
"""Similarity score pill class for displaying similarity percentages."""

META_CHIP = "meta-chip"
"""Metadata chip class for displaying document metadata."""

META_CHIP_CODE = "meta-chip code"
"""Code element within meta-chip for syntax highlighting."""

# ── Empty state classes ────────────────────────────────────────────────────────

EMPTY_STATE = "empty-state"
"""Main container for empty state messages."""

EMPTY_ICON = "empty-icon"
"""Icon element within empty state."""

EMPTY_TITLE = "empty-title"
"""Title element within empty state."""

EMPTY_DESC = "empty-desc"
"""Description element within empty state."""

# ── Sidebar classes ────────────────────────────────────────────────────────────

SIDEBAR_USER_BADGE = "sidebar-user-badge"
"""User badge container in sidebar."""

AVATAR = "avatar"
"""Avatar circle element in user badge."""

SIDEBAR_BRAND_TITLE = "sidebar-brand-title"
"""Sidebar brand/title element."""

SIDEBAR_BRAND_KICKER = "sidebar-brand-kicker"
"""Sidebar brand kicker/subtitle element."""

SIDEBAR_SECTION_LABEL = "sidebar-section-label"
"""Section label in sidebar."""

# ── Pipeline progress classes ──────────────────────────────────────────────────

PIPELINE_STEPS = "pipeline-steps"
"""Container for pipeline progress indicators."""

PIPELINE_STEP = "pipeline-step"
"""Individual pipeline step element."""

PIPELINE_STEP_ACTIVE = "pipeline-step active"
"""Active (currently running) pipeline step."""

PIPELINE_STEP_DONE = "pipeline-step done"
"""Completed pipeline step."""

PIPELINE_ARROW = "pipeline-arrow"
"""Arrow separator between pipeline steps."""

PIPELINE_DONE = "done"
"""Completed pipeline step state."""

PIPELINE_ACTIVE = "active"
"""Active pipeline step state."""

PIPELINE_ETA = "pipeline-eta"
"""Estimated processing time container."""

DOC_ROW = "doc-row"
"""Document listing row."""

WELCOME_BANNER = "welcome-banner"
"""Dashboard welcome banner."""


# ── Mono text ──────────────────────────────────────────────────────────────────

MONO_TEXT = "mono-text"
"""Monospace text class for code snippets and technical content."""

# ── Warning card classes ───────────────────────────────────────────────────────

WARNING_CARD_HIGH = "warning-card-high"
"""High severity warning card accent border."""

WARNING_CARD_MEDIUM = "warning-card-medium"
"""Medium severity warning card accent border."""

WARNING_CARD_LOW = "warning-card-low"
"""Low severity warning card accent border."""

LOW_CONFIDENCE_CARD = "low-confidence-card"
"""Low confidence detection card amber accent border."""

HIGH_SEVERITY_ROW = "high-severity-row"
"""High severity plagiarism row accent border."""

# ── Login container classes ────────────────────────────────────────────────────

LOGIN_CONTAINER = "login-container"
"""Login form container."""

LOGIN_HEADER = "login-container login-header"
"""Header within login container."""

LOGIN_ICON = "login-container login-icon"
"""Icon within login container."""

LOGIN_TITLE = "login-container login-title"
"""Title within login container."""

LOGIN_SUBTITLE = "login-container login-subtitle"
"""Subtitle within login container."""

LOGIN_ACCENT_BAR = "login-accent-bar"
"""Accent bar at bottom of login container."""

# ── Footer classes ─────────────────────────────────────────────────────────────

APP_FOOTER = "app-footer"
"""Application footer container."""

# ── Legend classes ─────────────────────────────────────────────────────────────

LEGEND_CONTAINER = "legend-container"
"""Container for legend elements."""

LEGEND_ITEM = "legend-item"
"""Individual legend item."""

LEGEND_COLOR = "legend-color"
"""Color indicator in legend."""

# ── Form input classes ─────────────────────────────────────────────────────────

ST_TEXT_INPUT = "stTextInput"
"""Text input field."""

ST_TEXT_AREA = "stTextArea"
"""Text area input."""

ST_NUMBER_INPUT = "stNumberInput"
"""Number input field."""

ST_BUTTON = "stButton"
"""Button element."""

ST_DOWNLOAD_BUTTON = "stDownloadButton"
"""Download button element."""

ST_FORM_SUBMIT_BUTTON = "stFormSubmitButton"
"""Form submit button element."""

ST_EXPANDER = "stExpander"
"""Expander element."""

ST_FORM = "stForm"
"""Form container."""

ST_DATAFRAME = "stDataFrame"
"""DataFrame display element."""

ST_TABLE = "stTable"
"""Table display element."""

ST_FILE_UPLOADER_DROPZONE = "stFileUploaderDropzone"
"""File uploader dropzone."""

ST_CAPTION_CONTAINER = "stCaptionContainer"
"""Caption container element."""

ST_CAPTION = "stCaption"
"""Caption element."""

# ── Tab classes ────────────────────────────────────────────────────────────────

ST_TABS = "stTabs"
"""Tabs container."""

ST_TABS_BUTTON = "stTabs button"
"""Individual tab button."""

ST_TABS_BUTTON_ACTIVE = 'stTabs button[aria-selected="true"]'
"""Active (selected) tab button."""

# ── Mobile layout ──────────────────────────────────────────────────────────────

MOBILE_LAYOUT_CSS = """
@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem !important;
        padding-right: 0.75rem !important;
        padding-left: 0.75rem !important;
        padding-bottom: 1rem !important;
    }

    [data-testid="stPlotlyChart"],
    .stPlotlyChart {
        height: 280px !important;
        max-height: 280px !important;
    }

    [data-testid="stPlotlyChart"] .js-plotly-plot,
    .js-plotly-plot {
        height: 260px !important;
        max-height: 260px !important;
    }
}
"""
"""Responsive rules for narrow viewports: tighter padding and shorter charts."""
