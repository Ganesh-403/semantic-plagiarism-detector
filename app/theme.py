import streamlit as st

from src.core.config import DEFAULT_THRESHOLDS, normalize_severity_label, severity_key

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


def initialize_theme() -> None:
    """Initialize the active theme for the current session."""
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"


def get_theme_name() -> str:
    """Return the active theme name."""
    initialize_theme()
    return st.session_state.theme


def set_theme(theme_name: str) -> None:
    """Set the active theme."""
    if theme_name in THEMES:
        st.session_state.theme = theme_name


def get_colors() -> dict:
    """Return the colors for the active theme."""
    return THEMES[get_theme_name()]


def inject_css() -> None:
    """Inject CSS for the currently selected Light or Dark theme."""
    colors = get_colors()

    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

        :root {{
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
            background-color: var(--background) !important;
            color: var(--ink) !important;
        }}

        [data-testid="stHeader"] {{
            background-color: var(--background) !important;
        }}

        [data-testid="stToolbar"] {{
            color: var(--ink) !important;
        }}

        h1,
        h2,
        h3,
        h4,
        h5,
        h6 {{
            font-family: 'Newsreader', Georgia, serif !important;
            color: var(--ink) !important;
            font-weight: 700 !important;
        }}

        p,
        label,
        span,
        li,
        [data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"] {{
            color: var(--ink);
        }}

        [data-testid="stCaptionContainer"],
        .stCaption {{
            color: var(--muted) !important;
        }}

        .hero-kicker {{
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.25rem;
        }}

        /* ── Sidebar ────────────────────────────────────────────────── */

        [data-testid="stSidebar"] {{
            background-color: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
        }}

        [data-testid="stSidebar"] * {{
            color: var(--ink);
        }}

        .sidebar-brand-title {{
            font-family: 'Newsreader', serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--ink);
            text-align: center;
            line-height: 1.2;
            margin-top: 0.25rem;
            margin-bottom: 0;
        }}

        .sidebar-brand-kicker {{
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            text-align: center;
            margin-bottom: 1.25rem;
        }}

        .sidebar-section-label {{
            font-family: 'Inter', sans-serif;
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 2px;
        }}

        .sidebar-user-badge {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 8px;
            background-color: var(--neutral-soft);
            border: 1px solid var(--border);
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--ink);
            margin-bottom: 0.75rem;
        }}

        .sidebar-user-badge .avatar {{
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background-color: var(--accent);
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
            border: 1px solid var(--border) !important;
            border-top: 4px solid var(--accent) !important;
            border-radius: 8px !important;
            padding: 14px 16px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12) !important;
        }}

        div[data-testid="stMetricLabel"] > div {{
            font-family: 'Inter', sans-serif !important;
            font-size: 0.75rem !important;
            font-weight: 700 !important;
            color: var(--muted) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }}

        div[data-testid="stMetricValue"] > div {{
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            color: var(--ink) !important;
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
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--ink);
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .meta-chip code {{
            font-family: 'IBM Plex Mono', monospace !important;
            background: none !important;
            padding: 0 !important;
            color: var(--accent) !important;
            font-weight: 700 !important;
        }}

        /* ── Login container ────────────────────────────────────────── */

        .login-container {{
            background-color: var(--card) !important;
            border: 1px solid var(--border) !important;
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
            color: var(--ink);
            margin-bottom: 0.25rem;
        }}

        .login-container .login-subtitle {{
            font-size: 0.85rem;
            color: var(--muted);
        }}

        .login-accent-bar {{
            height: 4px;
            background: linear-gradient(90deg, var(--accent), transparent);
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
            color: var(--muted);
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
            color: var(--ink) !important;
            border-color: var(--border) !important;
        }}

        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [role="listbox"] {{
            background-color: var(--card) !important;
            color: var(--ink) !important;
        }}

        .stButton button,
        .stDownloadButton button,
        .stFormSubmitButton button {{
            border-color: var(--border) !important;
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

        [data-testid="stExpander"],
        [data-testid="stForm"] {{
            background-color: var(--card) !important;
            border-color: var(--border) !important;
        }}

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {{
            border-color: var(--border) !important;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background-color: var(--surface) !important;
            border-color: var(--border) !important;
        }}

        /* ── Tabs ───────────────────────────────────────────────────── */

        [data-testid="stTabs"] button {{
            color: var(--muted) !important;
        }}

        [data-testid="stTabs"] button[aria-selected="true"] {{
            color: var(--accent) !important;
            border-bottom-color: var(--accent) !important;
        }}

        hr {{
            border-color: var(--border) !important;
        }}

        /* ── Enhanced footer ────────────────────────────────────────── */

        .app-footer {{
            text-align: center;
            padding: 1rem 0 0.5rem;
            font-size: 0.78rem;
            color: var(--muted);
        }}

        .app-footer a {{
            color: var(--accent);
            text-decoration: none;
        }}

        .app-footer a:hover {{
            text-decoration: underline;
        }}

        /* ── Empty state ────────────────────────────────────────────── */

        .empty-state {{
            text-align: center;
            padding: 2.5rem 1rem;
            color: var(--muted);
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
            color: var(--ink);
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
            color: var(--muted);
            border: 1px solid var(--border);
        }}

        .pipeline-step.active {{
            background-color: var(--accent);
            color: white;
            border-color: var(--accent);
            animation: pipelinePulse 1.2s ease-in-out infinite;
        }}

        .pipeline-step.done {{
            background-color: var(--success-soft);
            color: var(--success);
            border-color: var(--success);
        }}

        .pipeline-arrow {{
            color: var(--muted);
            font-size: 0.7rem;
        }}

        @keyframes pipelinePulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}

        /* ── Back to Top Button ─────────────────────────────────────── */

        #back-to-top-btn {{
            position: fixed;
            bottom: max(2rem, env(safe-area-inset-bottom, 2rem));
            right: max(2rem, env(safe-area-inset-right, 2rem));
            z-index: 9999;
            background-color: var(--accent);
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
            outline: 2px solid var(--accent);
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
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


# ── Severity helpers ──────────────────────────────────────────────────────────


def severity_tier(
    score: float,
    threshold: float = DEFAULT_THRESHOLDS.plagiarism,
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
    if tier == "high":
        return get_colors()["danger"]
    elif tier == "medium":
        return get_colors()["warning"]
    elif tier == "low":
        return get_colors()["success"]
    return get_colors()["neutral_soft"]


def badge_html(tier: str, label: str = None) -> str:
    """Generates standard HTML badge chip for severity."""
    if tier == "high":
        text_color = get_colors()["danger"]
        bg_color = get_colors()["danger_soft"]
        default_label = "🔴 High"
    elif tier == "medium":
        text_color = get_colors()["warning"]
        bg_color = get_colors()["warning_soft"]
        default_label = "🟡 Medium"
    else:
        text_color = get_colors()["success"]
        bg_color = get_colors()["success_soft"]
        default_label = "🟢 Low"

    display_label = label if label is not None else default_label
    return f'<span class="badge" style="background-color: {bg_color}; color: {text_color}; border: 1px solid {text_color};">{display_label}</span>'


# ── UI helpers ────────────────────────────────────────────────────────────────


def format_similarity_html(
    score: float,
    threshold: float = DEFAULT_THRESHOLDS.plagiarism,
) -> str:
    """Return a themed similarity pill using central severity boundaries."""
    del threshold
    colors = get_colors()
    tier = severity_key(score)

    if tier == "high":
        bg = colors["danger"]
    elif tier == "medium":
        bg = colors["warning"]
    else:
        bg = colors["success"]

    return (
        f'<span class="sim-pill" style="background:{bg};">'
        f"Similarity: {score * 100:.1f}%</span>"
    )


def empty_state_html(icon: str, title: str, description: str) -> str:
    """Return styled empty-state HTML block."""
    return (
        f'<div class="empty-state">'
        f'<div class="empty-icon">{icon}</div>'
        f'<div class="empty-title">{title}</div>'
        f'<div class="empty-desc">{description}</div>'
        f"</div>"
    )


def sidebar_user_badge_html(username: str, role: str) -> str:
    """Return the sidebar user badge with avatar circle."""
    initial = username[0].upper() if username else "?"
    return (
        f'<div class="sidebar-user-badge">'
        f'<div class="avatar">{initial}</div>'
        f"<div><strong>{username}</strong><br>"
        f'<span style="font-size:0.7rem;color:var(--muted);">{role.upper()}</span></div>'
        f"</div>"
    )


def pipeline_progress_html(steps: list[str], active_index: int = -1) -> str:
    """Return a horizontal pipeline progress indicator.

    Args:
        steps: List of step labels (e.g. ["Extract", "Chunk", "Embed", ...]).
        active_index: 0-based index of the currently running step.
            Steps before *active_index* are marked done; steps after are
            pending.  Pass -1 (default) to mark all steps as pending.
    """
    parts = []
    for i, step in enumerate(steps):
        if active_index < 0:
            cls = "pipeline-step"
        elif i < active_index:
            cls = "pipeline-step done"
        elif i == active_index:
            cls = "pipeline-step active"
        else:
            cls = "pipeline-step"

        prefix = "✓ " if (active_index >= 0 and i < active_index) else ""
        parts.append(f'<span class="{cls}">{prefix}{step}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="pipeline-arrow">→</span>')

    return f'<div class="pipeline-steps">{"".join(parts)}</div>'


def back_to_top_html() -> str:
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
    return """
    <button id="back-to-top-btn"
            type="button"
            aria-label="Back to top"
            title="Back to top">
        ⬆️ Top
    </button>
    <script>
    (function () {
        if (window.__backToTopInitialized) return;
        window.__backToTopInitialized = true;

        var SCROLL_THRESHOLD = 250;

        /* Streamlit >= 1.28 scrolls inside the parent of
           [data-testid="block-container"], not the window. */
        var scrollContainer =
            document.querySelector('[data-testid="block-container"]')
                ?.parentElement
            || document.querySelector('section.main > div')
            || window;

        /* Event delegation — works even after Streamlit recreates the
           button element on a rerun. */
        scrollContainer.addEventListener('click', function (e) {
            if (e.target.closest('#back-to-top-btn')) {
                scrollContainer.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });

        /* Re-query the button every scroll tick so the .visible class
           is always applied to the live element, not a detached one. */
        scrollContainer.addEventListener('scroll', function () {
            var btn = document.getElementById('back-to-top-btn');
            if (!btn) return;
            var scrollTop = scrollContainer === window
                ? window.scrollY
                : scrollContainer.scrollTop;
            btn.classList.toggle('visible', scrollTop > SCROLL_THRESHOLD);
        }, { passive: true });
    })();
    </script>
    """


def version_check_widget_html(
    local_version: str,
    latest_tag: str,
    repo_url: str = "https://github.com/Ganesh-403/semantic-plagiarism-detector/releases/latest",
) -> str:
    """Return an HTML snippet that renders an update-available notification banner.

    The banner is intentionally lightweight — pure HTML/CSS with no external
    dependencies — so it renders reliably inside ``st.markdown(...,
    unsafe_allow_html=True)``.

    Parameters
    ----------
    local_version:
        The version string of the currently running application.
    latest_tag:
        The newer tag string returned by the GitHub API (e.g. ``"v1.2.0"``).
    repo_url:
        Link target for the "View release" call-to-action.

    Returns
    -------
    str
        A self-contained HTML string ready for ``st.markdown``.
    """
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
"""