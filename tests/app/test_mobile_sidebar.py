"""
Regression tests for issue #258 — Auto-Hide Sidebar on Mobile Devices.

Definition of Done: the sidebar should default to the collapsed state on
screens narrower than 768px. This is achieved by:

1. Setting ``initial_sidebar_state="auto"`` in ``st.set_page_config`` in
   ``app/streamlit_app.py``, which lets Streamlit's own responsive logic
   collapse the sidebar below its "md" (768px) breakpoint instead of
   forcing it open on every device.
2. A supplementary ``@media (max-width: 768px)`` rule in ``app/theme.py``
   that keeps the sidebar from covering the full viewport if a mobile
   user re-opens it, so the similarity matrix / heatmap stay legible.

These tests inspect the source directly rather than rendering a browser,
since viewport-driven behaviour isn't observable through
``streamlit.testing.v1.AppTest``.
"""

import os
import re

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "app"))
_STREAMLIT_APP_PATH = os.path.join(_APP_DIR, "streamlit_app.py")
_THEME_PATH = os.path.join(_APP_DIR, "theme.py")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_sidebar_state_is_auto_not_expanded():
    """The sidebar must not be forced to 'expanded' on every device."""
    source = _read(_STREAMLIT_APP_PATH)

    match = re.search(r'initial_sidebar_state\s*=\s*"([^"]+)"', source)
    assert match is not None, "st.set_page_config must set initial_sidebar_state"
    assert match.group(1) == "auto", (
        "initial_sidebar_state should be 'auto' so Streamlit collapses the "
        "sidebar by default on narrow (mobile) viewports; 'expanded' forces "
        "it open on phones/tablets too (issue #258)."
    )


def test_mobile_media_query_constrains_sidebar_width():
    """The 768px breakpoint block should keep the sidebar from covering the
    whole screen when a mobile user opens it."""
    source = _read(_THEME_PATH)

    media_query_match = re.search(
        r"@media \(max-width:\s*768px\).*?</style>",
        source,
        re.DOTALL,
    )
    assert media_query_match is not None, "Expected a 768px mobile media query block"

    block = media_query_match.group(0)
    assert 'data-testid="stSidebar"' in block, (
        "Mobile media query should include a rule constraining "
        "[data-testid='stSidebar'] width (issue #258)"
    )
