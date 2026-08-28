# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Warnings & Live Incident Stream View Component.

Renders Tab 1 containing detected plagiarism warnings, live feed auto-refresh,
date range filters, and expand/collapse controls.
"""

from datetime import date, timedelta

import streamlit as st

from app.session_keys import SessionKeys
from src.i18n.translator import get_text

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

try:
    from src.utils.warning_list import render_warning_controls, reset_warning_page
except ImportError:
    render_warning_controls = None
    reset_warning_page = None


def _set_warning_page(page: int) -> None:
    st.session_state.warning_page = page


def get_date_range_preset(preset: str) -> tuple[date, date]:
    """Calculate start and end dates based on preset string."""
    today = date.today()
    if preset == "Today":
        return today, today
    elif preset == "Last 7 Days":
        return today - timedelta(days=6), today
    elif preset == "Last 30 Days":
        return today - timedelta(days=29), today
    else:  # "All Time"
        return date(2020, 1, 1), today


def render_warnings_view(
    flags: list, threshold: float, ai_probabilities: dict, lang_code: str
):
    """Render Tab 1: Warnings & Live Incident Stream."""
    st.subheader(get_text("tab_warnings", lang=lang_code))

    auto_refresh_enabled = st.toggle(
        "Auto-refresh live feed (30s)",
        value=False,
        key=SessionKeys.INCIDENT_STREAM_AUTO_REFRESH,
        help=(
            "When enabled, the incident feed re-runs every 30 seconds "
            "to surface newly flagged submissions automatically."
        ),
    )

    if auto_refresh_enabled and st_autorefresh is not None:
        st_autorefresh(
            interval=30 * 1000,
            key="incident_stream_autorefresh",
        )

    st.session_state[SessionKeys.INCIDENT_STREAM_AUTO_REFRESH] = auto_refresh_enabled

    if auto_refresh_enabled:
        if st_autorefresh is None:
            st.warning(
                "Auto-refresh is enabled, but the `streamlit-autorefresh` "
                "package is not installed. Install it via `pip install streamlit-autorefresh`."
            )
        else:
            st.caption("🔴 Live — refreshing every 30 seconds.")
    else:
        st.caption("⚪ Live feed paused — toggle on to auto-refresh.")

    st.divider()

    if SessionKeys.WARNINGS_EXPAND_ALL not in st.session_state:
        st.session_state[SessionKeys.WARNINGS_EXPAND_ALL] = False

    st.markdown("### 📅 Incident Date Filter")
    date_preset = st.radio(
        "Select Date Range",
        options=["Today", "Last 7 Days", "Last 30 Days", "All Time"],
        horizontal=True,
        key="incident_date_preset",
        help="Quickly filter the incident table by common date ranges.",
    )

    start_date, end_date = get_date_range_preset(date_preset)
    st.caption(
        f"Filtering incidents from **{start_date.strftime('%Y-%m-%d')}** to "
        f"**{end_date.strftime('%Y-%m-%d')}**"
    )

    if not flags:
        st.info("No plagiarism incidents detected above configured threshold.")
    elif render_warning_controls is not None:
        if "warning_page" not in st.session_state:
            _set_warning_page(reset_warning_page())

        render_warning_controls(
            flags,
            threshold=threshold,
            ai_probabilities=ai_probabilities,
            set_warning_page=_set_warning_page,
        )

        button_label = (
            "📂 Expand All"
            if not st.session_state[SessionKeys.WARNINGS_EXPAND_ALL]
            else "📁 Collapse All"
        )

        if st.button(button_label, key="toggle_warning_accordions"):
            st.session_state[SessionKeys.WARNINGS_EXPAND_ALL] = not st.session_state[
                SessionKeys.WARNINGS_EXPAND_ALL
            ]
            st.rerun()

        render_warning_controls(
            flags,
            threshold=threshold,
            ai_probabilities=ai_probabilities,
            expanded=st.session_state[SessionKeys.WARNINGS_EXPAND_ALL],
            set_warning_page=_set_warning_page,
        )
