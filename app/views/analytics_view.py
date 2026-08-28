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
Analytics Dashboard View Component.

Renders Tab 7 analytics dashboard and pipeline timing breakdowns.
"""

import streamlit as st

from app.theme import get_chart_colors

try:
    from src.visualization.analytics import plot_processing_time_breakdown
except ImportError:
    plot_processing_time_breakdown = None


def render_analytics_view():
    """Render Tab 7: Analytics Dashboard."""
    st.subheader("📊 Analytics Dashboard")
    st.markdown("### ⏱️ Pipeline Processing Time Breakdown")
    stage_timings = st.session_state.get("last_stage_timings") or st.session_state.get(
        "stage_timings"
    )
    if plot_processing_time_breakdown:
        active_theme_colors = get_chart_colors() if callable(get_chart_colors) else None
        fig_time = plot_processing_time_breakdown(
            stage_timings=stage_timings,
            theme_colors=active_theme_colors,
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Analytics metrics summary loaded.")
