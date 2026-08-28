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
Document Similarity History View Component.

Renders Tab 10 similarity history metrics, trend line charts, bar charts,
and raw scan session log table.
"""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from app.theme import get_chart_colors
from src.db.corpus_db import get_scan_history
from src.visualization.history_charts import (
    plot_flagged_documents_bar,
    plot_similarity_trend_line,
)


def render_history_view():
    """Render Tab 10: Document Similarity History Dashboard."""
    st.subheader("📊 Document Similarity History Dashboard")
    st.caption(
        "Monitor plagiarism patterns and similarity trends across previous scan sessions."
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
            key="history_start_date",
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            key="history_end_date",
        )

    history_data = get_scan_history(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        limit=100,
    )

    if not history_data:
        st.info(
            "No scan history found for the selected date range. Run a scan to populate this dashboard."
        )
    else:
        trend_fig = plot_similarity_trend_line(
            history_data, theme_colors=get_chart_colors()
        )
        st.plotly_chart(trend_fig, use_container_width=True)

        st.divider()

        bar_fig = plot_flagged_documents_bar(
            history_data, theme_colors=get_chart_colors()
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        st.divider()

        st.markdown("### 📋 Raw Scan History Data")
        df_history = pd.DataFrame(history_data)
        df_history["timestamp"] = pd.to_datetime(df_history["timestamp"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        st.dataframe(
            df_history.style.format(
                {
                    "avg_similarity": "{:.2%}",
                    "max_similarity": "{:.2%}",
                    "threshold_used": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
