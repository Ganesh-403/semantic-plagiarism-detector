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
app/pages/1_Dashboard.py
------------------------
Streamlit multi-page app: Main Analytics Dashboard.

This page displays the high-level plagiarism detection metrics, including
total scans, flagged incidents, severity distributions, and trend charts.

Issue #2810: Decompose monolithic streamlit_app.py.
"""

from datetime import datetime

import streamlit as st

from src.core.app_config import get_branding_config
from src.db.auth import get_upload_count
from src.db.incidents import get_all_incidents

# Import core utilities and visualizations
from src.visualization.analytics import (
    calculate_severity_ratios,
    plot_high_severity_trends,
    plot_most_plagiarized_documents,
    plot_severity_donut_chart,
)

# Page configuration
st.set_page_config(
    page_title="Dashboard - Plagiarism Detector", page_icon="📊", layout="wide"
)


def render_dashboard():
    """Render the main analytics dashboard UI."""
    branding = get_branding_config()

    st.title(f"📊 {branding.get('app_name', 'Analytics Dashboard')}")
    st.markdown("High-level overview of plagiarism detection metrics and trends.")

    # Fetch data
    try:
        total_scans = get_upload_count()
        incidents = get_all_incidents(limit=10000)
    except Exception as e:
        st.error(f"Failed to load dashboard metrics: {e}")
        return

    flagged_incidents = len(incidents)

    # Top-level metrics cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Scans", f"{total_scans:,}")
    col2.metric("Flagged Incidents", f"{flagged_incidents:,}")

    # Calculate severity ratios
    severity_ratios = calculate_severity_ratios(incidents)
    col3.metric("High Severity", f"{severity_ratios.get('High', 0.0):.1f}%")
    col4.metric("Medium Severity", f"{severity_ratios.get('Medium', 0.0):.1f}%")

    st.divider()

    # Visualizations
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Severity Distribution")
        if incidents:
            fig_donut = plot_severity_donut_chart(incidents)
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No incidents to display.")

    with chart_col2:
        st.subheader("Most Plagiarized Documents")
        if incidents:
            # Aggregate incidents by document
            doc_counts = {}
            for inc in incidents:
                doc_a = inc.get("document_a", "Unknown")
                doc_counts[doc_a] = doc_counts.get(doc_a, 0) + 1

            doc_data = [
                {"document_name": k, "incident_count": v}
                for k, v in sorted(
                    doc_counts.items(), key=lambda x: x[1], reverse=True
                )[:10]
            ]

            if doc_data:
                fig_bar = plot_most_plagiarized_documents(doc_data)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No document data available.")
        else:
            st.info("No incidents to display.")

    st.divider()

    # Trend chart
    st.subheader("High Severity Trends (Last 30 Days)")
    if incidents:
        # Group by date
        trend_data = {}
        for inc in incidents:
            ts = inc.get("timestamp") or inc.get("date_flagged")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                    if inc.get("severity") == "High" or (
                        inc.get("similarity", 0) >= 0.80
                    ):
                        trend_data[date_str] = trend_data.get(date_str, 0) + 1
                except ValueError:
                    pass

        formatted_trend = [
            {"date": k, "count": v} for k, v in sorted(trend_data.items())
        ]

        if formatted_trend:
            fig_line = plot_high_severity_trends(formatted_trend)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No high severity trends in the last 30 days.")
    else:
        st.info("No incidents to display.")


if __name__ == "__main__":
    render_dashboard()
