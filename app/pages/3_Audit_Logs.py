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
app/pages/3_Audit_Logs.py
-------------------------
Streamlit multi-page app: Security Audit Logs.

This page provides a paginated, filterable view of the system security
audit trail, including login attempts, configuration changes, and data exports.

Issue #2810: Decompose monolithic streamlit_app.py.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from src.db.auth import get_distinct_audit_event_types
from src.db.security_audit import get_audit_events_count, get_recent_audit_events

st.set_page_config(
    page_title="Audit Logs - Plagiarism Detector", page_icon="📜", layout="wide"
)

EVENTS_PER_PAGE = 25


def render_audit_logs():
    """Render the security audit logs UI with pagination."""
    st.title("📜 System Security Audit Trail")

    user_role = st.session_state.get("role", "user")
    if user_role != "admin":
        st.error("🔒 Access Denied: Administrator privileges required.")
        return

    # Filters
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        try:
            distinct_events = get_distinct_audit_event_types()
        except Exception:
            distinct_events = []

        event_options = ["All Event Types"] + distinct_events
        selected_event = st.selectbox("🏷️ Event Type", options=event_options)
        event_filter = None if selected_event == "All Event Types" else selected_event

    with filter_col2:
        username_filter = st.text_input(
            "👤 Filter by Username", placeholder="Enter username..."
        ).strip()
        username_filter = username_filter if username_filter else None

    # Pagination state
    if "audit_page_offset" not in st.session_state:
        st.session_state.audit_page_offset = 0

    current_offset = st.session_state.audit_page_offset

    # Fetch data
    try:
        logs = get_recent_audit_events(
            limit=EVENTS_PER_PAGE,
            offset=current_offset,
            username=username_filter,
            event_type=event_filter,
        )

        total_records = get_audit_events_count(
            username=username_filter, event_type=event_filter
        )
    except Exception as e:
        st.error(f"Failed to load audit logs: {e}")
        return

    total_pages = max(1, (total_records + EVENTS_PER_PAGE - 1) // EVENTS_PER_PAGE)
    current_page = (current_offset // EVENTS_PER_PAGE) + 1

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("📋 Total Log Entries", total_records)
    m2.metric("🏷️ Active Filter", selected_event)
    m3.metric("📑 Page", f"{current_page} / {total_pages}")

    st.divider()

    # Data Table
    if logs:
        df = pd.DataFrame(logs)
        display_df = df.rename(
            columns={
                "id": "ID",
                "timestamp": "Timestamp (UTC)",
                "event_type": "Event Type",
                "username": "Username",
                "details": "Details / Payload",
            }
        )

        st.dataframe(
            display_df[
                ["ID", "Timestamp (UTC)", "Event Type", "Username", "Details / Payload"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        # Pagination Controls
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

        with nav_col1:
            if st.button(
                "← Previous", disabled=(current_offset == 0), use_container_width=True
            ):
                st.session_state.audit_page_offset = max(
                    0, current_offset - EVENTS_PER_PAGE
                )
                st.rerun()

        with nav_col2:
            start_range = current_offset + 1 if total_records > 0 else 0
            end_range = min(current_offset + EVENTS_PER_PAGE, total_records)
            st.caption(f"Showing {start_range} - {end_range} of {total_records} logs")

        with nav_col3:
            if st.button(
                "Next →",
                disabled=(current_offset + EVENTS_PER_PAGE >= total_records),
                use_container_width=True,
            ):
                st.session_state.audit_page_offset = current_offset + EVENTS_PER_PAGE
                st.rerun()

        # CSV Export
        st.divider()
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Audit Logs (CSV)",
            data=csv_bytes,
            file_name=f"audit_logs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary",
        )
    else:
        st.info("ℹ️ No security audit log records found matching the specified filters.")


if __name__ == "__main__":
    render_audit_logs()
