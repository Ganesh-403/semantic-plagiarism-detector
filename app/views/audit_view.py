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
Security Audit Logs View Component.

Renders Tab 11 system audit trail dataframe with date/user/event filters,
pagination, and CSV bulk export.
"""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from app.session_keys import SessionKeys
from src.db.auth import auth_repo
from src.i18n.translator import get_text


def render_audit_view(user_role: str, lang_code: str):
    """Render Tab 11: Security Audit Logs."""
    st.subheader(get_text("tab_audit_logs", lang=lang_code))

    if user_role != "admin":
        st.error(
            "🔒 Access Denied: Administrator privileges required to view security audit logs."
        )
    else:
        st.markdown("### 📜 System Security Audit Trail")

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            date_range = st.date_input(
                "📅 Date Range Filter",
                value=(),
                key="audit_date_range_picker",
                help="Filter audit log records by date range.",
            )

        start_date_str = None
        end_date_str = None
        if isinstance(date_range, (list, tuple)) and len(date_range) > 0:
            if len(date_range) == 1:
                start_date_str = date_range[0].strftime("%Y-%m-%d") + "T00:00:00Z"
                end_date_str = date_range[0].strftime("%Y-%m-%d") + "T23:59:59Z"
            elif len(date_range) == 2:
                start_date_str = date_range[0].strftime("%Y-%m-%d") + "T00:00:00Z"
                end_date_str = date_range[1].strftime("%Y-%m-%d") + "T23:59:59Z"

        with filter_col2:
            distinct_events = auth_repo.get_distinct_audit_event_types()
            event_type_options = ["All Event Types"] + distinct_events
            selected_event_type = st.selectbox(
                "🏷️ Event Type",
                options=event_type_options,
                key="audit_event_type_filter",
            )
            event_type_filter = (
                None
                if selected_event_type == "All Event Types"
                else selected_event_type
            )

        with filter_col3:
            username_filter_input = st.text_input(
                "👤 Filter by Username",
                value="",
                placeholder="Enter username...",
                key="audit_username_filter",
            ).strip()
            username_filter = username_filter_input if username_filter_input else None

        with filter_col4:
            per_page = st.selectbox(
                "📄 Rows Per Page",
                options=[10, 25, 50, 100],
                index=1,
                key="audit_per_page_select",
            )

        total_records = auth_repo.get_security_audit_log_count(
            username=username_filter,
            event_type=event_type_filter,
            start_date=start_date_str,
            end_date=end_date_str,
        )

        total_pages = max(1, (total_records + per_page - 1) // per_page)

        current_page = st.session_state.get(SessionKeys.AUDIT_LOG_PAGE, 1)
        if current_page > total_pages:
            current_page = total_pages
            st.session_state[SessionKeys.AUDIT_LOG_PAGE] = current_page
        if current_page < 1:
            current_page = 1
            st.session_state[SessionKeys.AUDIT_LOG_PAGE] = current_page

        offset = (current_page - 1) * per_page

        logs = auth_repo.get_security_audit_logs(
            username=username_filter,
            event_type=event_type_filter,
            start_date=start_date_str,
            end_date=end_date_str,
            limit=per_page,
            offset=offset,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("📋 Total Log Entries", total_records)
        m2.metric("🏷️ Active Filter", selected_event_type)
        m3.metric("📑 Page", f"{current_page} / {total_pages}")

        st.divider()

        if logs:
            df = pd.DataFrame(logs)
            display_df = df[
                ["id", "timestamp", "event_type", "username", "details"]
            ].rename(
                columns={
                    "id": "ID",
                    "timestamp": "Timestamp (UTC)",
                    "event_type": "Event Type",
                    "username": "Username",
                    "details": "Details / Payload",
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ID": st.column_config.NumberColumn("ID", width="small"),
                    "Timestamp (UTC)": st.column_config.TextColumn(
                        "Timestamp (UTC)", width="medium"
                    ),
                    "Event Type": st.column_config.TextColumn(
                        "Event Type", width="medium"
                    ),
                    "Username": st.column_config.TextColumn("Username", width="medium"),
                    "Details / Payload": st.column_config.TextColumn(
                        "Details / Payload", width="large"
                    ),
                },
            )

            nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 2, 2, 1])
            with nav_col1:
                if st.button(
                    "← Previous",
                    disabled=(current_page <= 1),
                    key="audit_prev_page",
                ):
                    st.session_state[SessionKeys.AUDIT_LOG_PAGE] = current_page - 1
                    st.rerun()

            with nav_col2:
                end_range = min(offset + per_page, total_records)
                start_range = offset + 1 if total_records > 0 else 0
                st.caption(
                    f"Showing {start_range} - {end_range} of {total_records} logs"
                )

            with nav_col3:
                page_select = st.number_input(
                    "Go to Page",
                    min_value=1,
                    max_value=total_pages,
                    value=current_page,
                    step=1,
                    key="audit_page_num_input",
                )
                if page_select != current_page:
                    st.session_state[SessionKeys.AUDIT_LOG_PAGE] = page_select
                    st.rerun()

            with nav_col4:
                if st.button(
                    "Next →",
                    disabled=(current_page >= total_pages),
                    key="audit_next_page",
                ):
                    st.session_state[SessionKeys.AUDIT_LOG_PAGE] = current_page + 1
                    st.rerun()

            st.divider()

            export_all_logs = auth_repo.get_security_audit_logs(
                username=username_filter,
                event_type=event_type_filter,
                start_date=start_date_str,
                end_date=end_date_str,
                limit=10000,
                offset=0,
            )
            export_df = pd.DataFrame(export_all_logs)
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Audit Logs (CSV)",
                data=csv_bytes,
                file_name=f"security_audit_logs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_audit_logs_csv",
                use_container_width=True,
                type="primary",
            )
        else:
            st.info(
                "ℹ️ No security audit log records found matching the specified filters."
            )
