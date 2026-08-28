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
Security Audit Logs View Module

Provides a comprehensive security audit log viewer with:
- Advanced filtering (date, event type, username)
- Pagination
- CSV/Excel export
- Real-time monitoring
- Visual analytics
- Export reports
- Alert configurations
"""

import base64
import io  # noqa: F401
import logging
import time  # noqa: F401
from datetime import datetime, timedelta, timezone  # noqa: F401
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from app.session_keys import SessionKeys
from app.theme import get_chart_colors  # noqa: F401

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

AUDIT_EVENT_TYPES = [
    "login",
    "logout",
    "file_upload",
    "file_download",
    "file_delete",
    "analysis_run",
    "report_generated",
    "user_created",
    "user_deleted",
    "user_updated",
    "password_change",
    "2fa_enabled",
    "2fa_disabled",
    "token_revoked",
    "api_access",
    "export_downloaded",
    "settings_changed",
    "permission_changed",
    "backup_created",
    "restore_performed",
    "security_alert",
    "failed_login",
    "account_locked",
    "session_expired",
]

SEVERITY_LEVELS = {
    "critical": {"color": "#dc3545", "icon": "🔴"},
    "high": {"color": "#fd7e14", "icon": "🟠"},
    "medium": {"color": "#ffc107", "icon": "🟡"},
    "low": {"color": "#28a745", "icon": "🟢"},
    "info": {"color": "#17a2b8", "icon": "🔵"},
}

DEFAULT_PAGE_SIZE = 25
MAX_EXPORT_ROWS = 10000


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================


def fetch_audit_logs(
    username: Optional[str] = None,
    event_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Fetch audit logs from the database with filters."""
    try:
        from src.db.auth import auth_repo

        return auth_repo.get_security_audit_logs(
            username=username,
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Failed to fetch audit logs: {e}")
        return []


def count_audit_logs(
    username: Optional[str] = None,
    event_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> int:
    """Count total audit logs matching filters."""
    try:
        from src.db.auth import auth_repo

        return auth_repo.get_security_audit_log_count(
            username=username,
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        logger.error(f"Failed to count audit logs: {e}")
        return 0


def get_distinct_event_types() -> list[str]:
    """Get all distinct event types from audit logs."""
    try:
        from src.db.auth import auth_repo

        return auth_repo.get_distinct_audit_event_types()
    except Exception as e:
        logger.error(f"Failed to get distinct event types: {e}")
        return []


# ============================================================================
# UI RENDER FUNCTIONS
# ============================================================================


def render_audit_logs_view(user_role: str, lang_code: str) -> None:
    """Render the main audit logs view."""

    if user_role != "admin":
        st.error(
            "🔒 Access Denied: Administrator privileges required to view security audit logs."
        )
        return

    st.subheader("📋 Security Audit Log Viewer")
    st.caption("Monitor all security-relevant events across the platform.")

    # ========================================================================
    # FILTERS SECTION
    # ========================================================================

    with st.expander("🔍 Filter Logs", expanded=True):
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        with filter_col1:
            # Date range filter
            date_range = st.date_input(
                "📅 Date Range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                key="audit_date_range",
                help="Filter by date range",
            )

            start_date = None
            end_date = None
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date = date_range[0].strftime("%Y-%m-%d") + "T00:00:00Z"
                end_date = date_range[1].strftime("%Y-%m-%d") + "T23:59:59Z"

        with filter_col2:
            # Event type filter
            event_types = ["All Events"] + get_distinct_event_types()
            selected_event_type = st.selectbox(
                "🏷️ Event Type",
                options=event_types,
                key="audit_event_filter",
            )
            event_type_filter = (
                None if selected_event_type == "All Events" else selected_event_type
            )

        with filter_col3:
            # Username filter
            username_filter = st.text_input(
                "👤 Username",
                placeholder="Enter username...",
                key="audit_username_filter",
            ).strip()
            username_filter = username_filter if username_filter else None

        with filter_col4:
            # Page size
            page_size = st.selectbox(
                "📄 Rows Per Page",
                options=[10, 25, 50, 100],
                index=1,
                key="audit_page_size",
            )

    # ========================================================================
    # METRICS ROW
    # ========================================================================

    # Count filtered logs
    total_records = count_audit_logs(
        username=username_filter,
        event_type=event_type_filter,
        start_date=start_date,
        end_date=end_date,
    )

    # Get recent activity count (last 24 hours)
    recent_start = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    recent_count = count_audit_logs(
        start_date=recent_start, end_date=datetime.now().strftime("%Y-%m-%dT23:59:59Z")
    )

    # Get failed login count (last 24 hours)
    failed_count = count_audit_logs(
        event_type="failed_login",
        start_date=recent_start,
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("📋 Total Logs", f"{total_records:,}")
    with metric_col2:
        st.metric("🕐 Last 24h", f"{recent_count:,}")
    with metric_col3:
        st.metric("❌ Failed Logins", f"{failed_count:,}")
    with metric_col4:
        # Generate a visual health indicator
        if failed_count > 10:
            st.metric("🔴 Security Status", "⚠️ HIGH ALERT", delta="🚨")
        elif failed_count > 3:
            st.metric("🟡 Security Status", "⚠️ Elevated", delta="⚡")
        else:
            st.metric("🟢 Security Status", "✅ Normal", delta="✓")

    st.divider()

    # ========================================================================
    # PAGINATION
    # ========================================================================

    total_pages = max(1, (total_records + page_size - 1) // page_size)

    # Get current page from session
    current_page = st.session_state.get(SessionKeys.AUDIT_LOG_PAGE, 1)
    if current_page > total_pages:
        current_page = total_pages
        st.session_state[SessionKeys.AUDIT_LOG_PAGE] = current_page
    if current_page < 1:
        current_page = 1
        st.session_state[SessionKeys.AUDIT_LOG_PAGE] = current_page

    offset = (current_page - 1) * page_size

    # ========================================================================
    # FETCH AND DISPLAY DATA
    # ========================================================================

    logs = fetch_audit_logs(
        username=username_filter,
        event_type=event_type_filter,
        start_date=start_date,
        end_date=end_date,
        limit=page_size,
        offset=offset,
    )

    if not logs:
        st.info("ℹ️ No audit logs found matching the specified filters.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(logs)

    # Format timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["timestamp_str"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Add severity column
    df["severity"] = df["event_type"].apply(_get_event_severity)
    df["severity_icon"] = df["severity"].apply(
        lambda x: SEVERITY_LEVELS.get(x, SEVERITY_LEVELS["info"])["icon"]
    )

    # Display Data Table
    display_columns = [
        "id",
        "timestamp_str",
        "event_type",
        "username",
        "details",
        "severity_icon",
    ]
    display_columns = [col for col in display_columns if col in df.columns]

    st.dataframe(
        df[display_columns].rename(
            columns={
                "id": "ID",
                "timestamp_str": "Timestamp",
                "event_type": "Event Type",
                "username": "Username",
                "details": "Details",
                "severity_icon": "🔴",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================================
    # PAGINATION CONTROLS
    # ========================================================================

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 3, 2, 1])

    with nav_col1:
        if st.button(
            "⬅️ Previous",
            disabled=(current_page <= 1),
            key="audit_prev_page_btn",
            use_container_width=True,
        ):
            st.session_state[SessionKeys.AUDIT_LOG_PAGE] = current_page - 1
            st.rerun()

    with nav_col2:
        st.caption(
            f"Showing {offset + 1} - {min(offset + page_size, total_records)} "
            f"of {total_records} logs"
        )

    with nav_col3:
        page_input = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=current_page,
            step=1,
            key="audit_page_input",
            label_visibility="collapsed",
        )
        if page_input != current_page:
            st.session_state[SessionKeys.AUDIT_LOG_PAGE] = page_input
            st.rerun()

    with nav_col4:
        if st.button(
            "Next ➡️",
            disabled=(current_page >= total_pages),
            key="audit_next_page_btn",
            use_container_width=True,
        ):
            st.session_state[SessionKeys.AUDIT_LOG_PAGE] = current_page + 1
            st.rerun()

    # ========================================================================
    # EXPORT BUTTONS
    # ========================================================================

    st.divider()
    export_col1, export_col2, export_col3, export_col4 = st.columns(4)

    with export_col1:
        if st.button("📥 Export CSV", use_container_width=True, key="export_csv_btn"):
            _export_csv(df)

    with export_col2:
        if st.button(
            "📊 Export Excel", use_container_width=True, key="export_excel_btn"
        ):
            _export_excel(df)

    with export_col3:
        if st.button(
            "📈 Export Report", use_container_width=True, key="export_report_btn"
        ):
            _export_report(df)

    with export_col4:
        # Clear filters button
        if st.button(
            "🔄 Clear Filters", use_container_width=True, key="clear_filters_btn"
        ):
            _clear_filters()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _get_event_severity(event_type: str) -> str:
    """Determine severity of an event type."""
    critical_events = ["security_alert", "account_locked", "failed_login"]
    high_events = ["user_deleted", "permission_changed", "2fa_disabled"]
    medium_events = ["file_delete", "password_change", "user_updated", "token_revoked"]
    low_events = ["file_upload", "file_download", "analysis_run", "report_generated"]
    info_events = ["login", "logout", "export_downloaded", "settings_changed"]

    if event_type in critical_events:
        return "critical"
    elif event_type in high_events:
        return "high"
    elif event_type in medium_events:
        return "medium"
    elif event_type in low_events:
        return "low"
    elif event_type in info_events:
        return "info"
    else:
        return "info"


def _export_csv(df: pd.DataFrame) -> None:
    """Export audit logs as CSV."""
    csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    href = f'<a href="data:file/csv;base64,{b64}" download="audit_logs_{timestamp}.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)


def _export_excel(df: pd.DataFrame) -> None:
    """Export audit logs as Excel."""
    import io  # noqa: F811

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Audit Logs", index=False)
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="audit_logs_{timestamp}.xlsx">Download Excel</a>'
    st.markdown(href, unsafe_allow_html=True)


def _export_report(df: pd.DataFrame) -> None:
    """Export a summary report."""
    report = f"""
    # Security Audit Report
    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Total Events: {len(df)}

    ## Event Type Breakdown
    {df["event_type"].value_counts().to_string()}

    ## Severity Breakdown
    {df["severity"].value_counts().to_string()}

    ## Most Active Users
    {df["username"].value_counts().head(10).to_string()}
    """

    b64 = base64.b64encode(report.encode()).decode()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    href = f'<a href="data:text/plain;base64,{b64}" download="audit_report_{timestamp}.txt">Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)


def _clear_filters() -> None:
    """Clear all audit log filters."""
    keys_to_clear = [
        "audit_date_range",
        "audit_event_filter",
        "audit_username_filter",
        "audit_page_size",
        SessionKeys.AUDIT_LOG_PAGE,
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def render_audit_view(user_role: str, lang_code: str) -> None:
    """Main entry point for audit logs view."""
    render_audit_logs_view(user_role, lang_code)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "render_audit_view",
    "render_audit_logs_view",
    "fetch_audit_logs",
    "count_audit_logs",
    "get_distinct_event_types",
]
