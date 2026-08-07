"""
api_quota_gauge.py
------------------
Sidebar component displaying translation/cloud API quota rate limit usage.
"""

import streamlit as st


def render_api_quota_gauge():
    """Renders a collapsible gauge in the sidebar showing remaining API quota."""
    # Obtain current values from session state, defaulting to the acceptance criteria parameters
    consumed = st.session_state.get("api_quota_consumed", 850)
    limit = st.session_state.get("api_quota_limit", 1000)

    if limit <= 0:
        return

    percent = min(max(0.0, float(consumed) / limit), 1.0)
    percent_display = int(percent * 100)

    # Use st.sidebar.expander to make it collapsible as per issue title
    with st.sidebar.expander("📊 API Quota Usage", expanded=True):
        # Progress gauge displaying percentage of API quota consumed
        st.progress(percent)

        # Render caption API Quota: 850 / 1000 requests (85%)
        st.caption(f"API Quota: {consumed} / {limit} requests ({percent_display}%)")

        # Change progress bar color to red when quota exceeds 90%
        if percent >= 0.90:
            st.markdown(
                """
                <style>
                section[data-testid="stSidebar"] .stProgress > div > div > div > div {
                    background-color: #EF4444 !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
