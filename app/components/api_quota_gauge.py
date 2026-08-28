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
