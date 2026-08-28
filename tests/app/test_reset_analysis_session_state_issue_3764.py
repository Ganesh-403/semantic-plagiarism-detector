"""Tests for reset_analysis_session_state() (Issue #3764)."""

import streamlit as st

from app.session_keys import SessionKeys
from app.state_manager import reset_analysis_session_state


def test_reset_analysis_session_state_clears_analysis_keeps_theme_and_session_id():
    st.session_state.clear()

    st.session_state[SessionKeys.SESSION_ID] = "sess-abc"
    st.session_state[SessionKeys.THEME] = "Dark"
    st.session_state[SessionKeys.FAILED_DOCUMENTS] = ["bad.pdf"]
    st.session_state[SessionKeys.DRIVE_FILES_DICT] = {"a.txt": b"data"}
    st.session_state[SessionKeys.ANALYSIS_RESULTS] = {"matrix": [[1.0, 0.2], [0.2, 1.0]]}
    st.session_state[SessionKeys.ANALYSIS_FILE_SIGNATURE] = "sig-1"
    st.session_state[SessionKeys.SELECTED_DOCUMENT_ID] = "doc-9"
    st.session_state[SessionKeys.SCANNING] = True
    st.session_state[SessionKeys.AUDIT_REPORT_GENERATED] = True
    st.session_state[SessionKeys.SENT_ALERTS] = {"alert-1"}

    reset_analysis_session_state()

    assert st.session_state[SessionKeys.SESSION_ID] == "sess-abc"
    assert st.session_state[SessionKeys.THEME] == "Dark"
    assert SessionKeys.FAILED_DOCUMENTS not in st.session_state
    assert SessionKeys.DRIVE_FILES_DICT not in st.session_state
    assert SessionKeys.ANALYSIS_RESULTS not in st.session_state
    assert SessionKeys.ANALYSIS_FILE_SIGNATURE not in st.session_state
    assert SessionKeys.SELECTED_DOCUMENT_ID not in st.session_state
    assert SessionKeys.SCANNING not in st.session_state
    assert SessionKeys.AUDIT_REPORT_GENERATED not in st.session_state
    assert SessionKeys.SENT_ALERTS not in st.session_state
