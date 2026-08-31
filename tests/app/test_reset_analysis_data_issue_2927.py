"""Unit tests for reset_analysis_data() preserving Auth and theme preferences (Issue #2927)."""

import streamlit as st
import pytest

from app.session_keys import SessionKeys
from app.state_manager import reset_analysis_data, reset_analysis_state


def test_reset_analysis_data_clears_analysis_keys_and_preserves_auth():
    """Verify reset_analysis_data removes document/results keys but preserves auth and theme."""
    st.session_state.clear()

    # Set up auth and preferences
    st.session_state[SessionKeys.AUTHENTICATED] = True
    st.session_state[SessionKeys.USERNAME] = "prof_alice"
    st.session_state[SessionKeys.ROLE] = "teacher"
    st.session_state["theme_selector"] = "Dark"
    st.session_state[SessionKeys.LANG] = "en"

    # Set up analysis keys
    st.session_state[SessionKeys.ANALYSIS_RESULTS] = {"score": 0.85}
    st.session_state[SessionKeys.ANALYSIS_FILE_SIGNATURE] = "abc123sig"
    st.session_state[SessionKeys.FAILED_DOCUMENTS] = ["empty.pdf"]
    st.session_state["file_uploader_1"] = b"raw_data"
    st.session_state["doc_chunk_list"] = ["chunk1", "chunk2"]
    st.session_state["analysis_metadata"] = {"pages": 12}

    reset_analysis_data()

    # Assert Auth and theme preferences are intact
    assert st.session_state.get(SessionKeys.AUTHENTICATED) is True
    assert st.session_state.get(SessionKeys.USERNAME) == "prof_alice"
    assert st.session_state.get(SessionKeys.ROLE) == "teacher"
    assert st.session_state.get("theme_selector") == "Dark"
    assert st.session_state.get(SessionKeys.LANG) == "en"

    # Assert analysis / document keys are deleted
    assert SessionKeys.ANALYSIS_RESULTS not in st.session_state
    assert SessionKeys.ANALYSIS_FILE_SIGNATURE not in st.session_state
    assert SessionKeys.FAILED_DOCUMENTS not in st.session_state
    assert "file_uploader_1" not in st.session_state
    assert "doc_chunk_list" not in st.session_state
    assert "analysis_metadata" not in st.session_state


def test_reset_analysis_state_alias_behavior():
    """Verify reset_analysis_state alias functions identically."""
    st.session_state.clear()
    st.session_state[SessionKeys.AUTHENTICATED] = True
    st.session_state[SessionKeys.USERNAME] = "student_bob"
    st.session_state[SessionKeys.ANALYSIS_RESULTS] = {"matches": 3}

    reset_analysis_state()

    assert st.session_state.get(SessionKeys.AUTHENTICATED) is True
    assert st.session_state.get(SessionKeys.USERNAME) == "student_bob"
    assert SessionKeys.ANALYSIS_RESULTS not in st.session_state
