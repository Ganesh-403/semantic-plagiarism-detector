from streamlit.testing.v1 import AppTest


def test_staged_files_counter_badge():
    # AppTest does not support file_uploader uploads. Exercise its rendering
    # boundary with the same size metadata returned by UploadedFile.
    at = AppTest.from_string("""
from types import SimpleNamespace
import streamlit as st
from app.views.upload_view import render_staged_files_summary
files = [SimpleNamespace(size=size) for size in st.session_state.get("sizes", [])]
render_staged_files_summary(files)
""").run()
    assert not at.exception
    assert not at.info
    at.session_state["sizes"] = [1024 * 1024, 512 * 1024]
    at.run()
    assert not at.exception
    assert at.session_state["staged_files_count"] == 2
    assert at.session_state["staged_files_size"] == 1572864
    assert "Staged 2 files (Total Size: 1.5 MB)" in at.info[0].value
    at.session_state["sizes"] = []
    at.run()
    assert at.session_state["staged_files_count"] == 0
    assert at.session_state["staged_files_size"] == 0
    assert not at.info
