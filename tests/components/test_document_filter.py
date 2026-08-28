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

"""Unit tests for render_document_filter component in app/components/document_filter.py."""

from unittest.mock import MagicMock, patch

import pandas as pd
import streamlit as st

from app.components.document_filter import render_document_filter


def test_render_document_filter_empty_query():
    """Verify full DataFrame returned when search query is empty."""
    df = pd.DataFrame([{"Filename": "doc1.txt"}, {"Filename": "doc2.pdf"}])
    with patch("streamlit.columns") as mock_cols, patch("streamlit.text_input"), patch(
        "streamlit.button"
    ) as mock_button:
        mock_col = MagicMock()
        mock_cols.return_value = [mock_col, mock_col]
        mock_button.return_value = False

        filtered = render_document_filter(df, search_key="test_search_key")
        assert len(filtered) == 2


def test_render_document_filter_matching_query():
    """Verify DataFrame filtering when query matches."""
    df = pd.DataFrame([{"Filename": "essay.txt"}, {"Filename": "report.pdf"}])
    st.session_state["test_search_key_match"] = "essay"
    with patch("streamlit.columns") as mock_cols, patch("streamlit.text_input"), patch(
        "streamlit.button"
    ) as mock_button:
        mock_col = MagicMock()
        mock_cols.return_value = [mock_col, mock_col]
        mock_button.return_value = False

        filtered = render_document_filter(df, search_key="test_search_key_match")
        assert len(filtered) == 1
        assert filtered.iloc[0]["Filename"] == "essay.txt"


def test_render_document_filter_clear_button_clicks():
    """Verify Clear Search button resets session state and calls st.rerun()."""
    st.session_state["test_clear_key"] = "some_query"
    with patch("streamlit.columns") as mock_cols, patch("streamlit.text_input"), patch(
        "streamlit.button"
    ) as mock_button, patch("streamlit.rerun") as mock_rerun:
        mock_col = MagicMock()
        mock_cols.return_value = [mock_col, mock_col]
        mock_button.return_value = True

        render_document_filter([], search_key="test_clear_key")
        assert st.session_state["test_clear_key"] == ""
        mock_rerun.assert_called_once()
