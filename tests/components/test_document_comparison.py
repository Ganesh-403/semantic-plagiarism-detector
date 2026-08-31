"""Unit tests for the render_document_comparison component in app/components/document_comparison.py."""

from unittest.mock import MagicMock, patch

import streamlit as st

from app.components.document_comparison import render_document_comparison


def test_render_document_comparison_inputs_empty():
    """Verify component renders correctly with empty inputs."""
    with (
        patch("streamlit.columns") as mock_cols,
        patch("streamlit.text_area") as mock_text_area,
        patch("streamlit.popover") as mock_popover,
    ):  # noqa: F841
        mock_col = MagicMock()
        mock_cols.return_value = [mock_col, mock_col]
        mock_text_area.return_value = ""

        render_document_comparison()
        assert st.session_state["comp_doc_a"] == ""
        assert st.session_state["comp_doc_b"] == ""


def test_render_document_comparison_highlight_overlap():
    """Verify side-by-side highlighting comparison is executed when both inputs are filled."""
    st.session_state["comp_doc_a"] = "This is a document content."
    st.session_state["comp_doc_b"] = "This is another document content."

    with (
        patch("streamlit.columns") as mock_cols,
        patch("streamlit.text_area") as mock_text_area,
        patch("streamlit.popover") as mock_popover,
        patch(  # noqa: F841
            "streamlit.markdown"
        ) as mock_markdown,
    ):
        mock_col = MagicMock()
        mock_cols.return_value = [mock_col, mock_col]
        mock_text_area.side_effect = [
            "This is a document content.",
            "This is another document content.",
        ]

        render_document_comparison()

        # Verify markdown highlights are printed
        calls = [
            args[0]
            for args, _ in mock_markdown.call_args_list
            if isinstance(args[0], str)
        ]
        assert any("Document A (Extracted Overlap)" in c for c in calls)
        assert any("Document B (Extracted Overlap)" in c for c in calls)


def test_render_document_comparison_clear_popover_clicks():
    """Verify that clicking Clear Comparison resets inputs in session state and reruns."""
    st.session_state["comp_doc_a"] = "staged text a"
    st.session_state["comp_doc_b"] = "staged text b"
    st.session_state["comp_doc_a_input"] = "staged text a"
    st.session_state["comp_doc_b_input"] = "staged text b"

    class MockPopoverContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    with (
        patch("streamlit.columns") as mock_cols,
        patch("streamlit.text_area") as mock_text_area,
        patch("streamlit.popover") as mock_popover,
        patch("streamlit.button") as mock_button,
        patch("streamlit.rerun") as mock_rerun,
    ):
        mock_col = MagicMock()
        mock_cols.return_value = [mock_col, mock_col]
        mock_text_area.side_effect = ["staged text a", "staged text b"]
        mock_popover.return_value = MockPopoverContext()
        mock_button.return_value = True

        render_document_comparison()

        assert st.session_state["comp_doc_a"] == ""
        assert st.session_state["comp_doc_b"] == ""
        assert st.session_state["comp_doc_a_input"] == ""
        assert st.session_state["comp_doc_b_input"] == ""
        mock_rerun.assert_called_once()
