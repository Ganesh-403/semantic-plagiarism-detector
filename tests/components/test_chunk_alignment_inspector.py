from unittest.mock import patch, MagicMock
import sys

sys.modules['streamlit'] = MagicMock()
from app.components.chunk_alignment_inspector import render_chunk_alignment_inspector

@patch("app.components.chunk_alignment_inspector.st.markdown")
@patch("app.components.chunk_alignment_inspector.st.columns")
@patch("app.components.chunk_alignment_inspector.st.info")
def test_render_chunk_alignment_inspector(mock_info, mock_columns, mock_markdown):
    # Mock Streamlit columns
    class MockCol:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    mock_columns.return_value = (MockCol(), MockCol())

    chunk_a = "This is a test document."
    chunk_b = "This is another test document."
    score = 0.85

    render_chunk_alignment_inspector(chunk_a, chunk_b, score)

    # Verify score badge is displayed
    assert any("Similarity Score:" in str(call_args) and "85.00%" in str(call_args) for call_args, _ in mock_markdown.call_args_list)

    # Verify matching words are highlighted with the identical style
    # "This is" and "test document." are matching.
    markdown_calls_html = [args[0] for args, _ in mock_markdown.call_args_list if len(args) > 0 and isinstance(args[0], str) and "<div" in args[0]]
    assert len(markdown_calls_html) == 2  # One for chunk_a, one for chunk_b
    
    html_a, html_b = markdown_calls_html
    assert "background-color: #ffeb3b" in html_a
    assert "This is" in html_a
    assert "test document." in html_a

@patch("app.components.chunk_alignment_inspector.st.markdown")
@patch("app.components.chunk_alignment_inspector.st.info")
def test_render_chunk_alignment_inspector_empty(mock_info, mock_markdown):
    render_chunk_alignment_inspector("", "", 0.0)
    mock_info.assert_called_once_with("Both chunks are empty.")
    
    # It shouldn't render the alignment chunks if empty
    assert not any("<div" in str(args) for args, _ in mock_markdown.call_args_list)
