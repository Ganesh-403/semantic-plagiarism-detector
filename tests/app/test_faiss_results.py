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

import streamlit as st

# Mock st.dialog to be a no-op decorator before faiss_results is imported
st.dialog = lambda *args, **kwargs: lambda f: f

import numpy as np
import pandas as pd

from app.components.faiss_results import RESULT_COLUMNS, faiss_results_dataframe


class MockRecord:
    def __init__(self, doc_name: str, chunk_index: int, chunk_text: str):
        self.doc_name = doc_name
        self.chunk_index = chunk_index
        self.chunk_text = chunk_text


def test_faiss_results_dataframe_empty():
    """Verify empty input returns empty DataFrame with correct columns."""
    df = faiss_results_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == RESULT_COLUMNS
    assert len(df) == 0


def test_faiss_results_dataframe_valid():
    """Verify valid results are correctly formatted, sorted, and ranked."""
    results = [
        (MockRecord("doc_a.pdf", 0, "This is chunk 1 text."), 0.75),
        (
            {
                "doc_name": "doc_b.pdf",
                "chunk_index": 1,
                "chunk_text": "This is chunk 2 text.",
            },
            0.92,
        ),
        (MockRecord("doc_c.pdf", 2, "This is chunk 3 text."), 0.60),
    ]

    df = faiss_results_dataframe(results)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == RESULT_COLUMNS
    assert len(df) == 3

    # Ranking & Sorting (highest score first)
    assert df.loc[0, "Rank"] == 1
    assert df.loc[0, "Target Document"] == "doc_b.pdf"
    assert df.loc[0, "Similarity Score"] == 0.92

    assert df.loc[1, "Rank"] == 2
    assert df.loc[1, "Target Document"] == "doc_a.pdf"
    assert df.loc[1, "Similarity Score"] == 0.75

    assert df.loc[2, "Rank"] == 3
    assert df.loc[2, "Target Document"] == "doc_c.pdf"
    assert df.loc[2, "Similarity Score"] == 0.60


def test_faiss_results_dataframe_similarity_filters():
    """Verify min_similarity and max_similarity filtering works."""
    results = [
        (MockRecord("doc_a.pdf", 0, "text1"), 0.50),
        (MockRecord("doc_b.pdf", 1, "text2"), 0.70),
        (MockRecord("doc_c.pdf", 2, "text3"), 0.90),
    ]

    # Min similarity filter
    df_min = faiss_results_dataframe(results, min_similarity=0.70)
    assert len(df_min) == 2
    assert "doc_a.pdf" not in df_min["Target Document"].values

    # Max similarity filter
    df_max = faiss_results_dataframe(results, max_similarity=0.80)
    assert len(df_max) == 2
    assert "doc_c.pdf" not in df_max["Target Document"].values

    # Both filters
    df_both = faiss_results_dataframe(results, min_similarity=0.60, max_similarity=0.80)
    assert len(df_both) == 1
    assert df_both.iloc[0]["Target Document"] == "doc_b.pdf"


def test_faiss_results_dataframe_nan_and_inf_scores():
    """Verify handling of NaN and Inf scores does not crash and works gracefully."""
    results = [
        (MockRecord("doc_a.pdf", 0, "text1"), float("nan")),
        (MockRecord("doc_b.pdf", 1, "text2"), float("inf")),
        (MockRecord("doc_c.pdf", 2, "text3"), 0.85),
    ]

    df = faiss_results_dataframe(results)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

    # inf score should sort first, then 0.85, then nan should be sorted at the end
    assert df.iloc[0]["Target Document"] == "doc_b.pdf"
    assert df.iloc[0]["Similarity Score"] == float("inf")

    assert df.iloc[1]["Target Document"] == "doc_c.pdf"
    assert df.iloc[1]["Similarity Score"] == 0.85

    assert df.iloc[2]["Target Document"] == "doc_a.pdf"
    assert np.isnan(df.iloc[2]["Similarity Score"])


def test_faiss_results_dataframe_missing_attributes():
    """Verify default values are used when fields are missing from records."""
    # Record has no doc_name or chunk_index attributes
    results = [
        (object(), 0.80),
    ]

    df = faiss_results_dataframe(results)
    assert len(df) == 1
    assert df.iloc[0]["Target Document"] == "Unknown document"
    assert df.iloc[0]["Chunk"] == 1  # chunk_index (default 0) + 1


def test_inspect_diff_dialog():
    """Verify inspect_diff_dialog calls streamlit elements to render comparison."""
    from unittest.mock import MagicMock, patch

    from app.components.faiss_results import inspect_diff_dialog

    with patch("app.components.faiss_results.st") as mock_st:
        col1, col2 = MagicMock(), MagicMock()
        mock_st.columns.return_value = (col1, col2)

        inspect_diff_dialog(
            "query text sample", "matched text sample", "test_doc.pdf", 0.85
        )

        mock_st.markdown.assert_any_call("### Match Similarity: **85.0%**")
        mock_st.columns.assert_called_once_with(2)


def test_render_faiss_results_ui():
    """Verify render_faiss_results_ui renders list of results and handles click."""
    from unittest.mock import patch

    from app.components.faiss_results import render_faiss_results_ui

    results = [
        (MockRecord("doc_a.pdf", 2, "Matched text here"), 0.88),
    ]
    with patch("app.components.faiss_results.st") as mock_st, patch(
        "app.components.faiss_results.inspect_diff_dialog"
    ) as mock_dialog:
        mock_st.button.return_value = True

        render_faiss_results_ui(results, "query text")

        mock_st.button.assert_called_once()
        mock_dialog.assert_called_once_with(
            "query text", "Matched text here", "doc_a.pdf", 0.88
        )


def test_render_faiss_results_ui_passes_matching_pdf_bytes():
    """When a source PDF is available for the matched document, it is forwarded to the dialog."""
    from unittest.mock import patch

    from app.components.faiss_results import render_faiss_results_ui

    results = [
        (MockRecord("doc_a.pdf", 2, "Matched text here"), 0.88),
    ]
    source_bytes = b"%PDF-1.4 raw bytes"

    with patch("app.components.faiss_results.st") as mock_st, patch(
        "app.components.faiss_results.inspect_diff_dialog"
    ) as mock_dialog:
        mock_st.button.return_value = True

        render_faiss_results_ui(
            results, "query text", document_pdf_bytes={"doc_a.pdf": source_bytes}
        )

        mock_dialog.assert_called_once_with(
            "query text", "Matched text here", "doc_a.pdf", 0.88, pdf_bytes=source_bytes
        )


def test_render_faiss_results_ui_no_matching_pdf_bytes():
    """When the matched document isn't in document_pdf_bytes, behavior is unchanged."""
    from unittest.mock import patch

    from app.components.faiss_results import render_faiss_results_ui

    results = [
        (MockRecord("doc_a.pdf", 2, "Matched text here"), 0.88),
    ]

    with patch("app.components.faiss_results.st") as mock_st, patch(
        "app.components.faiss_results.inspect_diff_dialog"
    ) as mock_dialog:
        mock_st.button.return_value = True

        render_faiss_results_ui(
            results, "query text", document_pdf_bytes={"other_doc.pdf": b"bytes"}
        )

        mock_dialog.assert_called_once_with(
            "query text", "Matched text here", "doc_a.pdf", 0.88
        )


def test_inspect_diff_dialog_offers_pdf_download_when_highlighting_succeeds():
    """Verify a 'Download Highlighted PDF' button is rendered using annotated bytes."""
    from unittest.mock import MagicMock, patch

    from app.components.faiss_results import inspect_diff_dialog

    fake_pdf_bytes = b"%PDF-1.4 source bytes"
    fake_annotated_bytes = b"%PDF-1.4 annotated bytes"

    fake_highlighter = MagicMock(return_value=fake_annotated_bytes)

    with patch("app.components.faiss_results.st") as mock_st, patch.dict(
        "sys.modules",
        {
            "src.utils.pdf_highlighter": MagicMock(
                highlight_pdf_matches=fake_highlighter
            )
        },
    ):
        col1, col2 = MagicMock(), MagicMock()
        mock_st.columns.return_value = (col1, col2)

        inspect_diff_dialog(
            "query text sample",
            "matched text sample",
            "essay1.pdf",
            0.85,
            pdf_bytes=fake_pdf_bytes,
        )

        fake_highlighter.assert_called_once_with(
            fake_pdf_bytes, ["matched text sample"]
        )
        mock_st.download_button.assert_called_once()
        _, kwargs = mock_st.download_button.call_args
        assert kwargs["data"] == fake_annotated_bytes
        assert kwargs["file_name"] == "highlighted_essay1.pdf"
        assert kwargs["mime"] == "application/pdf"


def test_inspect_diff_dialog_skips_download_without_pdf_bytes():
    """Verify no download button is offered when no source PDF is available."""
    from unittest.mock import MagicMock, patch

    from app.components.faiss_results import inspect_diff_dialog

    with patch("app.components.faiss_results.st") as mock_st:
        col1, col2 = MagicMock(), MagicMock()
        mock_st.columns.return_value = (col1, col2)

        inspect_diff_dialog(
            "query text sample", "matched text sample", "test_doc.pdf", 0.85
        )

        mock_st.download_button.assert_not_called()


def test_render_faiss_results_ui_doc_hash_copy_box():
    """Verify doc_hash is rendered using st.code for built-in 1-click clipboard copy (#1724)."""
    from unittest.mock import patch

    from app.components.faiss_results import render_faiss_results_ui

    sample_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    record = {"doc_name": "doc_a.pdf", "chunk_index": 0, "doc_hash": sample_hash}
    results = [(record, 0.88)]

    with patch("app.components.faiss_results.st") as mock_st:
        render_faiss_results_ui(results, "query text")
        mock_st.code.assert_any_call(sample_hash, language="text")


def test_inspect_diff_dialog_renders_doc_hash():
    """Verify inspect_diff_dialog renders st.code(doc_hash) when doc_hash is passed (#1724)."""
    from unittest.mock import MagicMock, patch

    from app.components.faiss_results import inspect_diff_dialog

    sample_hash = "a" * 64
    with patch("app.components.faiss_results.st") as mock_st:
        col1, col2 = MagicMock(), MagicMock()
        mock_st.columns.return_value = (col1, col2)

        inspect_diff_dialog("query", "match", "doc.pdf", 0.90, doc_hash=sample_hash)
        mock_st.code.assert_any_call(sample_hash, language="text")
