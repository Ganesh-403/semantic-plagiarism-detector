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

from unittest.mock import MagicMock, call, patch

import streamlit as st

# Keep imports deterministic when Streamlit is not running an app.
st.dialog = lambda *args, **kwargs: lambda function: function

from app.components.faiss_results import render_faiss_results_ui


class MockRecord:
    def __init__(
        self,
        doc_name: str,
        chunk_index: int,
        chunk_text: str,
        chunk_id: str = "chunk-7",
    ):
        self.doc_name = doc_name
        self.chunk_index = chunk_index
        self.chunk_text = chunk_text
        self.chunk_id = chunk_id


def _columns():
    return MagicMock(), MagicMock()


def test_matched_chunk_is_rendered_in_copyable_code_block():
    matched_chunk = "This sentence is the exact FAISS match that should " "be copyable."
    results = [
        (
            MockRecord(
                "essay.pdf",
                6,
                matched_chunk,
            ),
            0.91,
        )
    ]

    with patch("app.components.faiss_results.st") as mock_st:
        mock_st.columns.return_value = _columns()
        mock_st.button.return_value = False

        render_faiss_results_ui(
            results,
            "query text",
        )

    mock_st.caption.assert_any_call("📋 Matched Text")
    mock_st.code.assert_has_calls(
        [
            call("0.9100", language="text"),
            call("chunk-7", language="text"),
            call(matched_chunk, language="text"),
        ],
        any_order=False,
    )


def test_matched_chunk_copy_block_keeps_full_untruncated_text():
    matched_chunk = ("Long matched sentence. " * 30).strip()
    assert len(matched_chunk) > 300

    results = [
        (
            MockRecord(
                "long-essay.pdf",
                0,
                matched_chunk,
                "chunk-long",
            ),
            0.87,
        )
    ]

    with patch("app.components.faiss_results.st") as mock_st:
        mock_st.columns.return_value = _columns()
        mock_st.button.return_value = False

        render_faiss_results_ui(
            results,
            "query text",
        )

    mock_st.code.assert_any_call(
        matched_chunk,
        language="text",
    )

    rendered_values = [args[0] for args, _kwargs in mock_st.code.call_args_list if args]
    assert matched_chunk in rendered_values
    assert matched_chunk[:300] + "..." not in rendered_values


def test_one_matched_text_code_block_is_added_per_result():
    results = [
        (
            MockRecord(
                "a.pdf",
                0,
                "First matched chunk",
                "chunk-a",
            ),
            0.92,
        ),
        (
            MockRecord(
                "b.pdf",
                1,
                "Second matched chunk",
                "chunk-b",
            ),
            0.86,
        ),
    ]

    with patch("app.components.faiss_results.st") as mock_st:
        mock_st.columns.side_effect = [
            _columns(),
            _columns(),
        ]
        mock_st.button.return_value = False

        render_faiss_results_ui(
            results,
            "query text",
        )

    mock_st.code.assert_any_call(
        "First matched chunk",
        language="text",
    )
    mock_st.code.assert_any_call(
        "Second matched chunk",
        language="text",
    )
    assert mock_st.caption.call_args_list.count(call("📋 Matched Text")) == 2
