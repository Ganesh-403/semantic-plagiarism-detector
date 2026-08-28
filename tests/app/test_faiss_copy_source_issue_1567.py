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

from pathlib import Path

SOURCE = Path("app/components/faiss_results.py")
TESTS = Path("tests/app/test_faiss_copy_button_issue_1567.py")


def test_matched_text_uses_streamlit_code_block():
    source = SOURCE.read_text(encoding="utf-8")

    assert 'st.caption("📋 Matched Text")' in source
    assert 'st.code(chunk_text, language="text")' in source


def test_old_truncated_caption_is_removed():
    source = SOURCE.read_text(encoding="utf-8")

    assert "st.caption(chunk_text[:300]" not in source


def test_copy_regression_covers_full_long_chunk():
    source = TESTS.read_text(encoding="utf-8")

    assert "test_matched_chunk_copy_block_keeps_full_untruncated_text" in source
    assert "mock_st.code.assert_any_call(" in source
