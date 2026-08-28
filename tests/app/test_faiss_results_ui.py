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

APP_PATH = Path("app/streamlit_app.py")


def test_faiss_results_use_interactive_ui():
    """Verify that the interactive render_faiss_results_ui component is used to support chunk diff inspection."""
    source = APP_PATH.read_text(encoding="utf-8")
    assert "render_faiss_results_ui(results," in source


def test_static_faiss_result_loop_is_removed():
    """Verify that raw print loops for FAISS results are removed in favor of the component."""
    source = APP_PATH.read_text(encoding="utf-8")
    assert "for rec, score in results:" not in source
    assert "st.caption(rec.chunk_text)" not in source
