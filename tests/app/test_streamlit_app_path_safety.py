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

"""
tests/app/test_streamlit_app_path_safety.py
-------------------------------------------
Unit tests to verify app/streamlit_app.py does not use dangerous sys.path.insert(0) (Issue #2784).
"""

from pathlib import Path


def test_streamlit_app_no_dangerous_sys_path_insert():
    """Verify app/streamlit_app.py does not prepend to sys.path with sys.path.insert(0, ...)."""
    app_path = (
        Path(__file__).resolve().parent.parent.parent / "app" / "streamlit_app.py"
    )
    assert app_path.is_file(), f"File not found: {app_path}"

    content = app_path.read_text(encoding="utf-8")

    # Assert sys.path.insert(0 is not in the file
    assert "sys.path.insert(0" not in content, (
        "Found dangerous 'sys.path.insert(0, ...)' in app/streamlit_app.py. "
        "Use sys.path.append(...) or native module resolution to prevent standard library shadowing."
    )


def test_streamlit_app_syntax_compilation():
    """Verify app/streamlit_app.py compiles without syntax errors."""
    app_path = (
        Path(__file__).resolve().parent.parent.parent / "app" / "streamlit_app.py"
    )
    content = app_path.read_text(encoding="utf-8")
    compiled = compile(content, str(app_path), "exec")
    assert compiled is not None
