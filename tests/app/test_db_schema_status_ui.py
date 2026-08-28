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
tests/app/test_db_schema_status_ui.py
-------------------------------------
Unit tests for Refresh Database Schema Status Button in System Settings (#1729).
"""

from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")


def test_db_schema_status_ui_elements():
    """Verify that Check Database Schema button and message logic exist in settings tab."""
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Check Database Schema" in source
    assert "get_user_version" in source
    assert "db_schema_status_msg" in source
    assert "Corpus Schema: v" in source
    assert "Auth Schema: v" in source
    assert "st.toast(" in source
