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

"""Integration/UI tests for Storage Space Used widget in app/streamlit_app.py."""

from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")


def test_storage_widget_imports():
    """Verify calculate_storage_usage is imported in app/streamlit_app.py."""
    source = APP_PATH.read_text(encoding="utf-8")
    assert "from src.utils.storage_metrics import calculate_storage_usage" in source


def test_storage_widget_inside_admin_sidebar_block():
    """Verify Storage Space Used widget is placed inside the admin user block."""
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'if user_role == "admin":' in source
    assert "### 💾 Storage Space Used" in source
    assert 'label="Total Storage Used"' in source
    assert "calculate_storage_usage()" in source

    admin_pos = source.index('if user_role == "admin":')
    storage_widget_pos = source.index("### 💾 Storage Space Used")
    assert storage_widget_pos > admin_pos
