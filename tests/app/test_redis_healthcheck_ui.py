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
tests/app/test_redis_healthcheck_ui.py
--------------------------------------
Unit tests verifying Redis cache health indicator presence in UI sidebar.
"""

from pathlib import Path

CORPUS_VIEW_PATH = Path("app/views/corpus_view.py")
STREAMLIT_APP_PATH = Path("app/streamlit_app.py")


def test_sidebar_has_cache_status_indicator():
    source = CORPUS_VIEW_PATH.read_text(encoding="utf-8")
    assert "🟢 Cache: Redis" in source
    assert "🟡 Cache: In-Memory" in source
    assert "ping()" in source


def test_streamlit_app_system_health_has_cache_status():
    source = STREAMLIT_APP_PATH.read_text(encoding="utf-8")
    assert "• **Cache Backend:** 🟢 Redis" in source
    assert "• **Cache Backend:** 🟡 In-Memory" in source
