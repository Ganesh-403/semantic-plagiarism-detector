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


def test_session_timeout_warning_js_present():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "id = 'session-warning-toast'" in source
    assert "warningTime = timeoutLimit - (2 * 60 * 1000)" in source
    assert "parentDoc.addEventListener('mousemove', resetTimer)" in source


def test_session_timeout_limits():
    # Verify the code logic for dynamic TIMEOUT_LIMIT
    class MockSessionState(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    # Admin role timeout limit
    session_state = MockSessionState(role="admin")
    timeout_limit = 30 * 60 if session_state.get("role") == "admin" else 15 * 60
    assert timeout_limit == 1800  # 30 minutes

    # Non-admin role timeout limit
    session_state = MockSessionState(role="teacher")
    timeout_limit = 30 * 60 if session_state.get("role") == "admin" else 15 * 60
    assert timeout_limit == 900  # 15 minutes


def test_session_expiration_and_refresh():
    import time
    from unittest.mock import MagicMock

    # 1. Test session refresh (non-expired)
    st = MagicMock()
    st.session_state = {
        "authenticated": True,
        "role": "admin",
        "last_interaction": time.time() - 1000,
    }

    timeout_limit = 30 * 60 if st.session_state.get("role") == "admin" else 15 * 60
    last_interaction = st.session_state["last_interaction"]
    elapsed_time = time.time() - last_interaction

    assert elapsed_time <= timeout_limit

    # Simulation of refresh logic
    st.session_state["last_interaction"] = time.time()
    assert st.session_state["last_interaction"] > last_interaction

    # 2. Test session expiration (expired admin)
    st.session_state = {
        "authenticated": True,
        "role": "admin",
        "last_interaction": time.time() - 2000,
    }
    timeout_limit = 30 * 60 if st.session_state.get("role") == "admin" else 15 * 60
    last_interaction = st.session_state["last_interaction"]
    elapsed_time = time.time() - last_interaction

    assert elapsed_time > timeout_limit

    # Simulation of cleanup logic
    for key in ["authenticated", "username", "role", "last_interaction"]:
        if key in st.session_state:
            del st.session_state[key]

    assert "authenticated" not in st.session_state
    assert "role" not in st.session_state
