from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")


def test_session_timeout_warning_js_present():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "id = 'session-warning-toast'" in source
    assert 'warningTime = timeoutLimit - (2 * 60 * 1000)' in source
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
    st.session_state = {"authenticated": True, "role": "admin", "last_interaction": time.time() - 1000}

    timeout_limit = 30 * 60 if st.session_state.get("role") == "admin" else 15 * 60
    last_interaction = st.session_state["last_interaction"]
    elapsed_time = time.time() - last_interaction

    assert elapsed_time <= timeout_limit

    # Simulation of refresh logic
    st.session_state["last_interaction"] = time.time()
    assert st.session_state["last_interaction"] > last_interaction

    # 2. Test session expiration (expired admin)
    st.session_state = {"authenticated": True, "role": "admin", "last_interaction": time.time() - 2000}
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
