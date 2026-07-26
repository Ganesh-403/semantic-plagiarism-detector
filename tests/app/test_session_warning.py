from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")


def test_session_timeout_warning_js_present():
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'id = \\'session-warning-toast\\'' in source
    assert 'warningTime = timeoutLimit - (2 * 60 * 1000)' in source
    assert 'parentDoc.addEventListener(\\'mousemove\\', resetTimer)' in source
