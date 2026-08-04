from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")


def test_streamlit_page_title_uses_configured_value():
    source = APP_PATH.read_text(encoding="utf-8")

    assert "APP_TITLE = get_app_title()" in source
    assert "page_title=APP_TITLE" in source


def test_visible_branding_uses_app_title():
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'st.title(f"🔍 {APP_TITLE}")' in source
    assert 'st.caption(f"🎓 {APP_TITLE} · Streamlit")' in source
