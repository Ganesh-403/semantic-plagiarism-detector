from pathlib import Path

APP_PATH = Path("app/streamlit_app.py")


def test_admin_settings_contains_database_download():
    source = APP_PATH.read_text(encoding="utf-8")

    assert "create_corpus_database_snapshot()" in source
    assert 'label="⬇️ Download raw Database"' in source
    assert 'file_name="corpus.db"' in source
    assert 'mime="application/vnd.sqlite3"' in source


def test_download_uses_unique_widget_key():
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'key="download_raw_corpus_database"' in source


def test_download_is_inside_admin_settings_block():
    source = APP_PATH.read_text(encoding="utf-8")

    admin_position = source.index('if user_role == "admin":')
    download_position = source.index(
        'label="⬇️ Download raw Database"'
    )

    assert download_position > admin_position
