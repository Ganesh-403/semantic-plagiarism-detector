from unittest.mock import patch
import sqlite3
from tests.app.settings_helpers import settings_page


def test_admin_can_generate_valid_database_download(mock_db, tmp_path):
    with patch("streamlit.download_button") as download:
        at = settings_page().run()
        button = next(button for button in at.button if button.label == "⬇️ Generate Database Backup")
        button.click().run()
    assert not at.exception
    backups = [call.kwargs for call in download.call_args_list if call.kwargs.get("key") == "download_raw_corpus_database"]
    assert len(backups) == 1
    backup = backups[0]
    assert backup["mime"] == "application/vnd.sqlite3"
    assert backup["file_name"] == "corpus.db"
    path = tmp_path / "download.db"
    path.write_bytes(backup["data"])
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    finally:
        connection.close()


def test_non_admin_cannot_generate_database_backup(mock_db):
    at = settings_page("teacher").run()
    assert not at.exception
    assert not any("Database Backup" in button.label for button in at.button)
