import json
from unittest.mock import patch
from tests.app.settings_helpers import settings_page


def test_configuration_download_reflects_current_settings(mock_db):
    with patch("streamlit.download_button") as download:
        at = settings_page().run()
        at.slider(key="threshold_slider").set_value(.75).run()
    assert not at.exception
    backups = [call.kwargs for call in download.call_args_list if call.kwargs.get("key") == "backup_config_button"]
    assert backups[-1]["mime"] == "application/json"
    assert backups[-1]["file_name"] == "plagiarism_config_backup.json"
    assert json.loads(backups[-1]["data"])["threshold"] == .75


def test_non_admin_cannot_export_system_configuration(mock_db):
    with patch("streamlit.download_button") as download:
        at = settings_page("teacher").run()
    assert not at.exception
    download.assert_not_called()
