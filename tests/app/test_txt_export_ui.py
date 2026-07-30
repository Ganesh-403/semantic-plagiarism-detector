from pathlib import Path


APP_PATH = Path("app/streamlit_app.py")


def test_warnings_tab_generates_txt_export():
    source = APP_PATH.read_text(encoding="utf-8")

    assert (
        "txt_data = LMSExportEngine.generate_incident_txt("
        "raw_incidents)"
    ) in source


def test_txt_download_button_has_expected_metadata():
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'label="📝 Export TXT"' in source
    assert (
        'file_name="plagiarism_incident_summary.txt"'
        in source
    )
    assert 'mime="text/plain; charset=utf-8"' in source
    assert 'key="export_incidents_txt"' in source


def test_txt_export_has_disabled_empty_state():
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'key="export_incidents_txt_disabled"' in source
