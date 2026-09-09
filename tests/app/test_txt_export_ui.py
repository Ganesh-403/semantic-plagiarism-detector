from unittest.mock import patch
from streamlit.testing.v1 import AppTest

from app.components.incident_export import render_incident_txt_export


def test_txt_download_contains_real_filtered_report():
    incidents = [{"doc_a": "essay A", "doc_b": "essay B", "similarity": .9}]
    with patch("app.components.incident_export.st.download_button") as download:
        render_incident_txt_export(incidents)
    values = download.call_args.kwargs
    assert "essay A" in values["data"] and "essay B" in values["data"]
    assert "90.0%" in values["data"]
    assert values["file_name"] == "plagiarism_incident_summary.txt"
    assert values["mime"] == "text/plain; charset=utf-8"
    assert not values["disabled"]


def test_txt_export_has_disabled_empty_state():
    at = AppTest.from_string("from app.components.incident_export import render_incident_txt_export; render_incident_txt_export([])").run()
    assert not at.exception
    assert at.get("download_button")[0].proto.disabled
