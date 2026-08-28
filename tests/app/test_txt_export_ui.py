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


def test_warnings_tab_generates_txt_export():
    source = APP_PATH.read_text(encoding="utf-8")

    assert (
        "txt_data = LMSExportEngine.generate_incident_txt(" "raw_incidents)"
    ) in source


def test_txt_download_button_has_expected_metadata():
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'label="📝 Export TXT"' in source
    assert 'file_name="plagiarism_incident_summary.txt"' in source
    assert 'mime="text/plain; charset=utf-8"' in source
    assert 'key="export_incidents_txt"' in source


def test_txt_export_has_disabled_empty_state():
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'key="export_incidents_txt_disabled"' in source
