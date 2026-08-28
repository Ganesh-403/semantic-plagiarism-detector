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
    download_position = source.index('label="⬇️ Download raw Database"')

    assert download_position > admin_position
