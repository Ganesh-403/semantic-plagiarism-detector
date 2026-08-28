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

TRANSLATOR_PATH = Path("src/i18n/translator.py")


def test_translation_loader_uses_streamlit_cache_data():
    source = TRANSLATOR_PATH.read_text(encoding="utf-8")

    assert "@st.cache_data(show_spinner=False)" in source
    assert "def _load_translation_dictionary(" in source


def test_load_translations_delegates_to_cached_loader():
    source = TRANSLATOR_PATH.read_text(encoding="utf-8")

    assert "_load_translation_dictionary(file_path)" in source


def test_cache_can_be_cleared_explicitly():
    source = TRANSLATOR_PATH.read_text(encoding="utf-8")

    assert "def clear_translation_cache()" in source
    assert "_load_translation_dictionary.clear()" in source
