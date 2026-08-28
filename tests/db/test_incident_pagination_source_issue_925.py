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

INCIDENTS_PATH = Path("src/db/incidents.py")


def test_get_all_incidents_has_required_defaults():
    source = INCIDENTS_PATH.read_text(encoding="utf-8")

    assert "limit: int = 50" in source
    assert "offset: int = 0" in source


def test_query_uses_parameterized_limit_and_offset():
    source = INCIDENTS_PATH.read_text(encoding="utf-8")

    assert "LIMIT ? OFFSET ?" in source
    assert "(safe_limit, safe_offset)" in source


def test_total_count_helper_exists():
    source = INCIDENTS_PATH.read_text(encoding="utf-8")

    assert "def get_total_incidents_count(" in source
    assert "SELECT COUNT(*)" in source
