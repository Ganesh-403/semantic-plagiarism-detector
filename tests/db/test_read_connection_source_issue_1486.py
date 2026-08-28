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

SOURCE = Path("src/db/common.py")
TESTS = Path("tests/db/test_common.py")


def test_required_helper_and_return_type_exist():
    source = SOURCE.read_text(encoding="utf-8")

    assert "def get_read_connection(" in source
    assert "db_path: Path" in source
    assert ") -> sqlite3.Connection:" in source


def test_connection_uses_read_only_sqlite_uri():
    source = SOURCE.read_text(encoding="utf-8")

    assert "?mode=ro" in source
    assert "uri=True" in source
    assert "sqlite3.connect(" in source


def test_write_rejection_unit_test_exists():
    source = TESTS.read_text(encoding="utf-8")

    assert "test_get_read_connection_rejects_write_attempts" in source
    assert "sqlite3.OperationalError" in source
