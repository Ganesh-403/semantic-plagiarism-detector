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

AUTH_PATH = Path("src/db/auth.py")


def test_failed_event_is_logged():
    source = AUTH_PATH.read_text(encoding="utf-8")

    assert '"password_change_failed"' in source
    assert "def _log_password_change_failure(" in source


def test_required_reasons_exist():
    source = AUTH_PATH.read_text(encoding="utf-8")

    assert '"incorrect_old_password"' in source
    assert '"complexity_failed"' in source


def test_update_password_accepts_old_password():
    source = AUTH_PATH.read_text(encoding="utf-8")

    assert "old_password: str | None = None" in source
    assert "_verify_stored_password(" in source
