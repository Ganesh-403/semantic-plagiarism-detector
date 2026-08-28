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

import json
from unittest.mock import patch

from src.cli import run_db_status


def test_text_status_output(capsys):
    with patch(
        "src.db.migrations.get_migration_status",
        return_value={
            "current_version": 2,
            "target_version": 4,
            "pending_migrations": [3, 4],
        },
    ):
        result = run_db_status("corpus.db", "corpus")

    output = capsys.readouterr().out
    assert result == 0
    assert "Current version: 2" in output
    assert "Target version: 4" in output
    assert "Pending migrations: 3, 4" in output


def test_json_status_output(capsys):
    expected = {
        "current_version": 4,
        "target_version": 4,
        "pending_migrations": [],
    }
    with patch(
        "src.db.migrations.get_migration_status",
        return_value=expected,
    ):
        result = run_db_status(
            "corpus.db",
            "corpus",
            output_format="json",
        )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_status_error_returns_nonzero(capsys):
    with patch(
        "src.db.migrations.get_migration_status",
        side_effect=FileNotFoundError("missing"),
    ):
        result = run_db_status("missing.db", "auth")

    assert result == 1
    assert "missing" in capsys.readouterr().err
