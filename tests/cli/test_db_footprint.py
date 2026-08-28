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

import pytest

from src.cli import main


def test_db_footprint_text_output(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["cli.py", "db", "footprint"])

    mock_res = {
        "embedding_bytes": 1536,
        "database_bytes": 10240,
        "embedding_percentage": 15.0,
        "chunk_count": 2,
    }

    with patch(
        "src.db.corpus_db.get_embedding_storage_footprint", return_value=mock_res
    ):
        with pytest.raises(SystemExit) as e:
            main()

        assert e.value.code == 0

    out, err = capsys.readouterr()
    assert "Total Database Size: 10,240 bytes" in out
    assert "Total Embedding Size: 1,536 bytes" in out
    assert "Embedding Storage Percentage: 15.00%" in out
    assert "Total Chunks: 2" in out


def test_db_footprint_json_output(capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["cli.py", "db", "footprint", "--output-format", "json"]
    )

    mock_res = {
        "embedding_bytes": 4096,
        "database_bytes": 8192,
        "embedding_percentage": 50.0,
        "chunk_count": 5,
    }

    with patch(
        "src.db.corpus_db.get_embedding_storage_footprint", return_value=mock_res
    ):
        with pytest.raises(SystemExit) as e:
            main()

        assert e.value.code == 0

    out, err = capsys.readouterr()
    parsed = json.loads(out)
    assert parsed["embedding_bytes"] == 4096
    assert parsed["embedding_percentage"] == 50.0


def test_db_footprint_error_handling(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["cli.py", "db", "footprint"])

    with patch(
        "src.db.corpus_db.get_embedding_storage_footprint",
        side_effect=Exception("DB Error"),
    ):
        with pytest.raises(SystemExit) as e:
            main()

        assert e.value.code == 1

    out, err = capsys.readouterr()
    assert "Error calculating storage footprint: DB Error" in err
