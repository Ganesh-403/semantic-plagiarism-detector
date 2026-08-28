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

from src.core.export_engine import LMSExportEngine

INCIDENTS = [
    {
        "doc_a": "alpha.pdf",
        "doc_b": "beta.pdf",
        "similarity": 0.93,
    }
]


def test_csv_generation_api_is_preserved():
    result = LMSExportEngine.generate_incident_csv(INCIDENTS)

    assert result is not None
    assert "alpha.pdf" in result
    assert "High" in result


def test_json_generation_api_is_preserved():
    result = LMSExportEngine.generate_incident_json(INCIDENTS)

    assert result is not None
    payload = json.loads(result)
    assert payload["metadata"]["total_incidents"] == 1


def test_txt_generation_api_is_preserved():
    result = LMSExportEngine.generate_incident_txt(INCIDENTS)

    assert result is not None
    assert "Similarity: 93.0%" in result


def test_empty_exports_still_return_none():
    assert LMSExportEngine.generate_incident_csv([]) is None
    assert LMSExportEngine.generate_incident_json([]) is None
    assert LMSExportEngine.generate_incident_txt([]) is None
