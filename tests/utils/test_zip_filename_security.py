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

import io
import zipfile

from src.utils.zip_processor import process_zip_file


def make_zip(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return stream.getvalue()


def test_zip_entries_are_sanitized():
    result = process_zip_file(
        make_zip(
            [
                ("folder/<script>alert(1)</script>.pdf", b"pdf"),
                ("folder/report.pdf", b"one"),
                ("other/report.pdf", b"two"),
            ]
        )
    )

    assert list(result) == [
        "alert_1.pdf",
        "report.pdf",
        "report_1.pdf",
    ]


import pytest


def test_zip_path_traversal_entry_is_rejected():
    with pytest.raises(
        ValueError, match="Malicious path traversal detected in ZIP archive entry"
    ):
        process_zip_file(
            make_zip(
                [
                    ("../../evil.pdf", b"bad"),
                    ("safe.pdf", b"good"),
                ]
            )
        )
