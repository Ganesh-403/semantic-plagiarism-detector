"""Filename regressions exercise the upload sanitizer and collision handling."""

from pathlib import PurePosixPath
import pytest
from src.utils.filename import sanitize_filename, sanitize_filename_mapping


@pytest.mark.parametrize(("name", "expected"), [
    ("report.pdf", "report.pdf"), ("my report.pdf", "my_report.pdf"),
    ("../report.pdf", "report.pdf"), ("..\\report.pdf", "report.pdf"),
    ("C:\\uploads\\report.pdf", "report.pdf"), ("café.txt", "café.txt"),
    ("CON.txt", "_CON.txt"), ("<b>report</b>.pdf", "report.pdf"),
])
def test_filename_sanitization(name, expected):
    assert sanitize_filename(name) == expected


@pytest.mark.parametrize("name", [None, "", "...", "foo'bar.txt", "a\x00b.pdf", "<script>alert(1)</script>.txt", "猫" * 200 + ".txt"])
def test_output_is_safe_and_bounded(name):
    result = sanitize_filename(name)
    assert result and PurePosixPath(result).name == result
    assert len(result.encode("utf-8")) <= 255
    assert all(char not in result for char in "<>:'\\/\x00")


def test_mapping_preserves_colliding_uploads():
    original = {"../report.txt": b"first", "report.txt": b"second", "dir/report.txt": b"third"}
    result = sanitize_filename_mapping(original)
    assert len(result) == 3
    assert list(result.values()) == [b"first", b"second", b"third"]
    assert len(set(result)) == 3
    assert all(name.endswith(".txt") for name in result)
