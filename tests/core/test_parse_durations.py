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

"""
tests/core/test_parse_durations.py
----------------------------------
Unit tests for the parse duration registry (Issue #1728).
"""

from src.core.parse_durations import (
    clear_parse_durations,
    format_duration,
    get_all_parse_durations,
    get_parse_duration,
    record_parse_duration,
)


def test_record_and_get():
    clear_parse_durations()
    record_parse_duration("test.pdf", 0.42)
    assert get_parse_duration("test.pdf") == 0.42


def test_get_missing_returns_none():
    clear_parse_durations()
    assert get_parse_duration("nonexistent.pdf") is None


def test_get_all():
    clear_parse_durations()
    record_parse_duration("a.pdf", 0.1)
    record_parse_duration("b.pdf", 0.2)
    all_durations = get_all_parse_durations()
    assert all_durations == {"a.pdf": 0.1, "b.pdf": 0.2}


def test_clear():
    clear_parse_durations()
    record_parse_duration("test.pdf", 0.42)
    clear_parse_durations()
    assert get_all_parse_durations() == {}


def test_format_duration():
    assert format_duration(0.423456) == "0.42s"
    assert format_duration(1.0) == "1.00s"
    assert format_duration(0.0) == "0.00s"


def test_format_duration_none():
    assert format_duration(None) == ""
