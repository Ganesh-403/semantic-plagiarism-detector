"""Property-based tests for sanitize_filename using hypothesis."""

import re
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from src.utils.filename import MAX_FILENAME_LENGTH, sanitize_filename

_SAFE_PATTERN = re.compile(r"^[A-Za-z0-9._ -]+$")
_ILLEGAL_CHARS = {"<", ">", '"', "/", "\\"}


@given(st.text())
@settings(max_examples=300)
def test_output_never_contains_null_bytes(filename):
    result = sanitize_filename(filename)
    assert "\x00" not in result


@given(st.text())
@settings(max_examples=300)
def test_output_never_contains_directory_traversal(filename):
    result = sanitize_filename(filename)
    assert ".." not in result


@given(st.text())
@settings(max_examples=300)
def test_output_never_contains_illegal_os_chars(filename):
    result = sanitize_filename(filename)
    for char in _ILLEGAL_CHARS:
        assert char not in result, f"Illegal char {char!r} found in {result!r}"


@given(st.text())
@settings(max_examples=300)
def test_output_length_is_bounded(filename):
    result = sanitize_filename(filename)
    assert len(result) <= MAX_FILENAME_LENGTH


@given(st.text(), st.integers(min_value=8, max_value=500))
@settings(max_examples=200)
def test_output_length_respects_custom_max(filename, max_length):
    result = sanitize_filename(filename, max_length=max_length)
    assert len(result) <= max_length


@given(st.text())
@settings(max_examples=300)
def test_output_is_always_non_empty(filename):
    result = sanitize_filename(filename)
    assert len(result) > 0


@given(st.text())
@settings(max_examples=300)
def test_output_contains_only_safe_chars(filename):
    result = sanitize_filename(filename)
    assert _SAFE_PATTERN.match(result), f"Unsafe chars in {result!r}"
