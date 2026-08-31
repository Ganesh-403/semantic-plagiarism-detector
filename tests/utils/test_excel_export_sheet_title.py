"""
tests/utils/test_excel_export_sheet_title.py
--------------------------------------------
Tests for worksheet-title sanitization in the Excel export helpers
(issue #3782).

Two things are pinned here.

First, that ``src.utils.excel_export`` imports at all. A stray branch name was
left in the middle of the import block by a bad merge, which made the whole
module raise ``NameError`` on import and took every export path down with it.
The import at the top of this file is itself the regression test; the explicit
test below states the intent so the reason is not lost.

Second, that a caller-supplied sheet title is actually sanitized on *both*
workbook paths. ``sanitize_sheet_title`` previously only ever saw the hardcoded
string ``"Similarity Matrix"``, and the ``write_only=True`` path bypassed it
entirely, so the helper defended nothing.
"""

import importlib

import pandas as pd
import pytest

from src.utils.excel_export import (
    DEFAULT_SHEET_TITLE,
    DEFAULT_WORKSHEET_TITLE,
    MAX_SHEET_TITLE_LENGTH,
    build_similarity_workbook,
    export_similarity_matrix_to_excel,
    sanitize_sheet_title,
)

# Characters Excel refuses in a worksheet title.
INVALID_TITLE_CHARS = ["[", "]", "*", "?", ":", "/", "\\", "."]


@pytest.fixture
def matrix():
    labels = ["DocA.txt", "DocB.txt", "DocC.txt"]
    return pd.DataFrame(
        [[1.0, 0.85, 0.12], [0.85, 1.0, 0.45], [0.12, 0.45, 1.0]],
        index=labels,
        columns=labels,
    )


# ── module import ──────────────────────────────────────────────────────────────


def test_module_imports_cleanly():
    """The stray ``fix/excel-sheet-title-sanitization`` line must stay gone.

    It parsed as ``fix / excel - sheet - title - sanitization`` and raised
    ``NameError: name 'fix' is not defined`` at import time.
    """
    module = importlib.import_module("src.utils.excel_export")
    assert module is not None
    assert callable(module.build_similarity_workbook)


def test_comment_import_is_not_duplicated():
    """The bad merge left two ``from openpyxl.comments import Comment`` lines."""
    import inspect

    import src.utils.excel_export as module

    source = inspect.getsource(module)
    assert source.count("from openpyxl.comments import Comment") == 1


# ── sanitize_sheet_title ───────────────────────────────────────────────────────


@pytest.mark.parametrize("char", INVALID_TITLE_CHARS)
def test_each_invalid_character_is_stripped(char):
    result = sanitize_sheet_title(f"Sheet{char}Name")
    assert char not in result
    assert result == "SheetName"


def test_all_invalid_characters_stripped_together():
    assert sanitize_sheet_title("a/b:c*d?e[f]g.h\\i") == "abcdefghi"


def test_long_title_is_clamped_to_excel_limit():
    result = sanitize_sheet_title("x" * 100)
    assert len(result) == MAX_SHEET_TITLE_LENGTH


def test_title_at_the_limit_is_untouched():
    exact = "y" * MAX_SHEET_TITLE_LENGTH
    assert sanitize_sheet_title(exact) == exact


def test_clamping_happens_after_stripping():
    """Invalid characters must not consume part of the length budget.

    A title of 31 legal characters padded with slashes should survive whole.
    """
    title = "/" * 10 + "z" * MAX_SHEET_TITLE_LENGTH + "/" * 10
    assert sanitize_sheet_title(title) == "z" * MAX_SHEET_TITLE_LENGTH


def test_surrounding_whitespace_is_trimmed():
    assert sanitize_sheet_title("   Report   ") == "Report"


@pytest.mark.parametrize("empty", ["", "   ", "...", "///", "[]*?:"])
def test_titles_with_nothing_left_fall_back_to_default(empty):
    """openpyxl also rejects an empty title, so a fallback is required."""
    assert sanitize_sheet_title(empty) == DEFAULT_SHEET_TITLE


@pytest.mark.parametrize("value", [12345, 3.14, None, True])
def test_non_string_titles_are_coerced(value):
    """A non-string label (e.g. an integer assignment ID) must not raise."""
    result = sanitize_sheet_title(value)
    assert isinstance(result, str)
    assert result


def test_sanitization_is_idempotent():
    once = sanitize_sheet_title("Course: Semester/2024 [final].xlsx")
    assert sanitize_sheet_title(once) == once


def test_benign_title_is_left_alone():
    assert sanitize_sheet_title("Similarity Matrix") == "Similarity Matrix"


# ── build_similarity_workbook ──────────────────────────────────────────────────


def test_default_sheet_title_is_used(matrix):
    ws = build_similarity_workbook(matrix).active
    assert ws.title == DEFAULT_WORKSHEET_TITLE


def test_custom_sheet_title_is_honoured(matrix):
    ws = build_similarity_workbook(matrix, sheet_title="Midterm Essays").active
    assert ws.title == "Midterm Essays"


def test_custom_sheet_title_is_sanitized(matrix):
    """The in-memory (DOM) path."""
    ws = build_similarity_workbook(
        matrix, sheet_title="CS101: Essay/Draft [v2].xlsx"
    ).active
    assert ws.title == "CS101 EssayDraft v2xlsx"


def test_write_only_path_also_sanitizes(matrix):
    """The write-only path used to bypass sanitization entirely."""
    wb = build_similarity_workbook(
        matrix, sheet_title="CS101: Essay/Draft [v2].xlsx", write_only=True
    )
    assert wb.worksheets[0].title == "CS101 EssayDraft v2xlsx"


def test_write_only_path_clamps_length(matrix):
    wb = build_similarity_workbook(matrix, sheet_title="w" * 80, write_only=True)
    assert len(wb.worksheets[0].title) == MAX_SHEET_TITLE_LENGTH


def test_both_paths_agree_on_the_same_title(matrix):
    """The two branches must not drift apart again."""
    messy = "Term 2: Cohort/B [late].csv"

    dom = build_similarity_workbook(matrix, sheet_title=messy).active.title
    stream = build_similarity_workbook(
        matrix, sheet_title=messy, write_only=True
    ).worksheets[0].title

    assert dom == stream


def test_workbook_with_hostile_title_actually_saves(matrix):
    """The point of sanitizing: openpyxl must not raise while writing."""
    payload = export_similarity_matrix_to_excel(
        matrix, sheet_title="Ünit 4: Report/Final [rev].xlsx"
    )
    assert payload.startswith(b"PK")  # xlsx is a zip container


def test_workbook_with_overlong_title_actually_saves(matrix):
    payload = export_similarity_matrix_to_excel(matrix, sheet_title="q" * 200)
    assert payload.startswith(b"PK")


def test_matrix_content_is_unaffected_by_the_title(matrix):
    """Sanitizing the title must not disturb the exported data."""
    ws = build_similarity_workbook(matrix, sheet_title="Anything: Goes/Here").active

    assert ws.cell(row=1, column=1).value == "Document"
    assert ws.cell(row=2, column=1).value == "DocA.txt"
    assert ws.cell(row=2, column=2).value == pytest.approx(1.0)
