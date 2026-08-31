"""Regression tests for #4080.

``src/utils/warning_list.py`` line 8 held two statements run together with no
newline between them::

    import refrom typing import Any, Callable, Iterable, Mapping, Sequence

which is ``import re`` and ``from typing import ...`` collapsed onto one line.
The module raised ``SyntaxError`` on import, so the Streamlit warnings panel,
the drilldown view and the copy-snippet button were all dead, and five test
modules failed at *collection* rather than at assertion — a failure mode that
reads like a broken environment rather than broken code.

The damage came from the same edit that added the binary-character warning and
the CSV/JSON export helpers (``3736``/``3737``/``3738``): several newlines were
eaten, and line 8 was the only one that happened to be fatal.

These tests do three things:

1. Assert the module imports, and that ``re`` *and* the ``typing`` names are
   genuinely used — so nobody "fixes" a future recurrence by deleting half of
   line 8.
2. Guard every module under ``src/utils/`` against the same class of defect.
3. Exercise the module's public surface, which until now had no test coverage
   that could actually run.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "src" / "utils" / "warning_list.py"

UTILS_FILES = sorted(
    p
    for p in (REPO_ROOT / "src" / "utils").rglob("*.py")
    if "__pycache__" not in p.parts
)


# ── The parse failure itself ─────────────────────────────────────────────────


def test_module_parses() -> None:
    """The exact failure: ``ast.parse`` raised SyntaxError at line 8."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    ast.parse(source, filename=str(MODULE_PATH))


def test_module_imports() -> None:
    module = importlib.import_module("src.utils.warning_list")
    assert module is not None


@pytest.mark.parametrize(
    "path", UTILS_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_every_utils_module_parses(path: pathlib.Path) -> None:
    """No module under src/utils/ may fail to parse.

    This one did, and because the importers are Streamlit views rather than
    tests, nothing surfaced it until the test suite was run directly.
    """
    try:
        ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - only on regression
        pytest.fail(
            f"{path.relative_to(REPO_ROOT)}:{exc.lineno}: {exc.msg}\n"
            f"    {(exc.text or '').rstrip()}"
        )


def test_the_original_broken_line_really_is_a_syntax_error() -> None:
    """Pin the defect so the test is not quietly satisfied by anything."""
    with pytest.raises(SyntaxError):
        ast.parse("import refrom typing import Any\n")


def test_both_halves_of_line_eight_are_still_needed() -> None:
    """Neither import may be dropped as a shortcut fix.

    ``re`` and the ``typing`` names are both genuinely referenced, so a
    "resolution" that deletes one half would compile and then fail at runtime
    or under ``from __future__ import annotations`` evaluation.
    """
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)

    assert "re" in imported, "import re went missing"
    for name in ("Any", "Callable", "Iterable", "Mapping", "Sequence"):
        assert name in imported, f"{name} went missing from the typing import"

    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {
        n.value.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
    }
    # Annotations are strings under `from __future__ import annotations`, so
    # fall back to a source scan for the typing names.
    assert "re" in used, "import re is unused — it should not be"
    for name in ("Any", "Callable", "Iterable", "Mapping", "Sequence"):
        assert name in source, f"{name} is unused — it should not be"


def test_no_other_collapsed_definition_lines() -> None:
    """The same edit also collapsed ``def _normalise_warning(\\n    warning``.

    A ``def name(    arg`` shape — an open paren followed by run-in whitespace
    and an argument on the same line — is what an eaten newline looks like when
    it happens to stay syntactically valid.
    """
    import re as _re

    source = MODULE_PATH.read_text(encoding="utf-8")
    # ``[ \t]`` rather than ``\s`` — ``\s`` matches the newline, so the
    # pattern would happily span a correctly wrapped signature.
    collapsed = _re.findall(r"^def \w+\([ \t]{2,}\S.*$", source, _re.MULTILINE)
    assert collapsed == [], f"collapsed signatures left in the file: {collapsed}"


def test_the_collapsed_signature_detector_is_not_vacuous() -> None:
    """The pattern must catch the shape it exists for, and only that shape."""
    import re as _re

    pattern = _re.compile(r"^def \w+\([ \t]{2,}\S.*$", _re.MULTILINE)

    broken = "def _normalise_warning(    warning: Mapping[str, Any],\n) -> dict:\n"
    assert pattern.findall(broken)

    correct = "def _normalise_warning(\n    warning: Mapping[str, Any],\n) -> dict:\n"
    assert pattern.findall(correct) == []

    single_line = "def f(a, b):\n    return a\n"
    assert pattern.findall(single_line) == []


# ── The module's public surface ──────────────────────────────────────────────


@pytest.fixture()
def wl():
    return importlib.import_module("src.utils.warning_list")


def test_normalise_warning_fills_in_derived_fields(wl) -> None:
    out = wl._normalise_warning({"doc_a": "  a.pdf ", "doc_b": "b.pdf", "similarity": 0.91})
    assert out["doc_a"] == "a.pdf"
    assert out["similarity"] == pytest.approx(0.91)
    assert out["severity_rank"] == wl.severity_rank(out["severity"])


def test_normalise_warning_survives_a_non_numeric_similarity(wl) -> None:
    """A bad score must degrade to 0.0, not take the warnings panel down."""
    assert wl._normalise_warning({"similarity": "not a number"})["similarity"] == 0.0
    assert wl._normalise_warning({"similarity": None})["similarity"] == 0.0


def test_normalise_warning_flags_binary_looking_documents(wl) -> None:
    """The control-character warning added by the same edit that broke line 8."""
    out = wl._normalise_warning(
        {
            "doc_a": "a.bin",
            "control_char_ratio": wl.CONTROL_CHARACTER_RATIO_THRESHOLD + 0.5,
        }
    )
    assert wl.WARNING_BINARY_CHARACTERS in out["warnings"]


def test_normalise_warning_does_not_flag_clean_documents(wl) -> None:
    out = wl._normalise_warning({"doc_a": "a.txt", "control_char_ratio": 0.0})
    assert wl.WARNING_BINARY_CHARACTERS not in out.get("warnings", [])


def test_normalise_warning_ignores_an_unparseable_ratio(wl) -> None:
    out = wl._normalise_warning({"doc_a": "a.txt", "control_char_ratio": "n/a"})
    assert wl.WARNING_BINARY_CHARACTERS not in out.get("warnings", [])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ""),
        ("", ""),
        (12345, "12345"),
        (98.6, "98.6"),
        ("plain", "plain"),
    ],
)
def test_truncate_search_query_coerces(wl, raw, expected) -> None:
    assert wl._truncate_search_query(raw) == expected


def test_truncate_search_query_caps_length(wl) -> None:
    long = "x" * (wl.MAX_SEARCH_QUERY_LENGTH * 3)
    assert len(wl._truncate_search_query(long)) == wl.MAX_SEARCH_QUERY_LENGTH


def test_sort_warnings_orders_by_similarity_descending_by_default(wl) -> None:
    rows = [
        {"doc_a": "a", "doc_b": "x", "similarity": 0.2},
        {"doc_a": "b", "doc_b": "y", "similarity": 0.9},
        {"doc_a": "c", "doc_b": "z", "similarity": 0.5},
    ]
    assert [r["similarity"] for r in wl.sort_warnings(rows)] == [0.9, 0.5, 0.2]


def test_sort_warnings_falls_back_on_an_unknown_field(wl) -> None:
    """A sort key comes from a translated UI label; it must not reach any key.

    An unrecognised field falls back to the default rather than sorting by
    whatever happens to be in the dict under that name.
    """
    rows = [
        {"doc_a": "a", "doc_b": "x", "similarity": 0.2, "secret": 9},
        {"doc_a": "b", "doc_b": "y", "similarity": 0.9, "secret": 1},
    ]
    assert [r["similarity"] for r in wl.sort_warnings(rows, primary_field="secret")] == [
        0.9,
        0.2,
    ]


def test_sort_warnings_does_not_mutate_its_input(wl) -> None:
    rows = [{"doc_a": "a", "similarity": 0.2}, {"doc_a": "b", "similarity": 0.9}]
    before = [dict(r) for r in rows]
    wl.sort_warnings(rows)
    assert rows == before


def test_sort_warnings_handles_an_empty_input(wl) -> None:
    assert wl.sort_warnings([]) == []


def test_paginate_warnings_clamps_an_out_of_range_page(wl) -> None:
    rows = [{"doc_a": str(i), "similarity": 0.1} for i in range(12)]
    page = wl.paginate_warnings(rows, page=99, page_size=5)
    assert page.page == page.total_pages
    assert page.was_clamped is True


def test_paginate_warnings_returns_the_requested_slice(wl) -> None:
    rows = [{"doc_a": str(i), "similarity": 0.1} for i in range(12)]
    page = wl.paginate_warnings(rows, page=2, page_size=5)
    assert [r["doc_a"] for r in page.items] == ["5", "6", "7", "8", "9"]


def test_export_warnings_to_csv_has_the_documented_columns(wl) -> None:
    csv_text = wl.export_warnings_to_csv(
        [{"doc_a": "a.pdf", "warning_type": "SHORT", "severity": "Low", "message": "m"}]
    )
    header = csv_text.splitlines()[0]
    assert header == "Filename,Warning Code,Severity,Message"
    assert "a.pdf" in csv_text


def test_export_warnings_to_json_round_trips(wl) -> None:
    import json

    rows = json.loads(
        wl.export_warnings_to_json(
            [{"doc_a": "a.pdf", "warning_type": "SHORT", "severity": "Low", "message": "m"}]
        )
    )
    assert rows == [
        {
            "Filename": "a.pdf",
            "Warning Code": "SHORT",
            "Severity": "Low",
            "Message": "m",
        }
    ]


def test_export_accepts_dataclass_instances(wl) -> None:
    warning = wl.DocumentWarning(doc_a="a.pdf", warning_type="SHORT", severity="Low")
    assert "a.pdf" in wl.export_warnings_to_csv([warning])


def test_warning_list_collapses_repeats_of_the_same_warning(wl) -> None:
    """Duplicate (document, type) pairs bump a counter instead of stacking rows."""
    lst = wl.WarningList()
    lst.add_warning(wl.DocumentWarning(doc_a="a.pdf", warning_type="SHORT"))
    lst.add_warning(wl.DocumentWarning(doc_a="a.pdf", warning_type="SHORT"))
    lst.add_warning(wl.DocumentWarning(doc_a="b.pdf", warning_type="SHORT"))

    assert len(lst.warnings) == 2
    assert lst.warnings[0].occurrence_count == 2
    assert lst.warnings[1].occurrence_count == 1


def test_warning_list_ignores_an_unsupported_item(wl) -> None:
    lst = wl.WarningList()
    lst.add_warning(object())  # type: ignore[arg-type]
    assert lst.warnings == []


def test_warning_list_filters_by_severity_case_insensitively(wl) -> None:
    lst = wl.WarningList()
    lst.add_warning(wl.DocumentWarning(doc_a="a", warning_type="A", severity="High"))
    lst.add_warning(wl.DocumentWarning(doc_a="b", warning_type="B", severity="Low"))
    assert [w.doc_a for w in lst.filter_by_severity("HIGH")] == ["a"]


# ── The copy button's escaping, which depends on ``re`` ──────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("copy_ca_3", "copy_ca_3"),
        ('"><script>alert(1)</script><div id="', "scriptalert1scriptdivid"),
        ("<<<>>>", "copy-btn"),
        ("", "copy-btn"),
        (None, "copy-btn"),
        (42, "42"),
    ],
)
def test_sanitize_element_id(wl, raw, expected) -> None:
    """Exercises ``_SAFE_ELEMENT_ID_RE`` — the one use of ``re`` in the module."""
    assert wl.sanitize_element_id(raw) == expected


def test_sanitize_element_id_output_is_always_js_and_html_safe(wl) -> None:
    for hostile in ('a"b', "a'b", "a`b", "a<b", "a>b", "a&b", "a\nb", "a\\b"):
        out = wl.sanitize_element_id(hostile)
        assert all(c.isalnum() or c in "-_" for c in out), out


def test_sanitize_element_id_is_idempotent(wl) -> None:
    once = wl.sanitize_element_id('"><script>')
    assert wl.sanitize_element_id(once) == once


def test_escape_js_string_neutralises_a_script_close(wl) -> None:
    out = wl.escape_js_string("</script><script>alert(1)</script>")
    assert "</script>" not in out
    assert "\\u003C" in out


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\\b", "a\\\\b"),
        ('a"b', 'a\\"b'),
        ("a'b", "a\\'b"),
        ("a`b", "a\\`b"),
        ("a$b", "a\\$b"),
        ("a\nb", "a\\nb"),
        ("a\rb", "a\\rb"),
        ("a b", "a\\u2028b"),
        ("a b", "a\\u2029b"),
        (12, "12"),
    ],
)
def test_escape_js_string(wl, raw, expected) -> None:
    assert wl.escape_js_string(raw) == expected
