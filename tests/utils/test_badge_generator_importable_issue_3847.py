"""
tests/utils/test_badge_generator_importable_issue_3847.py
---------------------------------------------------------
Regression tests for Issue #3847.

``src/utils/badge_generator.py`` carried two consecutive ``def
validate_hex_color`` header lines with no body between them, so the module
raised ``IndentationError`` at import time and every symbol in it — the
colour validator, the three badge renderers and the
``CSS_NAMED_COLORS`` lookup table — was unreachable. The two existing badge test
modules errored during *collection*, which reports no failure count at all.

These tests pin down the three things that regression made invisible:

1. the module source parses and the module imports;
2. ``validate_hex_color`` is defined exactly once, and the surviving
   definition is the wrapped one whose default is ``DEFAULT_BADGE_COLOR``
   (not the stale ``"#2563eb"`` from the discarded header);
3. the public surface that rode on the broken import actually works.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import re
import xml.etree.ElementTree as ET

import pytest

from src.utils import badge_generator
from src.utils.badge_generator import (
    CSS_NAMED_COLORS,
    DEFAULT_BADGE_COLOR,
    generate_badge_svg,
    generate_svg_badge,
    get_badge_cache_key,
    has_pillow,
    has_reportlab,
    validate_hex_color,
)

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "src" / "utils" / "badge_generator.py"
)


@pytest.fixture(scope="module")
def module_tree() -> ast.Module:
    """Parsed AST of badge_generator.py, shared across the source-level tests."""
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"))


class TestModuleParses:
    """The import-time breakage itself."""

    def test_source_compiles(self) -> None:
        """The module source parses — this is what raised IndentationError."""
        compile(MODULE_PATH.read_text(encoding="utf-8"), str(MODULE_PATH), "exec")

    def test_module_imports(self) -> None:
        """A fresh import of the module succeeds."""
        importlib.reload(badge_generator)

    def test_no_stale_single_line_signature(self) -> None:
        """The discarded one-line signature is gone from the source."""
        source = MODULE_PATH.read_text(encoding="utf-8")
        assert 'default_color: str = "#2563eb"' not in source

    def test_no_consecutive_def_headers(self) -> None:
        """No ``def`` line is immediately followed by another ``def`` line.

        That adjacency is exactly the shape that produced the
        ``IndentationError``: a complete function header with the next
        statement at the same indentation.
        """
        lines = MODULE_PATH.read_text(encoding="utf-8").splitlines()
        offenders = [
            (index + 1, line)
            for index, line in enumerate(lines[:-1])
            if line.lstrip().startswith("def ")
            and line.rstrip().endswith(":")
            and lines[index + 1].lstrip().startswith("def ")
        ]
        assert offenders == []


class TestSingleDefinitions:
    """Every top-level name is defined once."""

    def test_validate_hex_color_defined_once(self, module_tree: ast.Module) -> None:
        definitions = [
            node
            for node in module_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "validate_hex_color"
        ]
        assert len(definitions) == 1

    def test_no_duplicate_top_level_functions(self, module_tree: ast.Module) -> None:
        """No top-level function name is defined twice."""
        names = [
            node.name
            for node in module_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        assert duplicates == []

    def test_imports_are_not_duplicated(self, module_tree: ast.Module) -> None:
        """The same bad merge duplicated ``re`` and ``logging`` at the top."""
        plain_imports: list[str] = []
        for node in module_tree.body:
            if isinstance(node, ast.Import):
                plain_imports.extend(alias.name for alias in node.names)
        duplicates = sorted(
            {name for name in plain_imports if plain_imports.count(name) > 1}
        )
        assert duplicates == []


class TestSurvivingSignature:
    """The definition that survived is the intended one."""

    def test_default_color_is_default_badge_color(self) -> None:
        signature = inspect.signature(validate_hex_color)
        assert signature.parameters["default_color"].default == DEFAULT_BADGE_COLOR

    def test_default_badge_color_value(self) -> None:
        assert DEFAULT_BADGE_COLOR == "#4f46e5"

    def test_parameter_names(self) -> None:
        signature = inspect.signature(validate_hex_color)
        assert list(signature.parameters) == ["color", "default_color"]

    def test_docstring_example_holds(self) -> None:
        """The docstring promises ``'#4f46e5'`` for an unrecognised name."""
        assert validate_hex_color("invalid_color_name") == "#4f46e5"


class TestValidateHexColorBehaviour:
    """The validator works now that it is reachable."""

    @pytest.mark.parametrize(
        "value",
        ["#fff", "#ffff", "#ffffff", "#ffffffff", "#123abc", "#0f0"],
    )
    def test_hex_forms_pass_through(self, value: str) -> None:
        assert validate_hex_color(value) == value

    def test_hex_is_lowercased(self) -> None:
        assert validate_hex_color("#ABCDEF") == "#abcdef"

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert validate_hex_color("  #ff0000  ") == "#ff0000"

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("red", "#ff0000"),
            ("lime", "#00ff00"),
            ("blue", "#0000ff"),
            ("black", "#000000"),
            ("white", "#ffffff"),
        ],
    )
    def test_css_named_colors_resolve(self, name: str, expected: str) -> None:
        assert validate_hex_color(name) == expected

    def test_named_color_is_case_insensitive(self) -> None:
        assert validate_hex_color("ReD") == validate_hex_color("red")

    @pytest.mark.parametrize("value", [None, "", "   ", 123, 4.5, [], {}, object()])
    def test_non_string_and_empty_fall_back(self, value: object) -> None:
        assert validate_hex_color(value) == DEFAULT_BADGE_COLOR  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "value", ["#ff", "#fffff", "#fffffff", "#gggggg", "ff0000", "rgb(1,2,3)"]
    )
    def test_malformed_hex_falls_back(self, value: str) -> None:
        assert validate_hex_color(value) == DEFAULT_BADGE_COLOR

    def test_explicit_default_is_honoured(self) -> None:
        assert validate_hex_color("nonsense", "#123456") == "#123456"

    def test_named_color_table_is_populated(self) -> None:
        assert len(CSS_NAMED_COLORS) >= 60
        assert all(isinstance(key, str) for key in CSS_NAMED_COLORS)

    def test_named_color_table_values_are_valid_hex(self) -> None:
        pattern = re.compile(r"^#(?:[0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$")
        invalid = {
            name: value
            for name, value in CSS_NAMED_COLORS.items()
            if not pattern.match(value)
        }
        assert invalid == {}

    def test_every_named_color_round_trips(self) -> None:
        """Each table entry validates to its own hex value."""
        mismatched = {
            name: validate_hex_color(name)
            for name, value in CSS_NAMED_COLORS.items()
            if validate_hex_color(name) != value
        }
        assert mismatched == {}


class TestReachableApi:
    """Symbols that were unreachable while the module failed to import."""

    def test_capability_probes_return_booleans(self) -> None:
        assert isinstance(has_pillow(), bool)
        assert isinstance(has_reportlab(), bool)

    def test_generate_badge_svg_is_well_formed_xml(self) -> None:
        svg = generate_badge_svg(student_name="Ada Lovelace", date="January 01, 2026")
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")

    def test_generate_badge_svg_contains_student_and_date(self) -> None:
        svg = generate_badge_svg(student_name="Ada Lovelace", date="January 01, 2026")
        assert "Ada Lovelace" in svg
        assert "January 01, 2026" in svg
        assert "Originality Verified" in svg

    def test_generate_badge_svg_escapes_markup_in_name(self) -> None:
        svg = generate_badge_svg(student_name="<script>alert(1)</script>")
        assert "<script>" not in svg
        ET.fromstring(svg)

    def test_generate_badge_svg_uses_validated_accent(self) -> None:
        svg = generate_badge_svg(accent_color="red")
        assert "#ff0000" in svg

    def test_generate_badge_svg_falls_back_on_bad_accent(self) -> None:
        svg = generate_badge_svg(accent_color="not-a-color")
        assert DEFAULT_BADGE_COLOR in svg

    def test_generate_svg_badge_shields_style(self) -> None:
        svg = generate_svg_badge("coverage", "95%")
        assert "coverage" in svg
        assert "95%" in svg
        assert svg.lstrip().startswith("<svg")

    def test_generate_svg_badge_validates_both_colors(self) -> None:
        svg = generate_svg_badge("build", "passing", "green", "blue")
        assert "#008000" in svg
        assert "#0000ff" in svg

    def test_badge_cache_key_is_deterministic(self) -> None:
        first = get_badge_cache_key("build", "passing", "#4c1")
        second = get_badge_cache_key("build", "passing", "#4c1")
        assert first == second
        assert len(first) == 32

    def test_badge_cache_key_varies_with_input(self) -> None:
        assert get_badge_cache_key("build", "passing", "#4c1") != get_badge_cache_key(
            "build", "failing", "#4c1"
        )
