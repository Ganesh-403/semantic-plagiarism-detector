"""Regression guard for issue #3901.

A merge of ``feature/translation-fallback-service`` into ``main`` removed the
``<<<<<<<`` / ``=======`` / ``>>>>>>>`` conflict markers from
``src/core/translator.py`` and ``src/core/cross_lingual.py`` but left the branch
name lines that follow them in place::

    src/core/translator.py:245:  feature/translation-fallback-service
    src/core/translator.py:309:  main

A bare indented ``feature/translation-fallback-service`` is not valid Python, so
both modules raised ``IndentationError`` at parse time.  Because
``src/core/parsers/cleaners.py`` imports ``translate_text`` from
``src.core.translator``, and ``src/__init__.py`` reaches that transitively, the
whole ``src`` package became unimportable and 302 test modules failed to
collect.

This module has two halves:

``TestNoConflictResidue``
    Walks every tracked Python file and asserts it parses and carries no
    conflict residue.  This is the part that would have caught #3901 before it
    reached ``main``, and it generalises to any future half-resolved merge.

``TestTranslatorSurfaceSurvivedTheMerge``
    Pins the specific behaviour that was at stake.  Both sides of the merge
    added real, non-overlapping functionality — ``translate_text_secondary``
    from the feature branch and ``translate_text_batch`` from ``main`` — and
    ``cross_lingual`` calls both.  Deleting either side would have "fixed" the
    syntax error while silently dropping a feature, so these tests assert that
    all three symbols are importable and wired together.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from typing import Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The three files that carried residue in #3901.  Named explicitly so a
# regression in any one of them fails with an obvious test name.
FILES_AFFECTED_BY_3901 = (
    "src/core/translator.py",
    "src/core/cross_lingual.py",
    "tests/core/test_cross_lingual.py",
)

# Real markers, as git writes them.  Anchored to the start of a line and
# requiring the full seven characters so a Markdown ``=======`` underline or a
# ``# ----`` banner in a docstring does not trip the scan.
_CONFLICT_MARKER = re.compile(r"^(<{7}|={7}|>{7})(\s|$)")

# The residue #3901 actually left behind: a bare branch name on its own line.
# ``git`` writes the branch name after the marker, so stripping only the marker
# characters leaves a line like ``` feature/translation-fallback-service```.
_BRANCH_RESIDUE = re.compile(
    r"^\s+(feature/[\w./-]+|main|master|HEAD|origin/[\w./-]+)\s*$"
)


def _tracked_python_files() -> list[Path]:
    """Return every ``.py`` file git tracks, as absolute paths.

    Using ``git ls-files`` rather than ``Path.rglob`` keeps generated trees --
    ``htmlcov/``, ``.venv/``, ``build/`` -- out of the scan without having to
    maintain an exclusion list that drifts.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        REPO_ROOT / name
        for name in result.stdout.split("\0")
        if name.endswith(".py")
    ]


def _iter_offending_lines(text: str) -> Iterator[tuple[int, str, str]]:
    """Yield ``(lineno, kind, line)`` for each line that looks like residue."""
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _CONFLICT_MARKER.match(line):
            yield lineno, "conflict marker", line
        elif _BRANCH_RESIDUE.match(line):
            yield lineno, "branch-name residue", line


def _read(path: Path) -> str:
    """Read source the way Python's own importer does.

    ``utf-8-sig`` strips a leading BOM.  Python's tokenizer accepts a UTF-8
    BOM at the start of a source file, but ``ast.parse`` on a ``str`` that
    still carries U+FEFF raises ``invalid non-printable character U+FEFF``.
    Two tracked test modules genuinely do start with a BOM and import fine,
    so reading them as plain ``utf-8`` here would report a parse failure that
    Python itself does not have.
    """
    return path.read_text(encoding="utf-8-sig")


class TestNoConflictResidue:
    """No tracked Python file carries merge-conflict residue, and all parse."""

    @pytest.mark.parametrize("relative_path", FILES_AFFECTED_BY_3901)
    def test_files_from_issue_3901_parse(self, relative_path: str) -> None:
        """The three files named in #3901 parse as Python.

        This is the narrowest possible statement of the bug: before the fix,
        ``ast.parse`` raised ``IndentationError`` on each of them.
        """
        path = REPO_ROOT / relative_path
        assert path.is_file(), f"{relative_path} is missing from the repository"

        try:
            ast.parse(_read(path), filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - only on regression
            pytest.fail(
                f"{relative_path} does not parse as Python: "
                f"line {exc.lineno}: {exc.msg}"
            )

    @pytest.mark.parametrize("relative_path", FILES_AFFECTED_BY_3901)
    def test_files_from_issue_3901_have_no_residue(
        self, relative_path: str
    ) -> None:
        """The three files named in #3901 carry no marker or branch residue."""
        path = REPO_ROOT / relative_path
        offenders = list(_iter_offending_lines(_read(path)))
        assert not offenders, (
            f"{relative_path} still contains merge-conflict residue: "
            + "; ".join(
                f"line {n} ({kind}): {line.strip()!r}" for n, kind, line in offenders
            )
        )

    def test_no_tracked_python_file_has_conflict_residue(self) -> None:
        """Repo-wide sweep -- the general guard against a repeat of #3901."""
        offenders: list[str] = []
        for path in _tracked_python_files():
            try:
                text = _read(path)
            except (OSError, UnicodeDecodeError):
                continue
            rel = path.relative_to(REPO_ROOT)
            offenders.extend(
                f"{rel}:{n}: {kind}: {line.strip()!r}"
                for n, kind, line in _iter_offending_lines(text)
            )

        assert not offenders, (
            "merge-conflict residue found in tracked Python files:\n"
            + "\n".join(offenders)
        )

    def test_every_tracked_python_file_parses(self) -> None:
        """Every tracked ``.py`` file is syntactically valid Python.

        #3901 was only visible at import time because the broken module sat
        under ``src/``.  A parse error anywhere else -- a script, a test helper
        -- is just as much a defect and just as cheap to catch here.
        """
        failures: list[str] = []
        for path in _tracked_python_files():
            try:
                source = _read(path)
            except (OSError, UnicodeDecodeError) as exc:
                failures.append(f"{path.relative_to(REPO_ROOT)}: unreadable: {exc}")
                continue
            try:
                ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}:{exc.lineno}: {exc.msg}"
                )

        assert not failures, "tracked Python files that do not parse:\n" + "\n".join(
            failures
        )


class TestResidueDetector:
    """The detector itself -- so a silent false negative cannot creep in."""

    @pytest.mark.parametrize(
        "line",
        [
            "<<<<<<< HEAD",
            "=======",
            ">>>>>>> feature/translation-fallback-service",
            "<<<<<<<",
        ],
    )
    def test_detects_real_conflict_markers(self, line: str) -> None:
        assert list(_iter_offending_lines(line)), f"missed marker: {line!r}"

    @pytest.mark.parametrize(
        "line",
        [
            " feature/translation-fallback-service",
            "    main",
            "  origin/main",
            "\tHEAD",
        ],
    )
    def test_detects_branch_name_residue(self, line: str) -> None:
        """Exactly the shape #3901 left behind."""
        assert list(_iter_offending_lines(line)), f"missed residue: {line!r}"

    @pytest.mark.parametrize(
        "line",
        [
            "# ===== SECTION =====",
            "x = a == b",
            "assert a >= b",
            '"""',
            "-------------------------",
            "    main = get_main()",
            "    return main",
            "from src.app import main",
            "def main():",
        ],
    )
    def test_does_not_flag_ordinary_code(self, line: str) -> None:
        """Docstring banners, comparisons and real uses of ``main`` are fine."""
        assert not list(_iter_offending_lines(line)), f"false positive: {line!r}"

    def test_flags_a_synthetic_half_resolved_merge(self, tmp_path: Path) -> None:
        """End-to-end: reconstruct the exact #3901 shape and confirm a catch."""
        broken = tmp_path / "broken.py"
        broken.write_text(
            "def f():\n"
            "    return 1\n"
            "\n"
            " feature/translation-fallback-service\n"
            "def g():\n"
            "    return 2\n"
            " main\n",
            encoding="utf-8",
        )
        offenders = list(_iter_offending_lines(_read(broken)))
        assert len(offenders) == 2
        assert [n for n, _, _ in offenders] == [4, 7]

        with pytest.raises(SyntaxError):
            ast.parse(_read(broken))


class TestTranslatorSurfaceSurvivedTheMerge:
    """Both sides of the merge are still present and still wired up.

    The tempting "fix" for #3901 was to delete one arm of the conflict.  These
    tests make that fail loudly.
    """

    def test_translator_exports_all_three_entry_points(self) -> None:
        from src.core import translator

        for name in (
            "translate_text",
            "translate_text_batch",
            "translate_text_secondary",
        ):
            assert hasattr(translator, name), (
                f"src.core.translator lost {name!r} -- one side of the "
                "#3901 merge was dropped instead of resolved"
            )
            assert callable(getattr(translator, name))

    def test_cross_lingual_imports_all_three(self) -> None:
        """``cross_lingual`` binds every symbol it calls at module scope."""
        from src.core import cross_lingual

        for name in (
            "translate_text",
            "translate_text_batch",
            "translate_text_secondary",
        ):
            assert hasattr(cross_lingual, name), (
                f"src.core.cross_lingual does not bind {name!r}; the merged "
                "import statement is incomplete"
            )

    def test_cross_lingual_import_is_a_single_statement(self) -> None:
        """The two competing imports were merged, not merely both kept.

        The pre-fix file had two separate ``from src.core.translator import``
        lines, one from each branch.  Leaving both would parse, but the second
        would shadow the first and drop ``translate_text_secondary``.
        """
        tree = ast.parse(_read(REPO_ROOT / "src/core/cross_lingual.py"))
        translator_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "src.core.translator"
            and node.level == 0
        ]
        assert len(translator_imports) == 1, (
            f"expected one import from src.core.translator, found "
            f"{len(translator_imports)} at lines "
            f"{[n.lineno for n in translator_imports]}"
        )

        imported = {alias.name for alias in translator_imports[0].names}
        assert imported == {
            "translate_text",
            "translate_text_batch",
            "translate_text_secondary",
        }, f"unexpected import list: {sorted(imported)}"

    def test_src_package_imports(self) -> None:
        """The headline symptom of #3901: ``import src`` raised."""
        import importlib

        module = importlib.import_module("src")
        assert module is not None
