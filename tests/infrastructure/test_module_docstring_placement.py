"""Repo-wide guard for shadowed module docstrings (Issue #3049).

Sixteen modules placed ``from __future__ import annotations`` *above* their
module docstring::

    from __future__ import annotations

    \"\"\"
    rate_limiter.py
    ---------------
    Security utility for API rate limiting ...
    \"\"\"

That is legal Python, but the string is no longer the docstring — it is a
discarded expression statement, and the module's ``__doc__`` is ``None``.

The consequences are quiet. ``help()``, ``pydoc``, IDE hovers and Sphinx
``automodule`` all show nothing, and ``scripts/check_docstring_coverage.py`` —
which the project runs as a gate — counts every one of them as undocumented
via ``ast.get_docstring()``.

The defect has been fixed one file at a time before (#2557, ``app_config``). It
keeps coming back because nothing checks for it, so this walks the whole tree
and reports every offender at once.

Like ``test_source_compiles.py``, this parses rather than imports: it must stay
fast and must not depend on optional runtime dependencies being installed.
"""

from __future__ import annotations

import ast
import pathlib

# Directories whose Python sources are first-party.
CHECKED_DIRECTORIES = (
    "app",
    "src",
    "evaluation",
    "scripts",
    "tests",
)

# Directories that never contain first-party source.
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _is_excluded(path: pathlib.Path) -> bool:
    """Return True when *path* lives in a directory we never check."""
    return any(part in EXCLUDED_PARTS for part in path.parts)


def iter_source_files() -> list[pathlib.Path]:
    """Return every first-party Python file to inspect."""
    files: list[pathlib.Path] = []

    for directory in CHECKED_DIRECTORIES:
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        files.extend(
            path
            for path in sorted(root.rglob("*.py"))
            if path.is_file() and not _is_excluded(path)
        )

    return files


SOURCE_FILES = iter_source_files()


def _has_shadowed_docstring(tree: ast.Module) -> bool:
    """Return True when a module-level string exists but is not the docstring.

    An ``__init__.py`` with no string at all is simply undocumented — a
    different concern, and not one this guard should fail on. What it catches
    is a module that clearly *intended* a docstring and lost it to statement
    ordering.
    """
    if ast.get_docstring(tree) is not None:
        return False

    return any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for node in tree.body
    )


def collect_shadowed_docstrings() -> dict[str, int]:
    """Return ``{relative_path: line_of_the_orphaned_string}`` for offenders."""
    offenders: dict[str, int] = {}

    for path in SOURCE_FILES:
        relative = path.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            # test_source_compiles.py owns unparseable files.
            continue

        if not _has_shadowed_docstring(tree):
            continue

        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                offenders[relative] = node.lineno
                break

    return offenders


def test_source_files_were_discovered():
    """Guard the guard: an empty file list would make this suite vacuous."""
    assert len(SOURCE_FILES) > 100, (
        f"expected to discover the project sources, found {len(SOURCE_FILES)} "
        "files - has the layout changed?"
    )


def test_no_module_docstring_is_shadowed():
    """Every module-level docstring must be the first statement."""
    offenders = collect_shadowed_docstrings()

    assert not offenders, (
        "These modules have a docstring that Python does not see, because a "
        "statement precedes it (usually `from __future__ import annotations`). "
        "Move the docstring to the top of the file:\n"
        + "\n".join(f"  {path}:{line}" for path, line in sorted(offenders.items()))
    )


def test_detects_a_shadowed_docstring():
    """Guard the guard: the detector must actually fire on the bad shape."""
    tree = ast.parse('from __future__ import annotations\n\n"""Docstring."""\n')

    assert ast.get_docstring(tree) is None
    assert _has_shadowed_docstring(tree) is True


def test_accepts_a_correctly_placed_docstring():
    """The correct ordering must not be reported."""
    tree = ast.parse('"""Docstring."""\n\nfrom __future__ import annotations\n')

    assert ast.get_docstring(tree) == "Docstring."
    assert _has_shadowed_docstring(tree) is False


def test_accepts_a_shebang_before_the_docstring():
    """A shebang is a comment, so it does not displace the docstring."""
    tree = ast.parse('#!/usr/bin/env python3\n"""Docstring."""\nimport os\n')

    assert ast.get_docstring(tree) == "Docstring."
    assert _has_shadowed_docstring(tree) is False


def test_ignores_a_module_with_no_string_at_all():
    """An undocumented module is a different concern and not an offender."""
    tree = ast.parse("import os\n")

    assert _has_shadowed_docstring(tree) is False


def test_ignores_a_string_that_is_merely_a_later_statement():
    """A bare string after real code is not a displaced docstring...

    ...unless the module has no docstring, which is exactly the case this
    guard reports. This pins the distinction so the rule stays legible.
    """
    documented = ast.parse('"""Real docstring."""\nimport os\n"""stray"""\n')

    assert _has_shadowed_docstring(documented) is False
