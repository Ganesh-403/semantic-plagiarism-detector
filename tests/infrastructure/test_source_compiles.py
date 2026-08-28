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

"""Repo-wide compile guard (Issue #2198).

Three separate files were sitting on ``main`` at the same time in a state
where Python could not parse them at all:

* ``app/streamlit_app.py``  - orphaned merge leftovers (#2198)
* ``src/utils/file_parser.py`` - stray bodiless ``def`` (#2197)
* ``src/security/mime_validator.py`` - dedented ``except`` clause (#2196)

None of them were caught, because a module that fails to import mostly shows
up as a *collection* error in one test file, which is easy to miss in a long
CI log, and because some suites were exercising pasted copies of the source
rather than importing the real module.

This test compiles every tracked Python file under ``app/``, ``src/``,
``evaluation/``, ``scripts/`` and ``tests/`` and reports **all** failures at
once, so a botched merge fails loudly and points at every broken file in a
single run.

It deliberately uses ``compile()`` rather than ``import``: it must stay fast
and must not depend on optional runtime dependencies (torch, streamlit,
redis) being installed.
"""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Directories whose Python sources must always parse.
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


def _is_excluded(path: pathlib.Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def iter_source_files() -> list[pathlib.Path]:
    """Return every first-party Python file that must parse."""
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

# Files known not to parse on ``main`` at the time this guard was added, each
# with an open fix. They are allow-listed so this test can land independently
# of those PRs instead of being blocked behind them.
#
# Remove an entry as soon as its fix merges. A stale entry is harmless (the
# subset assertion below still passes), but it hides a real regression in that
# file, so please keep this list empty.
KNOWN_BROKEN: dict[str, str] = {
    "src/utils/file_parser.py": "#2197",
    "src/security/mime_validator.py": "#2196",
}


def test_source_files_were_discovered():
    """Guard the guard: an empty file list would make this suite vacuous."""
    assert len(SOURCE_FILES) > 100, (
        f"expected to discover the project sources, found {len(SOURCE_FILES)} "
        "files - has the layout changed?"
    )


def collect_syntax_failures() -> dict[str, str]:
    """Compile every source file and return ``{relative_path: reason}``."""
    failures: dict[str, str] = {}

    for path in SOURCE_FILES:
        relative = path.relative_to(REPO_ROOT).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            failures[relative] = f"not valid UTF-8 ({exc})"
            continue

        try:
            compile(source, relative, "exec")
        except SyntaxError as exc:
            # IndentationError and TabError are SyntaxError subclasses.
            failures[relative] = f"line {exc.lineno}: {type(exc).__name__}: {exc.msg}"

    return failures


def test_every_source_file_compiles():
    """Every first-party Python file must be syntactically valid.

    Failures are collected and reported together rather than aborting on the
    first one, so a merge that breaks several files shows all of them at once.
    """
    failures = collect_syntax_failures()
    unexpected = {
        path: reason for path, reason in failures.items() if path not in KNOWN_BROKEN
    }

    assert not unexpected, "Python files that do not parse:\n" + "\n".join(
        f"  - {path}: {reason}" for path, reason in sorted(unexpected.items())
    )


def test_streamlit_entry_point_compiles():
    """``app/streamlit_app.py`` is the application entry point.

    When it does not parse, ``streamlit run`` fails outright and there is no
    degraded mode to fall back to.
    """
    path = REPO_ROOT / "app" / "streamlit_app.py"
    assert path.is_file(), "app/streamlit_app.py is missing"

    compile(path.read_text(encoding="utf-8"), "app/streamlit_app.py", "exec")


@pytest.mark.parametrize("relative_path", sorted(KNOWN_BROKEN))
def test_known_broken_files_are_tracked(relative_path: str):
    """Every allow-listed file must still exist and name an open issue.

    This stops the allow-list from silently accumulating paths that no longer
    exist, and keeps each entry traceable to the fix that should remove it.
    """
    assert (REPO_ROOT / relative_path).is_file(), (
        f"{relative_path} is allow-listed in KNOWN_BROKEN but does not exist - "
        "remove the entry"
    )
    assert KNOWN_BROKEN[relative_path].startswith("#")
