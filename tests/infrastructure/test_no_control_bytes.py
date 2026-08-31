"""Reject stray control bytes in tracked Python sources (Issue #2556).

``src/core/cross_lingual.py`` sat on ``main`` with two literal ``0x00`` bytes
embedded in an f-string. The author meant to write the two-character escape
``\\x00``; what landed was the raw byte. CPython refuses to compile *any*
source containing a NUL, so the module -- and the six test modules plus five
application modules that import it -- could not be loaded at all.

The existing compile guard in ``test_source_compiles.py`` does catch this, but
only as::

    src/core/cross_lingual.py: line None: SyntaxError: source code string
    cannot contain null bytes

``line None`` is not enough to find a byte that is invisible in every diff view
and most editors. This guard reports the exact line, column and byte offset
instead, and additionally catches control bytes that are *not* fatal to the
parser but are almost never intentional inside source (an unescaped ESC, a
stray form feed left by a bad paste, a lone carriage return).

Deliberately allowed: tab, newline and carriage return, since those are
ordinary whitespace. Everything else in the C0 range, plus DEL, is rejected.
"""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

CHECKED_DIRECTORIES = (
    "app",
    "src",
    "evaluation",
    "scripts",
    "tests",
)

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

# Tab, line feed and carriage return are legitimate whitespace.
ALLOWED_CONTROL_BYTES = {0x09, 0x0A, 0x0D}

# C0 controls plus DEL.
DISALLOWED_BYTES = frozenset(set(range(0x00, 0x20)) - ALLOWED_CONTROL_BYTES | {0x7F})


def _is_excluded(path: pathlib.Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def iter_source_files() -> list[pathlib.Path]:
    """Return every first-party Python file that must be free of control bytes."""
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


def _describe_offset(data: bytes, offset: int) -> str:
    """Render a byte offset as ``line L, column C`` for a human reader."""
    line = data.count(b"\n", 0, offset) + 1
    line_start = data.rfind(b"\n", 0, offset) + 1
    return f"line {line}, column {offset - line_start + 1}"


def find_control_bytes(data: bytes) -> list[tuple[int, int]]:
    """Return ``(offset, byte_value)`` for every disallowed byte in ``data``."""
    return [
        (offset, byte) for offset, byte in enumerate(data) if byte in DISALLOWED_BYTES
    ]


def test_source_files_were_discovered():
    """Guard the guard: an empty file list would make this suite vacuous."""
    assert len(SOURCE_FILES) > 100, (
        f"expected to discover the project sources, found {len(SOURCE_FILES)} "
        "files - has the layout changed?"
    )


def test_no_source_file_contains_control_bytes():
    """No tracked Python source may contain a NUL or other stray control byte.

    Failures are collected and reported together, each with the exact line,
    column and byte value, so a bad paste that touches several files shows all
    of them at once.
    """
    failures: list[str] = []

    for path in SOURCE_FILES:
        data = path.read_bytes()
        found = find_control_bytes(data)
        if not found:
            continue

        relative = path.relative_to(REPO_ROOT).as_posix()
        for offset, byte in found[:5]:
            failures.append(
                f"  - {relative}: {_describe_offset(data, offset)} "
                f"(byte offset {offset}) contains 0x{byte:02X}; "
                f"write it as the escape \\x{byte:02x} instead"
            )
        if len(found) > 5:
            failures.append(f"  - {relative}: ...and {len(found) - 5} more")

    assert not failures, (
        "Python files containing disallowed control bytes:\n" + "\n".join(failures)
    )


def test_cross_lingual_has_no_null_bytes():
    """Direct regression guard for the file from Issue #2556."""
    path = REPO_ROOT / "src" / "core" / "cross_lingual.py"
    assert path.is_file(), "src/core/cross_lingual.py is missing"

    assert b"\x00" not in path.read_bytes(), (
        "src/core/cross_lingual.py contains a raw NUL byte again - the "
        "separator in TranslationMemoryCache._build_key must be written as the "
        "escape sequence \\x00, not as a literal byte"
    )


def test_all_sources_decode_as_utf8():
    """A control-byte check is meaningless on a file we cannot decode."""
    failures = []
    for path in SOURCE_FILES:
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            failures.append(f"  - {path.relative_to(REPO_ROOT).as_posix()}: {exc}")

    assert not failures, "Python files that are not valid UTF-8:\n" + "\n".join(
        failures
    )


# ---------------------------------------------------------------------------
# Guard-the-guard: the detector itself must work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload, expected",
    [
        (b"x = 1\n", 0),
        (b"x = '\x00'\n", 1),
        (b"x = '\x00\x00'\n", 2),
        (b"tabs\tand\nnewlines\r\n", 0),
        (b"esc \x1b[0m\n", 1),
        (b"form feed \x0c\n", 1),
        (b"delete \x7f\n", 1),
    ],
)
def test_find_control_bytes_detects_expected_payloads(payload: bytes, expected: int):
    assert len(find_control_bytes(payload)) == expected


def test_describe_offset_reports_line_and_column():
    data = b"first\nsecond\x00\n"
    offset = data.index(b"\x00")

    assert _describe_offset(data, offset) == "line 2, column 7"
