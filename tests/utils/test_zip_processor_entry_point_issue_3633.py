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
tests/utils/test_zip_processor_entry_point_issue_3633.py
--------------------------------------------------------
Regression tests for Issue #3633.

``src/utils/zip_processor.py`` used to define ``process_zip_file`` twice. The
second definition shadowed the first and delegated to an ``iter_zip_files``
symbol that did not exist, so every ZIP upload died with::

    NameError: name 'iter_zip_files' is not defined

The shadowed generator also read and wrote a ``used_filenames`` name it never
initialised, and the surviving wrapper had silently dropped the
``skip_corrupted`` flag from the public signature.

These tests pin the module's contract: a streaming generator, a dict-returning
wrapper, one shared ``skip_corrupted`` behaviour, and per-archive collision
tracking that actually works.
"""

import inspect
import io
import types
import zipfile
from unittest import mock

import pytest

from src.utils import zip_processor
from src.utils.zip_processor import iter_zip_files, process_zip_file


def _build_zip(entries: dict, compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    """Build an in-memory ZIP archive from a ``{name: bytes}`` mapping."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _encrypted_infolist(encrypted: str, plain: str) -> list:
    """Build the ``infolist`` an archive with one encrypted member would report.

    ``ZipFile.writestr`` clears the encryption flag bit on write, so the only
    way to exercise the encrypted-member branch is to patch ``infolist``.
    """
    encrypted_info = zipfile.ZipInfo(encrypted)
    encrypted_info.flag_bits = 0x1

    plain_info = zipfile.ZipInfo(plain)
    plain_info.flag_bits = 0x0

    return [encrypted_info, plain_info]


def _fake_read(_self, name_or_info):
    """Stand-in for ``ZipFile.read`` that serves the patched infolist."""
    name = (
        name_or_info.filename
        if isinstance(name_or_info, zipfile.ZipInfo)
        else name_or_info
    )
    return b"classified" if "secret" in name else b"readable"


class TestModuleSurface:
    """The module must expose exactly one of each entry point."""

    def test_iter_zip_files_is_defined(self):
        assert hasattr(zip_processor, "iter_zip_files")

    def test_iter_zip_files_is_a_generator_function(self):
        assert inspect.isgeneratorfunction(zip_processor.iter_zip_files)

    def test_process_zip_file_is_not_a_generator_function(self):
        assert not inspect.isgeneratorfunction(zip_processor.process_zip_file)

    def test_process_zip_file_is_defined_once(self):
        """A duplicate ``def`` is what caused this bug; catch it in the source."""
        source = inspect.getsource(zip_processor)
        assert source.count("\ndef process_zip_file(") == 1

    @pytest.mark.parametrize("func_name", ["iter_zip_files", "process_zip_file"])
    def test_both_entry_points_accept_skip_corrupted(self, func_name):
        signature = inspect.signature(getattr(zip_processor, func_name))
        assert "skip_corrupted" in signature.parameters
        assert signature.parameters["skip_corrupted"].default is False


class TestStreamingContract:
    """``iter_zip_files`` yields entries one at a time."""

    def test_returns_a_generator_object(self):
        zip_bytes = _build_zip({"a.txt": b"first"})
        assert isinstance(iter_zip_files(zip_bytes), types.GeneratorType)

    def test_yields_name_and_bytes_pairs(self):
        zip_bytes = _build_zip({"a.txt": b"first", "b.md": b"# second"})

        entries = list(iter_zip_files(zip_bytes))

        assert sorted(entries) == [("a.txt", b"first"), ("b.md", b"# second")]

    def test_is_lazy_and_does_not_drain_the_archive_up_front(self):
        zip_bytes = _build_zip({f"doc{i}.txt": b"x" * 32 for i in range(5)})

        generator = iter_zip_files(zip_bytes)
        first = next(generator)

        assert first[0].endswith(".txt")
        assert len(list(generator)) == 4

    def test_validation_errors_surface_on_iteration(self):
        """A generator defers its body, so the guard fires when consumed."""
        generator = iter_zip_files(b"")

        with pytest.raises(ValueError, match="ZIP archive is empty"):
            next(generator)


class TestWrapperContract:
    """``process_zip_file`` materialises the generator into a dict."""

    def test_returns_a_dict(self):
        zip_bytes = _build_zip({"a.txt": b"first"})
        assert process_zip_file(zip_bytes) == {"a.txt": b"first"}

    def test_matches_the_generator_output(self):
        zip_bytes = _build_zip(
            {
                "report.pdf": b"%PDF-1.4 body",
                "notes.md": b"# notes",
                "data.csv": b"a,b\n1,2",
            }
        )

        assert process_zip_file(zip_bytes) == dict(iter_zip_files(zip_bytes))

    def test_rejects_empty_input_eagerly(self):
        with pytest.raises(ValueError, match="ZIP archive is empty"):
            process_zip_file(b"")

    def test_rejects_non_zip_input(self):
        with pytest.raises(ValueError, match="missing ZIP header signature"):
            process_zip_file(b"not a zip file at all")

    def test_flattens_nested_directories_to_basenames(self):
        zip_bytes = _build_zip({"folder/subfolder/essay.pdf": b"body"})

        assert process_zip_file(zip_bytes) == {"essay.pdf": b"body"}

    def test_ignores_unsupported_extensions(self):
        zip_bytes = _build_zip(
            {
                "keep.txt": b"kept",
                "drop.png": b"\x89PNG binary",
                "drop.sh": b"#!/bin/sh",
            }
        )

        assert set(process_zip_file(zip_bytes)) == {"keep.txt"}

    def test_skips_empty_members(self):
        zip_bytes = _build_zip({"empty.txt": b"", "full.txt": b"content"})

        assert set(process_zip_file(zip_bytes)) == {"full.txt"}


class TestFilenameCollisionTracking:
    """``used_filenames`` used to be an undefined name; prove it now works."""

    def test_colliding_basenames_are_disambiguated(self):
        zip_bytes = _build_zip(
            {
                "draft/essay.txt": b"first draft",
                "final/essay.txt": b"second draft",
            }
        )

        result = process_zip_file(zip_bytes)

        assert len(result) == 2
        assert "essay.txt" in result
        assert set(result.values()) == {b"first draft", b"second draft"}

    def test_no_member_is_silently_overwritten(self):
        zip_bytes = _build_zip(
            {f"dir{i}/same.txt": f"body {i}".encode() for i in range(4)}
        )

        result = process_zip_file(zip_bytes)

        assert len(result) == 4
        assert len(set(result.values())) == 4

    def test_collision_state_does_not_leak_between_calls(self):
        """Each archive starts with a clean name registry."""
        zip_bytes = _build_zip({"essay.txt": b"body"})

        first = process_zip_file(zip_bytes)
        second = process_zip_file(zip_bytes)

        assert first == second == {"essay.txt": b"body"}


class TestSkipCorruptedIsForwarded:
    """The wrapper must pass the flag through, not drop it."""

    def test_encrypted_member_aborts_by_default(self):
        zip_bytes = _build_zip({"secret.txt": b"classified"})
        infolist = _encrypted_infolist("secret.txt", "public.txt")

        with mock.patch("zipfile.ZipFile.infolist", return_value=infolist):
            with pytest.raises(
                ValueError, match="encrypted ZIP files are not supported"
            ):
                process_zip_file(zip_bytes)

    def test_encrypted_member_is_skipped_when_requested(self):
        zip_bytes = _build_zip({"secret.txt": b"classified"})
        infolist = _encrypted_infolist("secret.txt", "public.txt")

        with mock.patch("zipfile.ZipFile.infolist", return_value=infolist), mock.patch(
            "zipfile.ZipFile.read", _fake_read
        ):
            result = process_zip_file(zip_bytes, skip_corrupted=True)

        assert result == {"public.txt": b"readable"}

    def test_generator_and_wrapper_agree_on_skip_corrupted(self):
        zip_bytes = _build_zip({"secret.txt": b"classified"})
        infolist = _encrypted_infolist("secret.txt", "public.txt")

        with mock.patch("zipfile.ZipFile.infolist", return_value=infolist), mock.patch(
            "zipfile.ZipFile.read", _fake_read
        ):
            streamed = dict(iter_zip_files(zip_bytes, skip_corrupted=True))
            materialised = process_zip_file(zip_bytes, skip_corrupted=True)

        assert streamed == materialised


class TestSecurityGuardsStillHold:
    """These guards were unreachable while the entry point was broken."""

    @pytest.mark.parametrize(
        "malicious_path",
        ["../../etc/passwd", "../evil.txt", "/etc/passwd", "..\\evil.txt"],
    )
    def test_path_traversal_entries_are_rejected(self, malicious_path):
        zip_bytes = _build_zip({malicious_path: b"payload"})

        with pytest.raises(ValueError, match="Malicious path traversal detected"):
            process_zip_file(zip_bytes)

    def test_decompression_ratio_limit_is_enforced(self):
        # Highly compressible payload well past the 100:1 ratio limit.
        zip_bytes = _build_zip({"bomb.txt": b"0" * (2 * 1024 * 1024)})

        with pytest.raises(ValueError, match="Zip Bomb detected"):
            process_zip_file(zip_bytes)

    def test_executable_double_extension_is_rejected(self):
        zip_bytes = _build_zip({"invoice.pdf.exe": b"MZ binary"})

        assert process_zip_file(zip_bytes) == {}
