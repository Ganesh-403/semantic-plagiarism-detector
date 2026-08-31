"""Tests for secure .tar.gz and .tar.bz2 extraction."""

import io
import tarfile
from unittest.mock import patch

import pytest

from src.utils.tar_processor import (
    ALLOWED_TAR_MEMBER_EXTENSIONS,
    MAX_ABSOLUTE_UNCOMPRESSED_SIZE,
    MAX_SINGLE_FILE_SIZE,
    MAX_TOTAL_DECOMPRESSED_SIZE,
    process_tar_file,
)


def make_tar_bytes(files: dict[str, bytes], compression: str = "gz") -> bytes:
    """Create an in-memory compressed TAR archive."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=f"w:{compression}") as archive:
        for filename, content in files.items():
            info = tarfile.TarInfo(filename)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_process_tar_gzip_extracts_supported_documents():
    archive = make_tar_bytes(
        {
            "essay.txt": b"plain text",
            "report.md": b"# markdown",
            "data.csv": b"a,b\n1,2",
            "image.png": b"not supported",
        }
    )

    result = process_tar_file(archive)

    assert result["essay.txt"] == b"plain text"
    assert result["report.md"] == b"# markdown"
    assert result["data.csv"] == b"a,b\n1,2"
    assert "image.png" not in result


def test_process_tar_bzip2_extracts_supported_documents():
    archive = make_tar_bytes({"essay.txt": b"bzip2 content"}, compression="bz2")

    result = process_tar_file(archive)

    assert result == {"essay.txt": b"bzip2 content"}


def test_process_tar_empty_archive_input():
    with pytest.raises(ValueError, match="TAR archive is empty"):
        process_tar_file(b"")


def test_process_tar_rejects_uncompressed_tar():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("essay.txt")
        content = b"content"
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    with pytest.raises(ValueError, match="Only .tar.gz and .tar.bz2"):
        process_tar_file(buffer.getvalue())


@pytest.mark.parametrize(
    "malicious_path",
    [
        "../../etc/passwd",
        "../evil.py",
        "/etc/passwd",
        r"..\evil.py",
        r"C:\evil.txt",
    ],
)
def test_process_tar_rejects_path_traversal(malicious_path):
    archive = make_tar_bytes({malicious_path: b"malicious"})

    with pytest.raises(
        ValueError, match="Malicious path traversal detected in TAR archive entry"
    ):
        process_tar_file(archive)


def test_process_tar_flattens_nested_names_and_resolves_collisions():
    archive = make_tar_bytes(
        {
            "folder/assignment.txt": b"first",
            "other/assignment.txt": b"second",
        }
    )

    result = process_tar_file(archive)

    assert result["assignment.txt"] == b"first"
    assert result["assignment_1.txt"] == b"second"


def test_process_tar_skips_non_regular_members():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        directory = tarfile.TarInfo("folder/")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)

        symlink = tarfile.TarInfo("linked.txt")
        symlink.type = tarfile.SYMTYPE
        symlink.linkname = "/etc/passwd"
        archive.addfile(symlink)

        content = b"safe"
        info = tarfile.TarInfo("safe.txt")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    assert process_tar_file(buffer.getvalue()) == {"safe.txt": b"safe"}


def test_process_tar_single_file_limit():
    member = tarfile.TarInfo("huge.txt")
    member.size = MAX_SINGLE_FILE_SIZE + 1

    with patch("tarfile.TarFile.getmembers", return_value=[member]):
        archive = make_tar_bytes({"safe.txt": b"safe"})
        with pytest.raises(ValueError, match="single file decompression safety limit"):
            process_tar_file(archive)


def test_process_tar_total_size_limit():
    members = []
    for name, size in (
        ("one.txt", MAX_TOTAL_DECOMPRESSED_SIZE // 2),
        ("two.txt", MAX_TOTAL_DECOMPRESSED_SIZE // 2),
        ("three.txt", 1),
    ):
        member = tarfile.TarInfo(name)
        member.size = size
        members.append(member)

    archive = make_tar_bytes({"safe.txt": b"safe"})
    with patch("tarfile.TarFile.getmembers", return_value=members):
        with pytest.raises(ValueError, match="total decompressed size"):
            process_tar_file(archive)


def test_process_tar_absolute_limit():
    member = tarfile.TarInfo("huge.txt")
    member.size = MAX_ABSOLUTE_UNCOMPRESSED_SIZE + 1

    archive = make_tar_bytes({"safe.txt": b"safe"})
    with patch("tarfile.TarFile.getmembers", return_value=[member]):
        with pytest.raises(ValueError, match="Tar Bomb detected"):
            process_tar_file(archive)


def test_process_tar_decompression_ratio_limit():
    # A tiny compressed archive mocked as having a very large declared payload.
    member = tarfile.TarInfo("bomb.txt")
    member.size = 101 * len(make_tar_bytes({"safe.txt": b"x"}))

    archive = make_tar_bytes({"safe.txt": b"x"})
    with patch("tarfile.TarFile.getmembers", return_value=[member]):
        with pytest.raises(ValueError, match="Decompression ratio exceeds security limit"):
            process_tar_file(archive)


def test_allowed_tar_member_extensions():
    assert ALLOWED_TAR_MEMBER_EXTENSIONS == {
        ".pdf",
        ".docx",
        ".txt",
        ".rtf",
        ".csv",
        ".odt",
        ".md",
    }
